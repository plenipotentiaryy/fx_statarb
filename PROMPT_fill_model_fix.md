# ЗАДАЧА: Исправить катастрофический баг fill_probability в ExecutionSimulator

## Прочитай перед правками
- `execution_stress.py` — метод `simulate_order()`, строки 151–169
- `step4a_backtest.py` — строки ~440–460 (построение `_sim_cols`) и ~1319–1330 (создание `_exec_sim`)

---

## Диагноз — почему real_trades = 0 несмотря на raw_trades > 0

### Симптом
Smoke-test: `raw_trades` от 1 до 177 по парам → `real_trades = 0` по всем парам.
`exit_reason` у всех записей = `LIMIT_NOT_FILLED`.

### Корневая причина

В `execution_stress.py`, строка 154:
```python
fill_probability = float(np.exp(-float(self.fill_kappa) * distance_to_mid / denom))
```

Где:
- `mid = execution_mid` = **`spread_vwap_mtf`** (15-минутная скользящая VWAP Kalman-спреда)
- `spread_vwap_mtf ≈ среднее спреда ≈ 0` (Kalman-спред осциллирует вокруг нуля по определению)
- `requested_price = df["spread"].iloc[next_i]` = спред при входе, где |z| ≥ 1.8
- **→ `|requested - mid|` ≈ 1.8σ, `denom = max(|mid|, 1e-12) ≈ 1e-12`**
- **→ `fill_probability = exp(-8 × 1.8σ / 1e-12) ≈ 0` для 100% сделок**

**Почему это баг, а не фича**: ExecutionSimulator проектировался под одиночные активы с ценой
1.0–1000 (акции, форекс-котировки). Там `vwap ≈ close` в пределах 0.1% → `distance/denom ≈ 0`.
Kalman-спред — синтетический инструмент с ценой вокруг 0. Нормализация на `|mid|` взрывается
при `mid → 0`, что разрушает всю логику вероятности исполнения.

**Числовой пример (реальные порядки для EURUSD/GBPUSD):**
```
spread_at_signal = -0.003   (z = -1.8, σ = 0.0017)
spread_vwap_15m  = -0.0002  (rolling mean, ≈ 0)
denom            = 0.0002
fill_probability = exp(-8 × |-0.003 - (-0.0002)| / 0.0002)
                 = exp(-8 × 0.0028 / 0.0002)
                 = exp(-112) ≈ 0
```

Вывод: **одна строка в инициализации ExecutionSimulator убивает все сделки**.

---

## Единственное изменение: `step4a_backtest.py`

Найди инициализацию `_exec_sim` (строка ~1319). Сейчас:

```python
_exec_sim = ExecutionSimulator(
    base_spread=COST_MAKER,
    spread_gamma=1.5,
    panic_multiplier=3.0,
    entry_delay=1,
    slippage_variance=0.0,
    fill_kappa=8.0,
    passive=True,
    use_vwap=True,          # ← БАГ: spread_vwap_mtf ≈ 0 → нормализация взрывается
    random_seed=42,
    toxicity_threshold=None,
)
```

Замени на:

```python
_exec_sim = ExecutionSimulator(
    base_spread=COST_MAKER,
    spread_gamma=1.5,
    panic_multiplier=3.0,
    entry_delay=1,
    slippage_variance=0.0,
    fill_kappa=8.0,
    passive=True,
    use_vwap=False,          # ИСПРАВЛЕНО: Kalman-спред ≈ 0 → VWAP-нормализация = деление на ~0
                             # Без VWAP: execution_mid = close = spread_at_fill_bar
                             # requested_price = spread_at_fill_bar (тот же бар, next_i)
                             # → distance_to_mid ≈ 0 → fill_probability ≈ 1.0 (корректно)
                             # Реальное трение: effective_spread × vol_adjustment (в fill_price)
    random_seed=42,
    toxicity_threshold=None,
)
```

**Почему это правильно экономически**: на ликвидных FX парах (EURUSD bid-ask ≈ 0.3–0.5 pips)
лимитный ордер на текущем спреде исполняется с вероятностью >95%. Трение сделки уже
учитывается через `effective_spread` в `fill_price` (строка 173 execution_stress.py):
```python
fill_price = mid + side * (spread_penalty + slippage)
```
И через `COST_MAKER/COST_TAKER` в PnL-расчёте backtester'а. Двойной учёт трения через
fill_probability на синтетическом спреде — архитектурная несовместимость, не фича.

---

## Что НЕ менять в этом промпте

- `passive=True` — оставить. Это контролирует добавление spread_penalty при расчёте fill_price.
  С `use_vwap=False` и `passive=True`: distance_to_mid ≈ 0 → fill_probability ≈ 1.0 → ордер
  заполняется, но fill_price = mid + 0.5 × effective_spread (честный half-spread penalty).
- `fill_kappa=8.0` — оставить. Теперь он применяется к нулевой дистанции → нет эффекта.
- `spread_gamma`, `panic_multiplier` — оставить, влияют только на effective_spread.
- `entry_delay=1` — оставить. Заполнение на следующем баре = реалистично.
- `_sim_cols` в `backtest_pair()` — не трогать. `spread_vwap_mtf` остаётся в prices_df для
  других расчётов (vol_ratio и т.д.), просто больше не используется как reference mid.

---

## Верификация

### Шаг 1: синтаксис
```bash
python -c "import ast; ast.parse(open('step4a_backtest.py').read()); print('step4a_backtest.py OK')"
```

### Шаг 2: unit-тест fill_probability
```bash
python -c "
import pandas as pd, numpy as np
from execution_stress import ExecutionSimulator

# Воспроизводим точно сценарий Kalman-спреда
idx = pd.date_range('2022-01-01', periods=100, freq='1min', tz='UTC')
spread_values = np.linspace(-0.003, 0.003, 100)  # спред осцилирует около 0
vwap_values = np.full(100, -0.0002)              # 15m VWAP ≈ 0

prices = pd.DataFrame({
    'close': spread_values,
    'vwap': vwap_values,   # ← это и был источник бага
}, index=idx)

# Старая конфигурация (баг)
sim_bug = ExecutionSimulator(fill_kappa=8.0, passive=True, use_vwap=True, random_seed=42)
sim_bug.precompute(prices)
result_bug = sim_bug.simulate_order(prices, signal_index=10, side=1,
                                     requested_price=spread_values[11])
print(f'use_vwap=True  fill_prob={result_bug.fill_probability:.6f}  status={result_bug.status}')

# Новая конфигурация (исправлено)
sim_fix = ExecutionSimulator(fill_kappa=8.0, passive=True, use_vwap=False, random_seed=42)
sim_fix.precompute(prices)
result_fix = sim_fix.simulate_order(prices, signal_index=10, side=1,
                                     requested_price=spread_values[11])
print(f'use_vwap=False fill_prob={result_fix.fill_probability:.6f}  status={result_fix.status}')

# Ожидаем:
# use_vwap=True  fill_prob≈0.000000  status=MISSED (был баг)
# use_vwap=False fill_prob≈1.000000  status=FILLED (исправлено)
"
```

### Шаг 3: smoke-test с подробной диагностикой
```bash
BACKTEST_RECENT_BARS=43200 python step4a_backtest.py 2>&1 | grep -E "trades=|blocks:|real_trades|LIMIT_NOT_FILLED|vwap_mtf|live_corr|rvol|rvr|innov_var"
```

### Шаг 4: что ожидать в выводе

**До исправления:**
```
blocks: macro_entry_blocked=... vwap_mtf=... live_corr=...
        exec_LIMIT_NOT_FILLED=177  ← все raw_trades умирают здесь
real_trades=0
```

**После исправления:**
```
blocks: macro_entry_blocked=... vwap_mtf=... live_corr=...
        exec_LIMIT_NOT_FILLED=0   ← больше не умирают
real_trades > 0                   ← появляются реальные сделки
```

---

## Если после исправления real_trades всё ещё мало

Смотреть на блок диагностики `blocks:` — что стало следующим по величине блокировщиком.

**Вероятный следующий блокировщик — `live_corr`** (`LIVE_CORR_MIN = 0.45`):
```python
# config.py строка 140:
LIVE_CORR_MIN = 0.45   # → если live_corr блокирует > 30% — ослабить до:
LIVE_CORR_MIN = 0.35
```
Обоснование: `live_corr` — 60-минутная rolling корреляция ног. FX пары временно теряют
корреляцию в боковых рынках. 0.35 — разумный минимум; ниже уже риск ложных коинтеграций.

**Вероятный следующий блокировщик — `vwap_mtf`** (если диагностика покажет > 20%):
```python
# config.py строка 112:
USE_VWAP_MTF = True  → False
```
Обоснование: VWAP_MTF фильтр дублирует RVR_FILTER + K-Means gate в защите от трендов.
На ликвидных FX парах добавляет задержку к сигналам mean-reversion без дополнительного edge.

**НЕ трогать без видимой проблемы:**
- `RVR_FILTER`, `KALMAN_INNOV_FILTER` — эти фильтры защищают от реальных событий
  (trending spread, kalman divergence). Их отключение = curve-fitting.
- `MTF_CONFIRM` — 5-минутное подтверждение направления. Умеренная защита, оставить.

---

## После появления real_trades > 0 — немедленно запустить полный backtest

```bash
python step4a_backtest.py 2>&1
```

Цели для Zero-account:
```
max_inactive_days:      < 30
trailing_loss_breached: False
daily_loss_breached:    False
max_profit_days_30d:    >= 7
```
