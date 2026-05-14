# AFES → FundingPips Zero: Мастер-последовательность патчей

## Статус промптов

| № | Промпт | Файлы | Статус |
|---|--------|-------|--------|
| 0 | `PROMPT_exec_simulator_integration.md` | `step4a_backtest.py` | ✅ Уже применён |
| 1 | `PROMPT_kalman_perf_fix.md` | `kalman.py` | ⏳ Применить |
| 2 | `PROMPT_zero_profile_config.md` | `config.py` | ⏳ Применить |
| 3 | `PROMPT_macro_gate_patch.md` | `filters.py`, `step4a_backtest.py` | ⏳ Применить |
| 4 | `PROMPT_universe_expansion.md` | `step2b_pairs_fx.py`, `config.py` (COINT_TOP_N), запуск скрипта | ⏳ Применить |

> `PROMPT_prop_tuning_question.md` и `PROMPT_zero_readiness.md` — вопрос/ответ референсы, не требуют имплементации.

---

## Порядок применения (ОБЯЗАТЕЛЬНО соблюдать)

### Шаг 1 — Kalman scalar fix (`PROMPT_kalman_perf_fix.md`)

**Зачем первым**: независим ни от чего. Ускоряет каждый последующий backtest-прогон в 7–10 раз.
Применяется изолированно: полная замена `kalman.py`.

```bash
# Верификация после применения:
python -c "
import numpy as np, time
rng = np.random.default_rng(42)
n = 10_000
p1 = np.cumsum(rng.normal(0, 0.0001, n)) + 1.1
p2 = np.cumsum(rng.normal(0, 0.0001, n)) + 1.27
from kalman import kalman_hedge
a, b, inn, var = kalman_hedge(p1, p2)
t0 = time.perf_counter()
for _ in range(5): kalman_hedge(p1, p2)
ms = (time.perf_counter()-t0)/5*1000
print(f'OK  beta_mean={b.mean():.4f}  speed={ms:.0f}ms / {n} bars')
"
# Ожидаем: speed < 200ms (было ~840ms на 132K баров)
```

---

### Шаг 2 — Zero-профиль в config.py (`PROMPT_zero_profile_config.md`)

**Зачем вторым**: устанавливает правильный риск-профиль до того как расширяется universe.
Иначе больше пар × $150 риска = мгновенный breach.

12 изменений в `config.py`. Итоговая проверка:
```bash
python -c "from config import *; print(f'TARGET_RISK={TARGET_RISK_USD}  ENTRY_Z={ENTRY_Z}  PAIR_MAX_LOSS={PAIR_MAX_LOSS}')"
# Ожидаем: TARGET_RISK=20.0  ENTRY_Z=1.8  PAIR_MAX_LOSS=-75.0
```

---

### Шаг 3 — MacroFilter K-Means gate patch (`PROMPT_macro_gate_patch.md`)

**Зачем третьим**: отдельные изменения в `filters.py` и одна строка в `step4a_backtest.py`.
Не зависит от universe — применяется до расширения пар.

Изменения:
- `filters.py`: `!= 1` → `== 2` в `is_entry_blocked()` (Trend=0 больше не блокирует)
- `filters.py`: добавить метод `km_size_multiplier()` (Sideways=1.0, Trend=0.5, Panic=0.0)
- `step4a_backtest.py`: применить `km_size_multiplier` к `sz` после `position_size()`

Ожидаемый эффект: `macro_entry_blocked` падает с 2.07M → ~90K баров.

```bash
# Smoke-test после патча:
BACKTEST_RECENT_BARS=43200 python step4a_backtest.py 2>&1 | grep -E "blocks:|trades="
```

---

### Шаг 4 — Расширение universe (`PROMPT_universe_expansion.md`)

**Зачем последним**: работает с новыми параметрами (CORR_MIN=0.35, HURST_MAX=0.48).
Меняет `step2b_pairs_fx.py` и COINT_TOP_N в уже обновлённом `config.py`.

Изменения:
- `step2b_pairs_fx.py`: HALF_LIFE_MAX_DAYS=30 (исправляет unit bug), HURST_MAX=0.48, CORR_MIN=0.35
- `config.py`: COINT_TOP_N = 3 → 15 (уже меняется в Шаге 2)

После правок — запустить:
```bash
python step2b_pairs_fx.py 2>&1
# Ожидаем: Saved >= 15 pairs to data/pairs_selected.csv

python -c "
import pandas as pd
df = pd.read_csv('data/pairs_selected.csv')
print(f'Пар: {len(df)}')
print(df[['pair','corr','hurst','half_life_bars']].to_string(index=False))
"
```

---

### Финальный прогон

После всех четырёх шагов:
```bash
# Smoke-test (1 месяц, ~2-3 минуты):
BACKTEST_RECENT_BARS=43200 python step4a_backtest.py 2>&1

# Полный backtest (если smoke exit code 0):
python step4a_backtest.py 2>&1
```

**Критерии готовности к live (FundingPips Zero):**

```
pairs_selected.csv:     >= 15 пар           (было 7)
macro_entry_blocked:    ~90K баров           (было 2.07M)
max_inactive_days:      < 30                 (было 4000+)
trailing_loss_breached: False
daily_loss_breached:    False
max_profit_days_30d:    >= 7
Net P&L:                > 0
```

---

## Что уже зашито и НЕ ТРОГАТЬ

| Файл | Что исправлено |
|------|----------------|
| `step3e_sizing.py:95` | `HMM_PANIC_MULT` вместо `0.0` — был баг: 34% дней size=0 |
| `step3e_sizing.py:106` | `pd.isna()` check на macro_alert — был баг: bool(NaN)=True блокировал всю историю |
| `step4a_backtest.py` | `real_trades` фильтр — rejection-записи не загрязняют zero_account_report |
| `step4a_backtest.py` | `_exec_sim = ExecutionSimulator(...)` — ExecutionSimulator создан и передаётся |

---

## Если после всех патчей max_inactive_days > 30

Приоритет диагностики:
1. Проверить `pairs_selected.csv` — если < 15 пар, значит `closes_daily.csv` беден тикерами → регенерировать из `closes_1min.csv` (см. Шаг 5 в `PROMPT_universe_expansion.md`)
2. Если пар >= 15 но инактивность высокая → проблема в ENTRY_Z. Снизить до 1.6, но не ниже 1.5 (curve-fitting риск)
3. Проверить диагностику `blocks:` в выводе по парам — что именно блокирует больше всего
