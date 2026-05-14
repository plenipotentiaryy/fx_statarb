# ЗАДАЧА: Подготовка AFES к live-торговле на FundingPips Zero ($5,000)

## Контекст — прочитай это полностью перед тем как трогать код

Система AFES (Automated FX Execution System) — FX stat-arb на парах (EUR/USD, GBP/USD, AUD/USD, NZD/USD).
Цель: запустить на FundingPips Zero account ($5,000, leverage 50x).

**Лимиты FundingPips Zero (зашиты в `config.py`):**
- Max daily loss: 3% от баланса дня (`ZERO_MAX_DAILY_LOSS_PCT = 0.03`)
- Max trailing drawdown: 5% (`ZERO_MAX_TRAILING_LOSS_PCT = 0.05`)
- Max inactive gap: 30 дней (`ZERO_MAX_INACTIVE_DAYS = 30`)
- Min profitable days / 30d window: 7 (`ZERO_MIN_PROFIT_DAYS_30D`)

**Что уже сделано (НЕ ТРОГАТЬ — баги уже исправлены):**
1. `step3e_sizing.py:95` — `HMM_PANIC_MULT` вместо `0.0` (был критический баг: 34% дней size=0)
2. `step3e_sizing.py:106` — `pd.isna()` проверка для macro_alert (был `bool(NaN)=True` → блокировал 20 лет)
3. `step4a_backtest.py` — `real_trades` фильтр (rejection-записи от ExecutionSimulator больше не загрязняют zero_account_report)
4. `step4a_backtest.py` — `_exec_sim = ExecutionSimulator(...)` создан и передаётся в `backtest_pair()`

**Что НЕ сделано — ЭТО твоя задача (два файла):**
- `config.py`: `USE_VELOCITY_GATE = True` → `False` (критично, объяснение ниже)
- `kalman.py`: Python-loop → scalar-loop (performance fix, объяснение ниже)

---

## Почему velocity gate нужно отключить — экономическое обоснование

Velocity gate проверяет: при входе в long (z < -2σ) требует vel > 0 (z уже разворачивается).
Задумка — "не лови падающий нож". На практике — математическая ловушка:

**Проблема**: Когда z **впервые** касается -2σ, он почти всегда ещё движется вниз (vel < 0).
Это означает: gate блокирует самые глубокие, самые профитные входы — first-touch в экстремуме.
"Подтверждённые" входы (vel > 0 при z < -2σ) — это **поздние** входы, z уже восстановился от минимума.

**Измеренный результат на 500K баров OU-процесса:**
- `vel < 0` (заблокированные): mean_pnl=0.1807, win=47%, Sharpe=37.16
- `vel > 0` (пропущенные): mean_pnl=0.1727, win=46%, Sharpe=27.43

Gate фильтрует **лучшие** входы с Sharpe=37 вместо плохих. При этом:
- Частота сделок: было 241K z-triggers → velocity проходили только ~34% → 82K
- Gap между сделками: 4031–4123 дня (против лимита 30 дней для Zero)
- Дублирование защиты: против trending рынков уже есть Hurst filter (H>0.55) + coint filter + K-Means macro

От trending рынков система защищена 3 независимыми фильтрами. Velocity gate — избыточен и вреден.

---

## Шаг 1 из 3: `config.py` — одна строка

Прочитай `config.py`, найди строку 123:
```python
USE_VELOCITY_GATE = True
```
Замени на:
```python
USE_VELOCITY_GATE = False   # отключён: блокировал лучшие first-touch входы (Sharpe 37 vs 27)
```

Больше в `config.py` ничего не менять. Не трогать `ENTRY_Z`, не трогать другие флаги.

---

## Шаг 2 из 3: `kalman.py` — performance fix (scalar loop)

Прочитай `kalman.py` полностью. Там `for t in range(n)` с numpy-массивами внутри.
840ms на 132K баров × 10 пар = 8.4 секунды только на Kalman.

Причина: numpy-overhead на матрицах 2×2 (~4μs/вызов) × 130K итераций.
Решение: заменить `np.array` и `np.eye(2)` на 5 Python-скаляров (a, b, P00, P01, P11).
Математика тождественна. Интерфейс функции — идентичен. Ожидаемое ускорение: 7–10x.

**Полная замена содержимого `kalman.py`:**

```python
from __future__ import annotations
import numpy as np


def kalman_hedge(
    p1: np.ndarray,
    p2: np.ndarray,
    delta: float = 1e-4,
    beta_init: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kalman Filter for time-varying hedge ratio.
    Observation model: p1[t] = alpha[t] + beta[t]*p2[t] + noise
    State transition:  [alpha, beta] ~ random walk  (Q = delta/(1-delta) * I)

    Scalar-loop implementation: replaces numpy 2x2 matrix ops with 5 Python
    floats (a, b, P00, P01, P11). Mathematically identical to the original;
    ~7-10x faster because it avoids per-iteration numpy C-extension overhead.

    Returns: (alpha_arr, beta_arr, innovations, innovation_var)
    Signature unchanged — all callers in step4a_backtest.py work without modification.
    """
    n = len(p1)
    Q: float = delta / (1.0 - delta)   # process noise scalar (added to P diagonal each step)
    R: float = 1.0                      # observation noise (spread in price units)

    # State — plain Python floats, not numpy arrays
    a: float   = 0.0
    b: float   = float(beta_init)
    # Covariance — symmetric 2×2, only 3 unique elements needed
    P00: float = 1.0
    P01: float = 0.0   # P[0,1] == P[1,0]
    P11: float = 1.0

    alpha_out = np.empty(n, dtype=np.float64)
    beta_out  = np.empty(n, dtype=np.float64)
    innov     = np.empty(n, dtype=np.float64)
    innov_var = np.empty(n, dtype=np.float64)

    # Cast inputs once outside the loop (avoids per-iteration type coercion)
    p1_ = np.asarray(p1, dtype=np.float64)
    p2_ = np.asarray(p2, dtype=np.float64)

    for t in range(n):
        h1: float = p2_[t]          # observation matrix H = [1.0, h1]

        # ── Predict ──────────────────────────────────────────────────────
        P00 += Q                    # P += Q*I  (only diagonal grows)
        P11 += Q

        # ── Innovation: e = p1[t] - H @ theta ───────────────────────────
        e: float = p1_[t] - (a + b * h1)

        # ── Innovation variance: S = H @ P @ H^T + R ────────────────────
        # H=[1,h1], so S = P00 + 2*P01*h1 + P11*h1^2 + R
        S: float = P00 + 2.0 * P01 * h1 + P11 * h1 * h1 + R

        # ── Kalman gain: K = P @ H^T / S ────────────────────────────────
        K0: float = (P00 + P01 * h1) / S
        K1: float = (P01 + P11 * h1) / S

        # ── Update state: theta += K * e ─────────────────────────────────
        a += K0 * e
        b += K1 * e

        # ── Update covariance: P_new = (I - K @ H) @ P ───────────────────
        # Computed from pre-update P values to avoid overwrite hazard
        new_P00 = P00 - K0 * P00 - K0 * h1 * P01
        new_P01 = P01 - K0 * P01 - K0 * h1 * P11
        new_P11 = P11 - K1 * P01 - K1 * h1 * P11
        P00, P01, P11 = new_P00, new_P01, new_P11

        alpha_out[t] = a
        beta_out[t]  = b
        innov[t]     = e
        innov_var[t] = S

    return alpha_out, beta_out, innov, innov_var
```

---

## Шаг 3 из 3: Верификация — запусти и прочитай цифры

После изменений в шагах 1–2, выполни последовательно:

### 3a. Синтаксическая проверка
```bash
python -c "import ast; ast.parse(open('config.py').read()); print('config.py OK')"
python -c "import ast; ast.parse(open('kalman.py').read()); print('kalman.py OK')"
python -c "import ast; ast.parse(open('step4a_backtest.py').read()); print('step4a_backtest.py OK')"
```

### 3b. Kalman regression test
```bash
python -c "
import numpy as np, time
rng = np.random.default_rng(42)
n = 10_000
p1 = np.cumsum(rng.normal(0, 0.0001, n)) + 1.1
p2 = np.cumsum(rng.normal(0, 0.0001, n)) + 1.27
from kalman import kalman_hedge
a, b, inn, var = kalman_hedge(p1, p2)
assert np.all(np.isfinite(b)), 'beta has non-finite values'
assert 0.5 < b.mean() < 2.0, f'beta mean out of range: {b.mean()}'
assert np.all(var > 0), 'innovation variance must be positive'
t0 = time.perf_counter()
for _ in range(5): kalman_hedge(p1, p2)
ms = (time.perf_counter()-t0)/5*1000
print(f'OK  beta_mean={b.mean():.4f}  speed={ms:.0f}ms / {n} bars')
"
```
Ожидаемый результат: `beta_mean` в диапазоне 0.5–2.0, speed < 200ms.

### 3c. Smoke-test backtest (1 месяц данных, ~2 мин)
```bash
BACKTEST_RECENT_BARS=43200 python step4a_backtest.py 2>&1
```

### 3d. Полный backtest (если smoke-test прошёл с exit code 0)
```bash
python step4a_backtest.py 2>&1
```

---

## Критерии успеха — что искать в выводе

**Обязательные условия (Zero account готов к живой торговле):**

```
# В выводе по каждой паре:
trades=  [не 0, желательно > 50 за весь период]
WR=      [> 45%]

# В итоговом блоке PORTFOLIO:
Trades:        [> 100 общих]
Win rate:      [> 45%]
Net P&L:       [положительный]
Max drawdown:  [в долларах < $250  =  5% от $5,000]

# В блоке FUNDINGPIPS ZERO CHECK:
max_inactive_days:    [< 30]          ← было 4031, после фикса должно упасть
trailing_loss_breached: False
daily_loss_breached:  False
max_profit_days_30d:  [>= 7]
```

**Если max_inactive_days всё ещё > 30 после отключения velocity gate:**
Это означает стратегия генерирует недостаточно сделок. Тогда (и только тогда) сделай второй шаг:
в `config.py` снизить `ENTRY_Z = 2.0` → `ENTRY_Z = 1.7` и перезапустить.
Не делать оба изменения сразу — нужно видеть эффект каждого по отдельности.

**Если exit code != 0:**
Покажи полный traceback. Не пытайся чинить наугад — сначала читай ошибку.

---

## Что НЕ трогать

- `step3e_sizing.py` — баги уже исправлены, трогать нельзя
- `step4a_backtest.py` — `real_trades` логика уже исправлена, трогать нельзя
- `execution_stress.py` — `ExecutionSimulator` завершён и работает
- `filters.py`, `copula_signals.py`, `macro_calendar.py` — не в scope
- Все остальные параметры `config.py` кроме указанных выше

Изменений ровно два: одна строка в `config.py` и полная замена `kalman.py`.
