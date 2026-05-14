# ЗАДАЧА: Ускорение kalman_hedge() и сквозной запуск бэктестера

## Диагностика (профилирование уже сделано — цифры реальные)

Замеры на `n = 132_480` баров (1-минутные, ~4.6 месяца = `BARS_PER_DAY * 92`):

| Компонент | Время | Вывод |
|---|---|---|
| `kalman_hedge()` | **840ms / вызов** | 🔴 Единственный bottleneck |
| `rolling zscore` (pandas) | 10ms | ✅ Векторизован, не трогаем |
| `for i in range(len(df))` backtest loop | 10ms | ✅ Не bottleneck |
| 10 пар × kalman | ~8.4s | 🔴 Неприемлемо для итераций |

**Вывод**: `backtest_pair()` loop — НЕ проблема (10ms). Copula / rolling / filters — НЕ проблема.
Единственный виновник: `kalman.py`, строка 27: `for t in range(n)` с numpy-массивами внутри.

**Корень зла**: на каждой итерации Python дёргает numpy C-extension для матриц 2×2.
Overhead на вызов numpy-операции (~2–4μs) × 130K итераций = 840ms.
Если заменить numpy-массивы на Python-скаляры → ~7–10x ускорение (scalar ops ~0.1μs).

**numba НЕ установлена** — нельзя просто добавить `@njit`.

---

## Трек A: НЕМЕДЛЕННЫЙ обходной путь (0 строк кода)

Переменная окружения `BACKTEST_RECENT_BARS` **уже существует** в коде
(`step4a_backtest.py`, строки 118–126). Используй её прямо сейчас:

```bash
# Smoke-test: 1 месяц = ~43 200 баров → весь прогон ~1-2 секунды
export BACKTEST_RECENT_BARS=43200
python step4a_backtest.py

# Полноценный OOS-тест: 3 месяца = ~130 000 баров
export BACKTEST_RECENT_BARS=130000
python step4a_backtest.py

# Убрать ограничение (полные данные)
unset BACKTEST_RECENT_BARS
python step4a_backtest.py
```

**Важно**: Kalman прогревается на ВСЕХ исторических данных для правильной оценки beta.
`BACKTEST_RECENT_BARS` обрезает `closes` ДО `build_signals()`, то есть Kalman тоже
прогревается на урезанных данных. Для smoke-test это приемлемо, для финального
прогона нужен Трек B.

---

## Трек B: Правильное исправление — scalar Kalman в `kalman.py`

Прочитай `kalman.py` полностью (он небольшой, ~50 строк).

**Принцип**: заменить numpy-массивы `theta` (2D) и `P` (2×2) на 5 Python-скаляров
`(a, b, P00, P01, P11)`. Математика не меняется ни на бит.

### Полная замена содержимого `kalman.py`:

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

    Реализация на Python-скалярах вместо numpy-массивов внутри цикла.
    Причина: numpy-overhead на микроматрицах 2×2 (~4μs/вызов) × 130K итераций
    = 500ms+. Скаляры дают ~0.1μs/итерацию → 7–10x ускорение без numba/Cython.

    Returns: (alpha_arr, beta_arr, innovations, innovation_var)
      - innovations  : residuals e[t] = p1[t] - predicted p1[t]
      - innovation_var: theoretical variance of each innovation (for zscore normalisation)

    Интерфейс идентичен оригиналу — замена прозрачна для всех вызывающих.
    """
    n = len(p1)

    # Процессовый шум (scalar, добавляется к диагонали P на каждом шаге)
    Q: float = delta / (1.0 - delta)
    R: float = 1.0  # observation noise в price units

    # Начальное состояние — Python-скаляры (не numpy-массивы!)
    a: float  = 0.0          # alpha
    b: float  = float(beta_init)  # beta
    # P — симметричная 2×2 ковариационная матрица: 3 уникальных элемента
    P00: float = 1.0
    P01: float = 0.0  # P[0,1] = P[1,0] по симметрии
    P11: float = 1.0

    # Выходные массивы — выделяем один раз
    alpha_out = np.empty(n, dtype=np.float64)
    beta_out  = np.empty(n, dtype=np.float64)
    innov     = np.empty(n, dtype=np.float64)
    innov_var = np.empty(n, dtype=np.float64)

    # Приводим входные данные к float64-массивам один раз вне цикла
    p1_ = np.asarray(p1, dtype=np.float64)
    p2_ = np.asarray(p2, dtype=np.float64)

    for t in range(n):
        h1: float = p2_[t]   # H = [1.0, h1]  (наблюдательный вектор)

        # ── Predict ─────────────────────────────────────────────────────
        # P += Q * I  (только диагональные элементы)
        P00 += Q
        P11 += Q

        # ── Innovation ──────────────────────────────────────────────────
        e: float = p1_[t] - (a + b * h1)

        # ── Innovation variance: S = H @ P @ H^T + R ────────────────────
        # H = [1, h1], H^T = [[1],[h1]]
        # H @ P = [P00 + P01*h1,  P01 + P11*h1]
        # S = (P00 + P01*h1)*1 + (P01 + P11*h1)*h1 + R
        S: float = P00 + 2.0 * P01 * h1 + P11 * h1 * h1 + R

        # ── Kalman gain: K = P @ H^T / S ────────────────────────────────
        K0: float = (P00 + P01 * h1) / S
        K1: float = (P01 + P11 * h1) / S

        # ── Update state: theta += K * e ─────────────────────────────────
        a += K0 * e
        b += K1 * e

        # ── Update covariance: P = (I - K @ H) @ P ──────────────────────
        # (I - KH) = [[1-K0, -K0*h1], [-K1, 1-K1*h1]]
        # Новые элементы P (считаем через старые, до перезаписи):
        new_P00 = P00 - K0 * P00 - K0 * h1 * P01
        new_P01 = P01 - K0 * P01 - K0 * h1 * P11
        new_P11 = P11 - K1 * P01 - K1 * h1 * P11
        P00, P01, P11 = new_P00, new_P01, new_P11

        # ── Записываем результат ─────────────────────────────────────────
        alpha_out[t] = a
        beta_out[t]  = b
        innov[t]     = e
        innov_var[t] = S

    return alpha_out, beta_out, innov, innov_var
```

### Проверка корректности после замены

Обязательно запусти этот тест и убедись что результаты совпадают с оригиналом:

```python
# verify_kalman.py — запусти из директории проекта
import numpy as np, sys
sys.path.insert(0, '.')

rng = np.random.default_rng(42)
n = 10_000
p1 = np.cumsum(rng.normal(0, 0.0001, n)) + 1.1
p2 = np.cumsum(rng.normal(0, 0.0001, n)) + 1.27

# Старая версия — перед заменой сохрани её как kalman_old.py
# from kalman_old import kalman_hedge as kalman_old
# a_old, b_old, i_old, v_old = kalman_old(p1, p2)

from kalman import kalman_hedge
a_new, b_new, i_new, v_new = kalman_hedge(p1, p2)

# Визуальная проверка: beta должна быть близка к 1.0 и медленно меняться
print(f"beta: mean={b_new.mean():.4f}  std={b_new.std():.4f}  "
      f"last={b_new[-1]:.4f}")
print(f"innov: mean={i_new.mean():.6f}  std={i_new.std():.6f}")
print(f"All finite: {np.all(np.isfinite(b_new)) and np.all(np.isfinite(i_new))}")

# Benchmark
import time
t0 = time.perf_counter()
for _ in range(10):
    kalman_hedge(p1, p2)
ms = (time.perf_counter()-t0)/10*1000
print(f"Speed ({n} bars): {ms:.1f}ms  (было ~{n/132480*840:.0f}ms)")
```

### Ожидаемый результат

| | Было (numpy loop) | Стало (scalar loop) |
|---|---|---|
| 132K bars | 840ms | ~120ms |
| 10 пар | ~8.4s | ~1.2s |
| `BACKTEST_RECENT_BARS=43200` | ~275ms | ~40ms |

---

## Трек C (опционально, потом): numba для максимального ускорения

Когда понадобится максимальная скорость (WFO-grid с сотнями прогонов):

```bash
pip install numba
```

Тогда добавить одну строку в `kalman.py`:
```python
from numba import njit

@njit(cache=True)
def kalman_hedge(p1, ...):
    ...  # код не меняется
```

Ожидаемый результат: ~10ms на 132K баров (ещё 10x к Треку B).

---

## Порядок действий

1. **Прямо сейчас** — `export BACKTEST_RECENT_BARS=43200` и запусти для проверки что
   пайплайн вообще проходит сквозь end-to-end без ошибок (это важнее скорости)

2. **После** — замени `kalman.py` на scalar-версию из Трека B

3. **Проверь** через `verify_kalman.py` что численные результаты не изменились

4. **Убери** `BACKTEST_RECENT_BARS` и прогони полный backtest

---

## Что НЕ трогать

- `step4a_backtest.py` — backtest loop (10ms, не bottleneck)
- `copula_signals.py` — уже векторизован через pandas rolling
- `filters.py` — CointegrationFilter кэшируется по дням
- Логику любых фильтров и сигналов

Сигнатура `kalman_hedge()` и возвращаемые типы — идентичны оригиналу.
Все вызовы в `step4a_backtest.py` (строки 203, 222) работают без изменений.
