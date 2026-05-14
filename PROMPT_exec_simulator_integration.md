# ЗАДАЧА: Интеграция ExecutionSimulator в step4a_backtest.py

## Контекст и цель

Я строю кванттрейдинговую инфраструктуру AFES (FX статарб по парам).

Файлы, которые нужно прочитать перед работой:
- `execution_stress.py` — полный код `ExecutionSimulator` и `ExecutionResult` (уже написаны, протестированы)
- `step4a_backtest.py` — основной бэктестер

Цель: заменить текущую модель "идеального исполнения" на вызов `ExecutionSimulator.simulate_order()`.
Пока бэктестер не получает отказы типа `TOXIC_FLOW_CANCEL` или `LIMIT_NOT_FILLED`, PnL-цифры — это
иллюзия. Все изменения должны быть **backward-compatible**: `exec_sim=None` оставляет поведение
идентичным текущему.

---

## Шаг 0 — Прочитай оба файла полностью

Перед написанием кода выполни `Read` на:
1. `execution_stress.py` — изучи интерфейс `ExecutionSimulator.simulate_order()` и `stress_exit_price()`
2. `step4a_backtest.py` строки 399–820 — функция `backtest_pair()`

Ключевые интерфейсы `ExecutionSimulator` (для справки, проверь сам):

```python
# Конструктор (основные параметры)
ExecutionSimulator(
    base_spread: float = 0.0001,
    spread_gamma: float = 1.5,
    panic_multiplier: float = 3.0,
    entry_delay: int = 1,          # ← совпадает с next_i = i + 1 в бэктестере
    slippage_variance: float = 0.0,
    fill_kappa: float = 8.0,
    passive: bool = True,
    use_vwap: bool = True,
    random_seed: int = 42,
    toxicity_threshold: float | None = None,
    price_col: str = "close",       # ← ищет "close" в prices_df
    vwap_col: str = "vwap",         # ← ищет "vwap" в prices_df
    vol_ratio_col: str = "vol_ratio",
    hmm_col: str = "hmm_regime",
    tox_buy_col: str = "tox_buy",
    tox_sell_col: str = "tox_sell",
)

# Главный вызов
result: ExecutionResult = sim.simulate_order(
    prices_df,           # DataFrame с колонками выше
    signal_index: int,   # индекс бара где сработал сигнал (bar i)
    side: int,           # +1 = buy/long, -1 = sell/short
    requested_price: float,
    order_type: str = "limit",   # "market" для принудительных выходов
)

# Результат
result.status      # "FILLED" | "MISSED" | "CANCELED"
result.fill_price  # реальная цена исполнения (NaN если не заполнено)
result.reason      # "FILLED" | "TOXIC_FLOW_CANCEL" | "LIMIT_NOT_FILLED" | "NO_BAR_AFTER_LATENCY"
result.fill_index  # индекс бара исполнения (= i + entry_delay)
result.effective_spread
result.slippage

# Выход по рынку (стоп, circuit-breaker)
exit_mid: float = sim.stress_exit_price(
    prices_df,
    bar_index: int,
    side_to_close: int,   # -position (противоположная сторона для закрытия)
)
```

---

## Шаг 1 — Импорт

В начало `step4a_backtest.py`, в блок импортов:

```python
from execution_stress import ExecutionSimulator, ExecutionResult
```

---

## Шаг 2 — Добавить параметр в сигнатуру `backtest_pair()`

Текущая сигнатура (строки 399–412):
```python
def backtest_pair(df, t1, t2, beta, pair_name: str = "",
                  ...,
                  session_window: tuple[int, int] | None = None) -> pd.DataFrame:
```

Добавить последним параметром:
```python
                  exec_sim: "ExecutionSimulator | None" = None) -> pd.DataFrame:
```

---

## Шаг 3 — Построить `prices_df` ДО основного цикла

Найди в `backtest_pair()` строку с `for i in range(len(df)):` и вставь ПЕРЕД ней:

```python
# ── Подготовка prices_df для ExecutionSimulator ───────────────────────────
# Симулятор ищет "close", "vwap", "vol_ratio", "hmm_regime".
# Для spread-стратегии "close" = сам спред (tradeable price).
# precompute() кэшируется — вызываем один раз до цикла, O(1) внутри.
if exec_sim is not None:
    _sim_cols: dict[str, pd.Series] = {"close": df["spread"]}
    if "spread_vwap_mtf" in df.columns:
        _sim_cols["vwap"] = df["spread_vwap_mtf"]
    if "vol_ratio" in df.columns:
        _sim_cols["vol_ratio"] = df["vol_ratio"]
    if "hmm_regime" in df.columns:
        _sim_cols["hmm_regime"] = df["hmm_regime"]
    _prices_df = pd.DataFrame(_sim_cols, index=df.index)
    exec_sim.precompute(_prices_df)
else:
    _prices_df = None  # не используется
```

---

## Шаг 4 — Entry: заменить идеальный fill на simulate_order()

Найди блок (примерно строки 781–795):
```python
                # Execute at NEXT bar (signal on close i, fill on bar i+1)
                next_i = i + 1
                if next_i >= len(df):
                    diag["no_next_bar"] += 1
                    position = 0
                    continue
                entry_spread   = df["spread"].iloc[next_i]   # ← ЗАМЕНИТЬ
                entry_t1       = df[t1_col].iloc[next_i]
                entry_t2       = df[t2_col].iloc[next_i]
                entry_beta     = df["beta"].iloc[next_i]
                entry_alpha    = df["alpha"].iloc[next_i]
                entry_std      = df["spread_std"].iloc[next_i]
                entry_bar      = next_i
```

Заменить строку `entry_spread = df["spread"].iloc[next_i]` на следующий блок
(все остальные строки `entry_t1`, `entry_t2` и т.д. — оставить без изменений):

```python
                # ── Execution simulation ───────────────────────────────
                # signal_index=i: симулятор сам добавит entry_delay=1 → fill на next_i.
                # side=position: +1 покупаем спред (long leg1/short leg2),
                #                -1 продаём спред.
                # requested_price: best-effort цена которую мы хотим (close[next_i]).
                if exec_sim is not None:
                    _fill = exec_sim.simulate_order(
                        _prices_df,
                        signal_index=i,
                        side=position,
                        requested_price=float(df["spread"].iloc[next_i]),
                        order_type="limit",
                    )
                    if _fill.status != "FILLED":
                        # Ордер отклонён: токсичный поток или лимит не тронут.
                        # Записываем в trades с net_pnl=0 чтобы fill_rate считался честно.
                        trades.append({
                            "pair":         f"{t1}-{t2}",
                            "entry_time":   df.index[i],
                            "exit_time":    df.index[i],
                            "direction":    "LONG" if position == 1 else "SHORT",
                            "holding_bars": 0,
                            "n_shares":     0.0,
                            "size":         0.0,
                            "gross_pnl":    0.0,
                            "tx_cost":      0.0,
                            "borrow_cost":  0.0,
                            "net_pnl":      0.0,
                            "cum_pnl":      round(cumulative_pnl, 4),
                            "exit_reason":  _fill.reason,
                            "entry_z":      round(z, 2),
                            "exit_z":       round(z, 2),
                        })
                        diag[f"exec_{_fill.reason}"] += 1
                        position = 0
                        continue
                    # Реальная цена входа с учётом half-spread + slippage
                    entry_spread = _fill.fill_price
                else:
                    entry_spread = df["spread"].iloc[next_i]
```

---

## Шаг 5 — Force-close exit: заменить `spread_now` на `stress_exit_price()`

Найди блок force-close (примерно строки 488–494):
```python
        if position != 0 and force_close:
            n              = entry_n_shares
            gross_pnl      = position * (spread_now - entry_spread) * n   # ← ЗАМЕНИТЬ
            notional       = (entry_t1 + abs(entry_beta) * entry_t2) * n
            tx_cost        = notional * COST_MAKER + notional * COST_TAKER
```

Заменить строку `gross_pnl = position * (spread_now - entry_spread) * n` на:
```python
            # Market exit: платим полный spread penalty (panic/coint-break выходим по рынку).
            if exec_sim is not None:
                _exit_mid  = exec_sim.stress_exit_price(_prices_df, bar_index=i, side_to_close=-position)
                gross_pnl  = position * (_exit_mid - entry_spread) * n
            else:
                gross_pnl  = position * (spread_now - entry_spread) * n
```

---

## Шаг 6 — Normal exit (SIGNAL / STOP / TIME_STOP): такая же замена

Найди блок normal exit (примерно строки 532–535):
```python
            if exit_signal or stop_signal or time_stop:
                n              = entry_n_shares
                gross_pnl      = position * (spread_now - entry_spread) * n   # ← ЗАМЕНИТЬ
                notional       = (entry_t1 + abs(entry_beta) * entry_t2) * n
```

Заменить `gross_pnl = position * (spread_now - entry_spread) * n` на:
```python
                # exit_signal → лимитный выход (force_fill=True: мы хотим выйти в любом случае).
                # stop_signal / time_stop → рыночный выход (stress_exit_price).
                if exec_sim is not None:
                    if exit_signal:
                        # Лимитный выход по сигналу: берём stress_exit для консерватизма
                        # (или можно simulate_order с force_fill=True — на твоё усмотрение).
                        _exit_mid = exec_sim.stress_exit_price(_prices_df, bar_index=i, side_to_close=-position)
                    else:
                        # Стоп или тайм-стоп: всегда рыночный выход, полный spread penalty.
                        _exit_mid = exec_sim.stress_exit_price(_prices_df, bar_index=i, side_to_close=-position)
                    gross_pnl = position * (_exit_mid - entry_spread) * n
                else:
                    gross_pnl = position * (spread_now - entry_spread) * n
```

---

## Шаг 7 — Обновить вызов `backtest_pair()` в основном цикле

Найди строку ~1323:
```python
    trades = backtest_pair(df_sig, t1, t2, beta,
                           pair_name=row["pair"],
                           ...
                           session_window=_sessions.get(row["pair"]))
```

ДО основного цикла по парам (один раз!) создать симулятор:
```python
# Создаём один экземпляр ExecutionSimulator на весь прогон.
# Параметры: entry_delay=1 соответствует логике "signal on close i, fill on i+1".
# toxicity_threshold=None означает TOXIC_FLOW_CANCEL отключён (включить когда
# добавим tox_buy/tox_sell колонки в df).
_exec_sim = ExecutionSimulator(
    base_spread=COST_MAKER,      # используем COST_MAKER как нижнюю оценку half-spread
    spread_gamma=1.5,
    panic_multiplier=3.0,
    entry_delay=1,
    slippage_variance=0.0,
    fill_kappa=8.0,
    passive=True,
    use_vwap=True,               # использует spread_vwap_mtf как reference price
    random_seed=42,
    toxicity_threshold=None,     # включить позже
)
```

Добавить `exec_sim=_exec_sim` в вызов `backtest_pair(...)`.

---

## Шаг 8 — Добавить метрики fill_rate в отчёт

После основного цикла, там где считается статистика по trades, добавить:

```python
# Fill rate: сколько сигналов реально исполнилось
if not all_trades.empty and "exit_reason" in all_trades.columns:
    total_signals    = len(all_trades)
    filled_signals   = int((all_trades["exit_reason"] != "TOXIC_FLOW_CANCEL")
                           & (all_trades["exit_reason"] != "LIMIT_NOT_FILLED")).sum()
    toxic_cancels    = int((all_trades["exit_reason"] == "TOXIC_FLOW_CANCEL").sum())
    limit_misses     = int((all_trades["exit_reason"] == "LIMIT_NOT_FILLED").sum())
    fill_rate        = filled_signals / max(total_signals, 1)
    print(f"  fill_rate={fill_rate:.1%}  toxic_cancels={toxic_cancels}  limit_misses={limit_misses}")
```

---

## Архитектурные ограничения (не нарушать)

1. `exec_sim=None` → поведение **идентично** текущему. Никаких изменений в логике при None.
2. `_prices_df` строится **один раз** до `for i in range(len(df))`. Не внутри итерации.
3. `precompute()` вызывается **один раз** до цикла. Повторные вызовы `simulate_order` используют кэш.
4. `entry_delay=1` в симуляторе **совпадает** с `next_i = i + 1` в бэктестере. Не менять.
5. Rejection records пишутся в `trades` с `net_pnl=0.0` — это необходимо для честного `fill_rate`.
6. `side=position` (не `side=-position`): `position=1` = long = buy spread = side=+1.

---

## Что НЕ трогать

- Логику всех фильтров (Hurst, coint, macro, MTF, RVOL, velocity и т.д.)
- `entry_t1`, `entry_t2`, `entry_beta`, `entry_alpha`, `entry_std`, `entry_bar`, `entry_n_shares` — читаются из `df[next_i]` как и раньше
- Расчёт `tx_cost` и `borrow_cost` — не меняется (они поверх gross_pnl)
- `position_size()` логику
- Все вызовы за пределами `backtest_pair()`

---

## Проверка после изменений

Запусти mini smoke-test:

```python
from execution_stress import ExecutionSimulator
import pandas as pd, numpy as np

# Проверь что при exec_sim=None результаты идентичны старому коду
# Проверь что при exec_sim=ExecutionSimulator() появляются записи с
# exit_reason="LIMIT_NOT_FILLED" или "TOXIC_FLOW_CANCEL" в trades DataFrame
# Проверь что fill_price != spread.iloc[next_i] (есть реальный slippage/spread penalty)
```
