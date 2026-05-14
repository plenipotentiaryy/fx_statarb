# ЗАДАЧА: Расширение торгового universe для Zero account

## Диагноз — почему у нас только 7 пар из 1128 возможных

Прочитай эти файлы полностью перед тем как писать код:
- `config.py` — параметры системы
- `step2b_pairs_fx.py` — скрипт выбора пар из локальных данных
- `data/pairs_selected.csv` — текущий universe (7 пар)

### Факты из кода

```
closes_1min.csv      → 66 тикеров
После NON_FX filter  → 48 FX тикеров
Возможных пар C(48,2) = 1128
Текущий universe     = 7 пар
```

**Причина 1 — `COINT_TOP_N = 3` в `config.py`:**
`step2a_pairs.py` (через yfinance) берёт только топ-3 пары. При нескольких anchor-тикерах
получается 5-10 пар максимум. Это жёсткое ограничение которое обрезает найденные пары.

**Причина 2 — сломанная логика `HALF_LIFE_MAX` в `step2b_pairs_fx.py:22`:**
```python
HALF_LIFE_MAX = 1440            # задумка: "не более 1 торгового дня"
...
hl_bars = round(hl * BARS_PER_DAY, 0)   # ← BUG: hl из daily данных УЖЕ в днях
                                         # умножение на 1440 даёт 7200+ для 5-day HL
if hl_bars > HALF_LIFE_MAX: continue    # ← отсеивает ВСЕ нормальные FX пары
```

`hl` из `compute_half_life()` на daily данных — это число **дней**. Типичная FX пара:
hl = 5–20 дней → hl_bars = 5 × 1440 = 7200 >> 1440 → все нормальные пары отсеяны.
Через фильтр проходят только аномально быстрые/шумные пары с hl < 1 дня (если вообще есть).

`step2b_pairs_fx.py` работает полностью на **локальных данных** (`closes_daily.csv`) — 
интернет не нужен. Это правильный скрипт для расширения universe.

---

## Шаг 1: Исправить `step2b_pairs_fx.py`

Прочитай файл. Найди строку с `HALF_LIFE_MAX = 1440` (строка ~22) и блок с `hl_bars`.

### Изменение 1.1 — убрать ошибочное умножение
```python
# БЫЛО (строка ~22):
HALF_LIFE_MAX = 1440   # bars on intraday tf — max 1 trading day at 1-min

# СТАЛО:
HALF_LIFE_MAX_DAYS = 30   # max daily half-life in calendar days (30 days = ~1 month)
```

### Изменение 1.2 — исправить расчёт и проверку
Найди в функции `main()` строки с `hl_bars`:
```python
# БЫЛО:
hl_bars = round(hl * BARS_PER_DAY, 0)   # correct for any bar size
if hl_bars > HALF_LIFE_MAX or hl_bars <= 0:
    continue

# СТАЛО:
# hl здесь — половинный период в ДНЯХ (из daily данных)
# Конвертируем в минутные бары для сохранения в pairs_selected.csv
hl_bars = round(hl * BARS_PER_DAY, 1)
if hl <= 0 or hl > HALF_LIFE_MAX_DAYS:
    continue
```

### Изменение 1.3 — ослабить HURST_MAX
```python
# БЫЛО (строка ~17):
HURST_MAX = 0.45   # tightened: H must be clearly mean-reverting

# СТАЛО:
HURST_MAX = 0.48   # consistent with config.py (HURST_MAX = 0.50 там; 0.48 чуть строже)
```

### Изменение 1.4 — ослабить CORR_MIN
```python
# БЫЛО (строка ~14):
CORR_MIN = 0.40

# СТАЛО:
CORR_MIN = 0.35   # ловим больше пар; live_corr_filter в backtest проверит актуальную корреляцию
```

---

## Шаг 2: Увеличить `COINT_TOP_N` в `config.py`

```python
# БЫЛО (строка 70):
COINT_TOP_N = 3

# СТАЛО:
COINT_TOP_N = 15   # step2b сохраняет всё сам; step2a ограничен этим числом
```

---

## Шаг 3: Запустить step2b и проверить результат

```bash
# Запуск — только локальные данные, интернет не нужен
python step2b_pairs_fx.py 2>&1
```

Ожидаемый вывод:
```
Loaded N symbols and M days.
Total potential pairs: ~500-1000
Layer 2 (corr >= 0.35): ~50-150 pairs remain
Layer 2b (SSD <= p...): ~30-80 pairs remain
Testing N pairs with Johansen...
Saved X pairs to data/pairs_selected.csv   ← хотим X >= 15
```

Если `Saved < 10 pairs` — значит `closes_daily.csv` не содержит достаточно тикеров.
Тогда смотри Шаг 5 (альтернативный путь).

После запуска проверь сохранённые пары:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/pairs_selected.csv')
print(f'Пар: {len(df)}')
print(df[['pair','corr','hurst','half_life_bars']].to_string(index=False))
"
```

---

## Шаг 4: Верификация backtest с расширенным universe

```bash
# Smoke-test: 1 месяц
BACKTEST_RECENT_BARS=43200 python step4a_backtest.py 2>&1

# Полный прогон (если smoke прошёл с exit code 0)
python step4a_backtest.py 2>&1
```

Ключевые метрики для сравнения с предыдущим прогоном:

```
# Было (7 пар):
max_inactive_days = 4000+
trades per pair   = 1-2

# Ожидаем (15-30 пар):
max_inactive_days < 30
trades per pair   >= 20
```

---

## Шаг 5: Если `closes_daily.csv` не содержит нужных тикеров

Проверь:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/closes_daily.csv', nrows=2)
print(f'Тикеры в closes_daily.csv: {len(df.columns)-1}')
print(sorted([c for c in df.columns if c != 'datetime' and c != df.columns[0]]))
"
```

Если тикеров < 20 — `closes_daily.csv` был сгенерирован ограниченным прогоном.
В этом случае нужно запустить `step2a_pairs.py` (требует интернет) ИЛИ пересоздать 
`closes_daily.csv` из имеющегося `closes_1min.csv` через ресэмплинг:

```python
# generate_daily_from_intraday.py (создать и запустить если нужно)
import pandas as pd
df = pd.read_csv('data/closes_1min.csv', index_col=0, parse_dates=True)
daily = df.resample('1D').last().dropna(how='all')
daily.to_csv('data/closes_daily.csv')
print(f'Saved: {daily.shape[0]} days × {daily.shape[1]} tickers')
```

После этого повторить Шаг 3.

---

## Критерии успеха

```
pairs_selected.csv:  >= 15 пар  (было 7)
max_inactive_days:   < 30       (было 4000+)
backtest exit code:  0          (чистый прогон)
```

Если пар >= 15 но max_inactive_days всё ещё > 30 — проблема в параметрах 
(ENTRY_Z, фильтры), а не в universe. Смотри PROMPT_zero_profile_config.md.

---

## Что НЕ трогать

- `step4a_backtest.py` — не трогать
- `step3e_sizing.py` — не трогать  
- `execution_stress.py` — не трогать
- `kalman.py` — не трогать (отдельный промпт)
- `data/pairs_selected.csv` — перезапишется автоматически при запуске step2b
