# ЗАДАЧА: Ослабление K-Means macro gate в filters.py

## Прочитай эти файлы перед правками
- `filters.py` — весь файл (особенно класс `MacroFilter`, строки ~115–180)
- `data/kmeans_regimes.csv` — первые 10 строк чтобы понять структуру

---

## Диагноз — почему это нужно менять

Из диагностики последнего прогона:
```
macro_entry_blocked = 2.07M баров  ← 49% всех z-triggers заблокированы
```

Из реальных данных `kmeans_regimes.csv`:
```
Regime 0 (Trend):    1470 дней = 26.4%  ← ЗАБЛОКИРОВАН полностью
Regime 1 (Sideways): 4007 дней = 72.0%  ← единственный разрешённый
Regime 2 (Panic):      90 дней =  1.6%  ← заблокирован (правильно)
```

**Текущая логика** (`filters.py:157`):
```python
if self._km and self._km.get(d, 1) != 1:  # K-Means: only Sideways=1 allowed
    return True  # ← блокирует И Trend=0, И Panic=2
```

**Проблема**: Trend=0 занимает 26% всех дней и блокируется полностью,
хотя для FX stat-arb это не обоснованно. Macro "тренд" = широкий рынок трендит.
Но FX пары (EURUSD/GBPUSD) продолжают mean-revert в Trend режиме —
тренд в акциях не уничтожает корреляционную структуру FX. Panic=2 — другое дело:
корреляции реально ломаются, выходы разбегаются → hard block оправдан.

Результат: теряем 26% торговых дней без реальной защиты.

---

## Изменение 1 — `filters.py`: разделить Trend и Panic

Найди метод `is_entry_blocked` в классе `MacroFilter` (строка ~152).

### Текущий код:
```python
def is_entry_blocked(self, ts) -> bool:
    """True → do not open new positions on this bar."""
    d = _to_date(ts)
    if self._alert.get(d, False):          # VIX9D backwardation
        return True
    if self._km and self._km.get(d, 1) != 1:  # K-Means: only Sideways=1 allowed
        return True
    if self._hmm.get(d, 0) == 1:           # global SPY HMM panic
        return True
    return False
```

### Новый код:
```python
def is_entry_blocked(self, ts) -> bool:
    """True → do not open new positions on this bar.

    K-Means gate change: только Panic=2 является hard block.
    Trend=0 блокировал 26% дней без обоснования для FX stat-arb:
    macro trend ≠ breakdown of FX pair correlation structure.
    Panic=2 (1.6% дней) — реальный риск разрыва корреляций → оставляем.
    """
    d = _to_date(ts)
    if self._alert.get(d, False):          # VIX9D backwardation
        return True
    if self._km and self._km.get(d, 1) == 2:  # K-Means: только Panic=2 блокирует
        return True
    if self._hmm.get(d, 0) == 1:           # global SPY HMM panic
        return True
    return False
```

**Единственное изменение**: `!= 1` → `== 2`

---

## Изменение 2 — `filters.py`: добавить soft multiplier для Trend режима

Добавить новый метод в класс `MacroFilter` сразу после `is_entry_blocked`:

```python
def km_size_multiplier(self, ts) -> float:
    """Soft position size scaling by K-Means regime.

    Вместо hard block на Trend=0 возвращаем multiplier < 1.0:
    - Sideways=1: полный размер (нормальный торговый день)
    - Trend=0:    уменьшенный размер (macro trend → чуть хуже условия для mean-rev)
    - Panic=2:    ноль (is_entry_blocked уже блокирует, но на случай если вызван отдельно)

    Используется в step4a_backtest.py вместо is_entry_blocked для K-Means части.
    """
    d = _to_date(ts)
    regime = self._km.get(d, 1)
    if regime == 1:
        return 1.0   # Sideways — торгуем полным размером
    if regime == 0:
        return 0.5   # Trend — режем размер вдвое, не блокируем
    return 0.0       # Panic — не торгуем
```

---

## Изменение 3 — `step4a_backtest.py`: применить soft multiplier при входе

Найди блок Entry gate (строка ~570 приблизительно), где вызывается `position_size()`.
Там есть условие `if macro_filter is not None and macro_filter.is_entry_blocked(ts):`
и чуть дальше вызов `position_size(...)`.

Найди строку где вычисляется `sz` для sizing (строка ~582):
```python
if sizing_args:
    sz = position_size(
        pair_name, ts,
        sizing_args["regimes"],
        sizing_args["iv_mult_s"],
        sizing_args["mc_conf"],
        sizing_args.get("macro_alert_s"),
        sizing_args.get("global_hmm_s"),
    )
    if sz < MIN_POSITION_SIZE:
        diag["size_too_small"] += 1
        continue
```

Добавить применение K-Means multiplier сразу ПОСЛЕ вычисления `sz`:
```python
if sizing_args:
    sz = position_size(
        pair_name, ts,
        sizing_args["regimes"],
        sizing_args["iv_mult_s"],
        sizing_args["mc_conf"],
        sizing_args.get("macro_alert_s"),
        sizing_args.get("global_hmm_s"),
    )
    # Применяем K-Means soft multiplier: Trend=0 → ×0.5, Panic=2 → ×0.0
    # (Panic=2 уже заблокирован в is_entry_blocked, но добавляем для надёжности)
    if macro_filter is not None:
        sz *= macro_filter.km_size_multiplier(ts)
    if sz < MIN_POSITION_SIZE:
        diag["size_too_small"] += 1
        continue
```

---

## Верификация

### Синтаксическая проверка
```bash
python -c "import ast; ast.parse(open('filters.py').read()); print('filters.py OK')"
python -c "import ast; ast.parse(open('step4a_backtest.py').read()); print('step4a_backtest.py OK')"
```

### Проверка логики MacroFilter
```bash
python -c "
import pandas as pd, sys
sys.path.insert(0, '.')
from filters import MacroFilter

# Тест: создаём фильтр с известными режимами
idx = pd.date_range('2020-01-01', periods=3, freq='D', tz='UTC')
km = pd.Series([0, 1, 2], index=idx)  # Trend, Sideways, Panic
mf = MacroFilter(None, None, km)

print('Trend=0  is_entry_blocked:', mf.is_entry_blocked(idx[0]))   # ожидаем False
print('Sideways=1 is_entry_blocked:', mf.is_entry_blocked(idx[1])) # ожидаем False
print('Panic=2  is_entry_blocked:', mf.is_entry_blocked(idx[2]))   # ожидаем True

print('km_size_multiplier Trend=0:', mf.km_size_multiplier(idx[0]))   # ожидаем 0.5
print('km_size_multiplier Sideways=1:', mf.km_size_multiplier(idx[1])) # ожидаем 1.0
print('km_size_multiplier Panic=2:', mf.km_size_multiplier(idx[2]))    # ожидаем 0.0
"
```

### Smoke-test backtest
```bash
BACKTEST_RECENT_BARS=43200 python step4a_backtest.py 2>&1
```

---

## Ожидаемый результат

В выводе по парам строка `blocks:` должна измениться:
```
# Было:
blocks: macro_entry_blocked=2.07M

# После патча:
blocks: macro_entry_blocked=~90K   (только Panic=2 дни, 1.6% × 5.5M баров)
```

z-triggers которые раньше блокировались в Trend=0 теперь пойдут дальше —
с размером позиции ×0.5 вместо 0. Это добавит сделки в 26% ранее заблокированных дней.

---

## Что НЕ трогать

- `is_force_close()` — оставить как есть (force-close только на Panic=2, это правильно)
- `force_close_reason()` — не трогать
- `CointegrationFilter`, `HurstFilter` — не трогать
- Всё остальное в `step4a_backtest.py` — не трогать
- `config.py` — не трогать в этом промпте
