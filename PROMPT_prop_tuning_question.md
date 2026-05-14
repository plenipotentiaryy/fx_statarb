# Вопрос: как подогнать FX stat-arb систему под условия пропа FundingPips Zero

## Контекст системы

AFES — FX pairs trading (stat-arb) на 1-минутных барах.
Пары: EURUSD, GBPUSD, AUDUSD, NZDUSD (4 пары, 2 leg каждая → max 4 открытых позиции).
Сигнал: Kalman-фильтр spread → rolling z-score → entry при |z| > threshold.
Исполнение: лимитные ордера с `ExecutionSimulator` (slippage, TOXIC_FLOW_CANCEL, LIMIT_NOT_FILLED).

---

## Правила FundingPips Zero ($5,000 счёт)

```
INITIAL_CAPITAL          = $5,000
LEVERAGE                 = 50x        → max notional $250,000
ZERO_MAX_DAILY_LOSS_PCT  = 3%         → $150/день hard limit
ZERO_MAX_TRAILING_LOSS   = 5%         → $250 от пика (trailing, не фиксированный)
ZERO_MAX_INACTIVE_DAYS   = 30         → не более 30 дней без сделок
ZERO_MIN_PROFIT_DAYS_30D = 7          → минимум 7 прибыльных дней за любые 30 дней
ZERO_MIN_PROFIT_DAYS_60D = 14         → минимум 14 прибыльных дней за любые 60 дней
```

Нет минимального profit target — нужна только стабильность и соблюдение лимитов.

---

## Текущие параметры системы (полный список)

### Сигнал и входы
```python
ENTRY_Z          = 2.0    # z-score порог входа (в σ)
EXIT_Z           = 0.3    # выход (z пересекает 0.3 обратно к нулю)
STOP_Z           = 3.5    # стоп-лосс (z уходит ещё дальше)
CIRCUIT_BREAKER  = 4.5    # принудительный выход + блокировка пары
ENTRY_Z_VOLATILE = 3.5    # порог в volatile HMM-режиме (per-pair)
```

### Размер позиции
```python
TARGET_RISK_USD      = 150.0   # целевой риск на сделку ($) = 1σ спреда
MIN_POSITION_SIZE    = 0.15    # минимальный sizing multiplier (ниже — пропускаем)
IV_SIZE_NORM         = 1.0     # базовый IV multiplier
REGIME_MULT_NORMAL   = 1.0     # full size в нормальном режиме (per-pair HMM)
REGIME_MULT_VOLATILE = 0.3     # reduced в volatile режиме
HMM_PANIC_MULT       = 0.333   # глобальный panic (macro HMM) — режет всё на ×3
```

### Транзакционные издержки
```python
COST_MAKER = 0.0075%   # лимитный вход / TP выход (earn half-spread)
COST_TAKER = 0.015%    # рыночный выход (стоп, circuit-breaker)
BORROW_RATE_ANNUAL = 0.5%   # overnight FX swap
```

### Фильтры входа (все включены)
```python
USE_VELOCITY_GATE = True    # ← планируем ОТКЛЮЧИТЬ (блокирует 65% лучших входов)
USE_RVOL_GATE     = True    # volume > 75% rolling avg
RVOL_THRESHOLD    = 0.75
RVR_FILTER        = True    # блокирует trending spread (short_var / long_var > 2.5)
RVR_MAX           = 2.5
LIVE_CORR_FILTER  = True    # rolling 60-min корреляция ног > 0.45
LIVE_CORR_MIN     = 0.45
MTF_CONFIRM       = True    # 5-min Z-score должен подтверждать направление
MTF_Z_MIN         = 0.3
KALMAN_INNOV_FILTER = True  # блокирует если innovation variance ×3 от нормы
KALMAN_INNOV_MAX    = 3.0
USE_COPULA        = True    # Gaussian + Clayton copula confirmation
COPULA_Z_MIN      = 0.5
HURST_ENTRY_MAX   = 0.55    # блокирует trending пары (H > 0.55)
SESSION_FILTER    = True    # торгуем только в определённые сессии
EVENT_FILTER      = True    # blackout ±10/30 баров вокруг NFP/CPI/FOMC
COINT_BREAK_P     = 0.05    # ADF p-value > 0.05 → suspend pair
```

### Аллокация капитала
```python
ALLOCATION_METHOD = "equal"   # равные веса по парам
MAX_PAIR_WEIGHT   = 0.15      # не более 15% капитала на пару → $750 per pair
PAIR_MAX_LOSS     = -1500.0   # отключить пару при кумулятивном убытке $1,500
```

---

## Проблема которую нужно решить

**Диагностика последнего прогона показала:**

| Причина блокировки | Количество баров | % от z-triggers |
|---|---|---|
| `size_too_small` | 4.08M | — | ← уже исправлено (был NaN-баг)
| `macro_entry_blocked` | 2.07M | ~49% | ← легитимно (K-Means Trend/Panic)
| `velocity` (планируем отключить) | ~80% | от оставшихся |
| Финальные сделки | 1–2 трейда | за весь период |

**Итог**: gap между сделками = 4031–4123 дня (лимит = 30 дней).
Даже без velocity gate — неизвестно хватит ли частоты.

**Критическое противоречие по sizing**:
`TARGET_RISK_USD = 150` = 3% от $5,000.
Это значит одна проигрышная сделка = ВЕСЬ дневной лимит убытка.
Два лузера в день = breach. При win_rate 47–50% это реальный сценарий каждую неделю.

---

## Конкретные вопросы

**1. Sizing: какой TARGET_RISK_USD корректен для 3% daily limit?**

Если торгуем несколько пар параллельно (до 4 одновременно), и каждая рискует $150 —
суммарный дневной риск $600 при максимальной загрузке, при лимите $150.
Что должно быть реальным TARGET_RISK_USD чтобы даже при 3 лузерах в день не пробить 3%?
Как правильно считать daily VaR под этот лимит?

**2. Trailing drawdown: как защититься от $250 trailing limit?**

С 50x leverage: нотионал на пару = $750 × 50 = $37,500.
Одно adverse движение 0.67% на парном спреде = $250 = весь trailing drawdown.
Нужно ли снижать LEVERAGE или это покрывается через TARGET_RISK_USD?
Как соотносится PAIR_MAX_LOSS = $1,500 с trailing limit $250 — не слишком ли высокий?

**3. Trade frequency: какие фильтры реально нужны для Zero, а какие избыточны?**

Есть 8 фильтров входа + session + event blackout + macro HMM + coint filter.
Macro HMM уже блокирует 49% баров (K-Means Trend/Panic режимы).
Какие из оставшихся фильтров дают реальный edge, а какие просто убивают частоту
не добавляя защиту (дублирование Hurst/coint/RVR)?
Нужен ли SESSION_FILTER если торгуем round-the-clock FX?

**4. ENTRY_Z = 2.0: правильный ли это уровень для Zero?**

При 2σ входах: ~4.5% баров дают сигнал теоретически, после фильтров → единицы.
Для Zero нужна торговля минимум раз в 30 дней.
При каком ENTRY_Z (с отключённым velocity gate, остальные фильтры сохранены)
можно ожидать достаточной частоты без потери edge?
Есть ли риск curve-fitting при снижении до 1.5?

**5. Profitable days: как структурировать выходы для 7+ прибыльных дней / 30 дней?**

Сейчас EXIT_Z = 0.3 (выход когда spread почти вернулся к нулю).
Типичный holding: несколько часов до нескольких дней.
При низкой частоте торговли (1–3 сделки/неделю) легко не набрать 7 прибыльных дней.
Нужно ли менять EXIT_Z, добавлять partial exits, или это решается только частотой?

**6. Есть ли конфликт между стратегией (редкие, качественные входы) и Zero (частая активность)?**

Stat-arb на FX парах — это терпеливая стратегия с редкими, но качественными сделками.
FundingPips Zero требует стабильную активность каждые 30 дней.
Или эти вещи совместимы при правильных параметрах, или нужна принципиально другая
тактика (например добавить pairs в другие классы активов для заполнения пробелов)?

---

## Что ожидаю от ответа

Конкретные рекомендации по числам для каждого параметра с объяснением логики.
Не общие слова — а: "TARGET_RISK_USD должен быть X потому что Y".
Если параметры конфликтуют между собой (частота vs риск) — скажи прямо и предложи trade-off.
Если стратегия фундаментально не подходит под Zero условия — скажи это честно.
