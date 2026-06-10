from pathlib import Path

PAIRS = [
    ("eurusd", "gbpusd"),
    ("audusd", "nzdusd"),
    ("usdcad", "usdchf"),
    ("eurjpy", "gbpjpy"),
    ("audjpy", "nzdjpy"),
    ("gbpusd", "usdchf"),
    ("audusd", "usdcad"),
]

# Pure FX pairs only — equity indices and commodities removed to prevent
# borrow_cost unit inflation (index prices 5k–35k vs FX prices ~1.0).
TICKERS = [
    # AUD crosses
    "audcad", "audchf", "audjpy", "audnzd", "audusd",
    # CAD / CHF crosses
    "cadchf", "cadjpy", "chfjpy",
    # EUR crosses
    "euraud", "eurcad", "eurchf", "eurczk", "eurdkk",
    "eurgbp", "eurhuf", "eurjpy", "eurnok", "eurnzd",
    "eurpln", "eursek", "eurtry", "eurusd",
    # GBP crosses
    "gbpaud", "gbpcad", "gbpchf", "gbpjpy", "gbpnzd", "gbpusd",
    # NZD crosses
    "nzdcad", "nzdchf", "nzdjpy", "nzdusd",
    # SGD / ZAR crosses
    "sgdjpy", "zarjpy",
    # USD crosses
    "usdcad", "usdchf", "usdczk", "usddkk",
    "usdhkd", "usdhuf", "usdjpy", "usdmxn",
    "usdnok", "usdpln", "usdsek", "usdsgd",
    "usdtry", "usdzar",
]

# Equity indices and commodities explicitly excluded from pairing
# (price scale 5k–35k breaks borrow_cost formula; different regime drivers)
NON_FX_TICKERS = {
    "auxaud", "etxeur", "frxeur", "grxeur", "hkxhkd",
    "jpxjpy", "nsxusd", "spxusd", "udxusd", "ukxgbp",
    "bcousd", "wtiusd", "xagusd", "xauaud", "xauchf",
    "xaueur", "xaugbp", "xauusd",
}

START_DATE   = "2006-01-01"
END_DATE     = "2026-04-28"
DAILY_START  = "2006-01-01"

REQUEST_SLEEP = 25

RTH_START    = "00:00"
RTH_END      = "23:59"
SIGNAL_START = "00:00"

BAR_MINUTES  = 1                               # ← 1-min bars (M1 Dukascopy data)
BARS_PER_DAY = int(24 * 60 / BAR_MINUTES)      # 1440 for 1-min

CLOSES_FILE  = f"closes_{BAR_MINUTES}min.csv"
VOLUMES_FILE = f"volumes_{BAR_MINUTES}min.csv"
VWAPS_FILE   = f"vwaps_{BAR_MINUTES}min.csv"
FAST_RUN_BARS = 0          # 0 = full dataset, >0 = load only last N bars
WFO_SKIP_VOLUMES = False   # True = skip volume CSV loads in WFO/backtests/profilers

RECENT_BARS  = BARS_PER_DAY * 92             # ~4.6 months regardless of bar size
TRAIN_RATIO  = 0.70   # first 70% → find pairs; last 30% → out-of-sample test

CORR_THRESHOLD = 0.5
CORR_TOP_N = 10
COINT_TOP_N = 15   # step2b saves the full local universe; step2a is capped by this

# ── Z-score thresholds — calibrated for 1-min FX majors ─────────────────────
# Majors are liquid and tight-spread → Z-score is cleaner than exotics
ENTRY_Z     = 1.8
EXIT_Z      = 0.5    # faster profit realization for Zero profitable-day constraints
STOP_Z      = 3.2
ENTRY_Z_MIN = 1.7    # leave a small floor below live ENTRY_Z for WFO/grid space
ENTRY_Z_VOLATILE = 2.4   # still stricter than normal, but not trade-starving on 1-min FX

# ── Regime-Conditioned Profiling (RCDP) ─────────────────────────────────────
# Wider grids for per-regime grid search (regime_profiler.py)
RCDP_ENTRY_GRID  = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
RCDP_EXIT_GRID   = [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
RCDP_STOP_GRID   = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0]
RCDP_MIN_TRADES_VOLATILE = 5   # volatile slice has fewer bars — lower threshold

# Transaction costs (per side, per leg, as fraction of price)
COST_COMMISSION = 0.0000     # commissions often zero or fixed per lot in FX
COST_SPREAD     = 0.0001     # 1 pip on EURUSD is ~0.01%
COST_SLIPPAGE   = 0.00005    # 0.5 pip

# Passive Aggressor: Limit orders inside spread for entries and TP (earn spread/rebate)
COST_MAKER      = COST_SPREAD / 2 + COST_SLIPPAGE / 2      # ~0.0075%
COST_TAKER      = COST_SPREAD + COST_SLIPPAGE              # ~0.015%

# ── Idiosyncratic Circuit Breaker ────────────────────────────────────────────
CIRCUIT_BREAKER_Z = 4.5  # Max Z-score before immediate hard-stop and pair block

BORROW_RATE_ANNUAL = 0.005  # FX overnight swap cost (0.5% p.a.) — prop firm CFD financing

KALMAN_DELTA = 5e-5  # 1-min majors: adapt over ~20,000 bars ≈ 14 trading days

# ── Volume-Weighted Z-score (VW-Z) & RVOL Gate ───────────────────────────────
USE_VWZ          = True
VWZ_MIN_VOLUME   = 1.0
USE_RVOL_GATE    = True
RVOL_THRESHOLD   = 0.60    # looser gate: keep thin-market protection without starving trades
RVOL_WINDOW      = 200     # ~3.3 hours rolling baseline for "normal" volume

# ── Volume Zones (HVN/LVN profile) ───────────────────────────────────────────
# Empirical finding (step3k_volume_zones.py): for a MEAN-REVERTING spread the
# volume piles up at the mean (HVN = POC = the reversion TARGET) and the entry
# extremes sit in thin/NEUTRAL territory — so a static "enter only in an HVN bin"
# gate is inert. The two genuinely actionable signals are:
#   1. MAGNET STRENGTH (per-pair): how concentrated volume is in the value area.
#      A strong magnet → reliable pull back to the mean → trade the pair.
#   2. ACCEPTANCE vs REJECTION at the extreme (per-bar): is volume BUILDING as the
#      spread reaches the extreme (acceptance → breakout risk) or DRYING UP
#      (rejection → snapback). Measured by relative volume (rvol) at the trigger.
# Volume comes from CME FX futures if data/futures_volumes_*.parquet exists, else
# Dukascopy tick volume.
USE_VOLUME_ZONES   = False   # master switch — keep OFF until zones are profiled
ZONE_MODE          = "REJECT"  # "REJECT" = fade drying extremes (rvol low) |
                               # "ACCEPT"  = fade churning extremes (rvol high)
ZONE_RVOL_SPLIT    = 1.0     # rvol <= split → drying (REJECT bucket); > split → building
ZONE_MIN_MAGNET    = 0.0     # per-pair gate: skip pairs with value-area vol share below this
VOLUME_ZONE_BINS   = 60      # histogram resolution of the Volume-at-Z profile
VOLUME_ZONE_VA_PCT = 0.70    # value-area fraction (classic Market Profile = 70%)
VOLUME_ZONE_HVN_Q  = 0.80    # bins above this volume-quantile are High-Volume Nodes
VOLUME_ZONE_LVN_Q  = 0.20    # bins below this volume-quantile are Low-Volume Nodes
VOLUME_ZONE_WINDOW = 5000    # trailing bars used to (re)build the causal profile
VOLUME_ZONE_STEP   = 500     # rebuild the profile every N bars (speed/recency trade-off)

# LVN-anchored stop: park the stop just beyond the nearest Low-Volume Node so the
# spread is not stopped inside a churn zone. Clamped to the [min,max] Z band below.
USE_LVN_STOP       = False   # requires USE_VOLUME_ZONES
LVN_STOP_Z_MIN     = 2.6     # never tighter than this
LVN_STOP_Z_MAX     = 5.0     # never wider than this

# ── VWAP prices for Kalman input ─────────────────────────────────────────────
USE_VWAP        = True
USE_VWAP_MTF    = False    # disable hard higher-tf anchor gate; keep VWAP itself for Kalman input
VWAP_MTF_TF     = "15min"  # 15-min anchor for 1-min strategy

# ── Return-spread mode (for 1-min intraday trading) ──────────────────────────
# Uses cumulative returns over RETURN_WINDOW bars instead of price levels.
# Return spread is stationary by construction → short half-life → many signals.
# Signal: cum_r1[t] - beta × cum_r2[t]  over the last N bars.
USE_RETURN_SPREAD  = True
RETURN_WINDOW      = 30   # bars over which to accumulate returns (30 min at 1-min)

# ── Velocity gate ─────────────────────────────────────────────────────────────
USE_VELOCITY_GATE = False   # blocks first-touch mean-reversion entries with better observed Sharpe
VELOCITY_WINDOW   = 5    # bars  →  5 × 1min = 5-min momentum window

# ── Multi-Timeframe Z-score confirmation (MTF) ────────────────────────────────
MTF_CONFIRM  = False   # disabled: kills too many entries on 1-min return-spread
MTF_Z_MIN    = 0.3
MTF_RESAMPLE = "5min"   # resample Z to 5-min (was 1H on 15-min bars)

# ── Realized Variance Ratio (RVR) ─────────────────────────────────────────────
RVR_FILTER       = True
RVR_WINDOW_SHORT = 10    # bars  →  10 min  (short-term variance)
RVR_WINDOW_LONG  = 60    # bars  →  60 min  (baseline variance)
RVR_MAX          = 2.5

# ── Live rolling correlation at entry ─────────────────────────────────────────
LIVE_CORR_FILTER = True
LIVE_CORR_WINDOW = 60    # bars  →  60 min rolling correlation
LIVE_CORR_MIN    = 0.20  # loosened: 0.45 was killing too many entries on return-spread

# ── Session filter ────────────────────────────────────────────────────────────
SESSION_FILTER = False   # 24h FX plus event blackout is sufficient for Zero profile

# ── Macro event blackout ──────────────────────────────────────────────────────
EVENT_FILTER      = True
EVENT_BARS_BEFORE = 10     # bars  →  10 min before release
EVENT_BARS_AFTER  = 30     # bars  →  30 min after  release

# ── Copula signals ───────────────────────────────────────────────────────────
USE_COPULA         = False   # redundant with live_corr and innov_var for this profile
COPULA_WINDOW      = 500    # bars  →  ~8 hours (enough warmup < 1 trading day)
COPULA_Z_MIN       = 0.5
COPULA_LAMBDA_MIN  = 0.10

# ── Kalman innovation variance spike ─────────────────────────────────────────
KALMAN_INNOV_FILTER = False  # disabled: redundant with live_corr on return-spread mode
KALMAN_INNOV_WINDOW = 200
KALMAN_INNOV_MAX    = 3.0
# 3e-6 → beta adapts over ~333,000 bars (~12,800 trading days) — near-fixed beta
# 3e-5 → adapts over ~33,000 bars (too fast — innovations become white noise)
# 1e-4 → adapts over ~10,000 bars (way too fast — kills mean-reversion signal)

HALF_LIFE_MAX_BARS = 120  # 1-min: max 120 bars = 2 hours (reject sluggish pairs)

HURST_MAX = 0.50    # max Hurst exponent for pair spread (< 0.5 = mean reverting)

# ── Hurst Entry Gate (dynamic, per-trade) ────────────────────────────────────
# Computed lazily on the DAILY spread when |z| >= entry threshold.
# Blocks entries when the spread is trending (structural drift), regardless of macro regime.
HURST_ENTRY_WINDOW = 60     # daily bars of spread history for rolling Hurst at entry time
HURST_ENTRY_SOFT_MAX = 0.50 # above this: guarded mean-reversion mode only
HURST_ENTRY_MAX      = 0.55 # above this: reject the trade entirely
HURST_GUARDED_ENTRY_ADD = 0.20
HURST_GUARDED_EXIT_Z    = 0.50
CORR_MIN  = 0.50    # min log-return correlation over training window
RECENT_CORR_DAYS = 120   # rolling window for recent correlation check (calendar days)
RECENT_CORR_MIN  = 0.50  # pair disabled if recent 120-day correlation drops below this

# ── Gatev et al. SSD pre-filter (pairs_universe.py Layer 2b) ───────────────────
# Sum of Squared Deviations of normalised cumulative return indices.
# Keeps only the X% of pairs with smallest SSD — fast pre-filter before Johansen.
SSD_PERCENTILE = 80   # keep bottom 80% by SSD (discard top 20% most divergent)

# ── Phase 1: rolling window cointegration ────────────────────────────────────
COINT_WINDOW_DAYS  = 90    # rolling window for EG cointegration test (trading days)
COINT_BREAK_P      = 0.05  # if rolling coint p > this during backtest → suspend pair
COINT_RECHECK_DAYS = 5     # recheck coint every N trading days during backtest

# ── Phase 4: K-Means macro regime ────────────────────────────────────────────
KMEANS_N_CLUSTERS = 3      # 0=Trend, 1=Sideways, 2=Panic
KMEANS_VOL_WINDOW = 20     # rolling window for macro features (trading days)

# ── Walk-Forward Optimization (WFO) ───────────────────────────────────────
# Gatev et al. (2006) baseline: 12-month train, 6-month OOS, 6-month step
WFO_TRAIN_MONTHS = 6    # ~180 days formation window
WFO_TEST_MONTHS  = 2    # ~60 days trading window (OOS)
WFO_STEP_MONTHS  = 2    # non-overlapping sliding step
WFO_MIN_TRADES   = 1    # minimum trades per OOS window to count as valid (2m OOS windows are sparse)
WFO_EXPANDING    = False # True = expanding window (train_start anchored to data origin)
                         # False = rolling window (classic Gatev fixed-width train)

ZERO_ROLLING_WINDOW_DAYS   = 60
ZERO_MIN_PROFIT_DAYS_30D   = 3   # loosened for sparse early OOS blocks
ZERO_MIN_PROFIT_DAYS_60D   = 6
ZERO_MIN_PROFIT_DAY_PCT    = 0.0025

PAIR_MAX_LOSS = -75.0   # one broken pair must not consume the full Zero trailing buffer

IV_LOOKBACK   = 60         # days for IV percentile calculation
IV_THRESHOLD  = 75         # percentile above which → reduce position size
IV_SIZE_HIGH  = 0.5        # position size when IV is elevated
IV_SIZE_NORM  = 1.0        # position size when IV is normal

# Dynamic position sizing (step9)
REGIME_MULT_NORMAL   = 1.0   # full size in normal regime
REGIME_MULT_VOLATILE = 0.3   # reduced size in volatile regime (per-pair HMM)
HMM_PANIC_MULT       = 0.333 # global macro HMM panic: cut ALL sizes by 3
IV_MULT_MAX          = 1.0   # full size when IV is at its lowest
IV_MULT_MIN          = 0.5   # half size when IV is at its highest
MIN_POSITION_SIZE    = 0.05  # allow small probes when multiple soft risk multipliers overlap

INITIAL_CAPITAL = 5_000         # FundingPips Zero account size in USD
LEVERAGE        = 50            # 1:50 leverage — max notional = INITIAL_CAPITAL × LEVERAGE
ALLOCATION_METHOD = "equal"       # "equal" | "sharpe" | "markowitz" (MVO)
MAX_PAIR_WEIGHT   = 0.10        # cap pair concentration for Zero account

TARGET_RISK_USD = 20.0               # Zero-safe: total stop-risk stays below daily loss limit
ENTRY_COST_SAFETY = 3.0              # expected gross edge must cover costs by this multiple

# ── FundingPips Zero profile ────────────────────────────────────────────────
# Zero account is pre-passed, funded at $5k, with live-account style risk rules.
ACCOUNT_PROVIDER = "FundingPips"
ACCOUNT_MODEL    = "Zero"
ACCOUNT_SIZE     = 5_000
ZERO_MAX_TRAILING_LOSS_PCT = 0.05   # 5% trailing loss until 5% profit, then locked at initial balance
ZERO_MAX_DAILY_LOSS_PCT    = 0.03   # 3% daily loss limit
ZERO_MAX_INACTIVE_DAYS      = 30
ZERO_AVOID_NEWS_TRADING     = True

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# ── The Black Swan Hedge (Tail Risk Convexity) ───────────────────────────────
TAIL_HEDGE_DRAG_ANNUAL = 0.015  # 1.5% annual drag on portfolio (buying far OTM Puts)
TAIL_HEDGE_PAYOUT_MULT = 10.0   # Convexity multiplier when HMM detects Panic
