# FX Pairs Trading System (fp_fx_betatest)

Statistical arbitrage engine for FX markets. Discovers cointegrated currency pairs, applies a multi-layer risk/regime filter stack, runs a Kalman-filter backtest, and produces a visual dashboard.

---

## Pipeline Overview

```
Step 0  Pre-process       FX CSVs → 15-min Parquet, daily closes
Step 1  Download          15-min intraday via Polygon + daily closes via yfinance
Step 2  Pair Discovery    Rolling EG cointegration (equities) or FX Johansen
Step 3  Filters & Sizing  HMM · K-Means · IV · OU Monte Carlo · WFO · grid search
Step 4  Backtest          Kalman-filter spread, circuit breaker, full cost model
Step 5  Analysis          Dashboard · stress test · bootstrap · signal plots
Step 6  Paper Trading     Live signal monitoring
Step 7  Universe Curation Top-pair selection for live deployment
Step 8  Universe Miner    Exhaustive candidate mining
```

Run interactively:

```bash
python RUN_ALL.py
```

Or pass steps directly:

```bash
python RUN_ALL.py all          # default: 2c 3a 3b 3c 3d 3e 4a 5d
python RUN_ALL.py 2c 4a 5d    # FX major pairs → backtest → dashboard
python RUN_ALL.py 4a 5e       # backtest + stress test only
```

---

## Directory Layout

```
fp_fx_betatest/
├── config.py                  # All parameters (Z-scores, costs, WFO windows, …)
├── RUN_ALL.py                 # Interactive pipeline runner
├── requirements.txt
│
├── Core library modules
│   ├── data_loader.py         # Parquet/CSV loader with column pruning
│   ├── filters.py             # CointegrationFilter · MacroFilter · HurstFilter · KDE
│   ├── kalman.py              # Time-varying hedge ratio (Kalman filter)
│   └── pairs_universe.py      # 5-layer pair-selection funnel (stocks)
│
├── Pipeline scripts (step0 → step8)
│   ├── step0_preprocess_fx.py / step0b_generate_daily.py
│   ├── step1_download.py / step1a_download.py
│   ├── step2a_pairs.py / step2b_pairs_fx.py
│   ├── step3a_hmm.py … step3j_wfo.py / step3l_fx_sessions.py
│   ├── step4a_backtest.py / step4b_backtest_strict.py / step4c_sniper_grid.py
│   └── step5a_grid_exit.py … step5g_plot_best.py / step6_paper_trade.py …
│
├── Standalone utilities
│   ├── download.py                    # data quality audit (daily + intraday)
│   ├── download_incremental.py        # incremental intraday download
│   ├── download_redownload.py         # re-download failed tickers
│   ├── download_retry.py              # retry with exponential backoff
│   ├── check_recent_correlations.py   # 90/180-day correlation audit
│   ├── check_wfo_fx.py                # WFO diagnostic for FX pairs
│   ├── exhaustive_wfo_diag.py         # brute-force WFO across all tickers
│   ├── plot_correlations.py           # 3-year correlation heatmap
│   ├── enrich_pairs.py                # refresh beta on WFO top-50
│   └── gen_full_sessions.py           # generate FX trading-session windows
│
├── data/                      # Downloaded & processed data (git-ignored)
├── output/                    # Charts, CSVs, reports (git-ignored)
└── archive/                   # Obsolete migration & debug scripts (safe to ignore)
```

---

## Key Parameters (`config.py`)

| Parameter | Value | Purpose |
|---|---|---|
| `PAIRS` | 8 FX pairs | Trading universe |
| `BAR_MINUTES` | 15 | Intraday bar size |
| `ENTRY_Z` | 2.0 | Z-score entry threshold (normal regime) |
| `ENTRY_Z_VOLATILE` | 2.8 | Z-score entry threshold (HMM volatile) |
| `EXIT_Z` | 0.0 | Z-score exit (mean reversion) |
| `STOP_Z` | 3.5 | Stop-loss Z-score |
| `CIRCUIT_BREAKER_Z` | 4.5 | Hard-stop + pair suspension |
| `KALMAN_DELTA` | 3e-6 | Process noise (~fixed beta over ~12k days) |
| `COST_MAKER` | ~0.0075% | Limit order cost (entries/TP) |
| `COST_TAKER` | ~0.015% | Market order cost (stops/panic) |
| `WFO_TRAIN_MONTHS` | 36 | Walk-forward training window |
| `WFO_TEST_MONTHS` | 12 | Walk-forward OOS window |
| `INITIAL_CAPITAL` | $10 000 | Starting portfolio balance |

---

## Filter Stack (Step 3)

Entry is gated by five independent layers applied in sequence:

1. **HMM** (per-pair, 2-state) — volatile regime blocks entries; size scales to 30%
2. **K-Means macro** (SPY + VIX, 3 clusters) — only `Sideways` allows new entries; `Panic` force-closes all positions
3. **Implied Volatility** — IV > 75th percentile → half position size
4. **OU Monte Carlo** — 10k-path simulation; `win_rate` scales final size
5. **Hurst / KDE** — blocks entry when spread is trending (H > 0.55) or at a low-density Z node

---

## Position Sizing

```
final_size = base_size
           × regime_mult      # 1.0 normal / 0.3 volatile (per-pair HMM)
           × hmm_panic_mult   # 0.333 on global SPY panic
           × iv_mult          # 0.5 – 1.0 (IV percentile)
           × mc_confidence    # OU win_rate

# Trade skipped entirely if final_size < MIN_POSITION_SIZE (0.15)
```

---

## Data Sources

| Source | Data | Used in |
|---|---|---|
| Polygon.io | 15-min intraday OHLCV | Signals, backtest |
| Yahoo Finance | Daily closes (20 yr) | Pair selection |

Set your Polygon API key in `.env`:

```
POLYGON_API_KEY=your_key_here
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## `archive/` Directory

Contains migration scripts (`apply_v2.py` – `apply_v8_paper.py`) that were one-time patches already applied, plus superseded script versions (`step4_backtest.py`, `step5_vwap.py`) and ad-hoc debug utilities. Safe to ignore entirely.
