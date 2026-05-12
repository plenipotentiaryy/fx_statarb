"""
step6_paper_trade.py — Forward Testing (Live Simulation)
Fetches recent 15m data via yfinance, loads ML/MVO state from WFO,
and logs virtual trades.
"""

import time
import datetime
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR, SIGNAL_START, RTH_END
from step3j_wfo import build_signals
from filters import validate_kde_density

PAPER_LOG = OUTPUT_DIR / "paper_trades_live.csv"

def fetch_recent_data(tickers):
    """Fetch last 5 days of 15-minute data from yfinance."""
    if not tickers: return None, None
    print(f"Fetching 15m data for {len(tickers)} tickers via yfinance...")
    
    # Download in bulk
    df = yf.download(tickers, period="5d", interval="15m", progress=False)
    if df.empty:
        return None, None
        
    closes = df['Close'].copy()
    volumes = df['Volume'].copy()
    
    # yfinance returns tz-aware local or UTC depending on version. Ensure US/Eastern
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize('UTC')
        volumes.index = volumes.index.tz_localize('UTC')
        
    closes.index = closes.index.tz_convert('US/Eastern')
    volumes.index = volumes.index.tz_convert('US/Eastern')
    
    # Align to market hours
    closes = closes.between_time(SIGNAL_START, RTH_END)
    volumes = volumes.between_time(SIGNAL_START, RTH_END)
    
    return closes, volumes

def run_iteration():
    state_file = OUTPUT_DIR / "live_state.pkl"
    if not state_file.exists():
        print(f"Error: {state_file} not found. Run step3j_wfo.py first.")
        return
        
    state = joblib.load(state_file)
    window_best = state["window_best"]
    weights = state["optimal_weights"]
    
    active_pairs = [p for p, w in weights.items() if w > 1e-4]
    if not active_pairs:
        print("No active pairs with MVO weight > 0.")
        return
        
    print(f"\n--- Live Iteration: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Loaded {len(active_pairs)} active pairs from live state.")
    
    # Get all unique tickers
    tickers = set()
    for p in active_pairs:
        t1, t2 = p.split("-")
        tickers.add(t1); tickers.add(t2)
        
    closes, volumes = fetch_recent_data(list(tickers))
    if closes is None:
        print("Failed to fetch data.")
        return
        
    trades_taken = []
    
    for pair in active_pairs:
        t1, t2 = pair.split("-")
        params = window_best[pair]
        beta = params.get("dynamic_beta", 1.0)
        hl = params.get("dynamic_hl", 60)
        
        # Build signals for the last 5 days
        try:
            sig = build_signals(closes, volumes, None, t1, t2, beta, hl)
        except Exception as e:
            print(f"Error building signals for {pair}: {e}")
            continue
            
        if len(sig) < 2:
            continue
            
        # Get the latest bar
        last_bar = sig.iloc[-1]
        z = last_bar["zscore"]
        vr = last_bar["rvol"]
        ez = params["entry_z"]
        
        # 1. Microstructure Filter (VW-Z)
        if vr < 0.5:
            continue
            
        # 2. Trigger
        if abs(z) >= ez:
            # 3. Macrostructure Filter (KDE)
            is_kde_valid = validate_kde_density(sig["zscore"], ez, threshold_ratio=0.5)
            if not is_kde_valid:
                print(f"[{pair}] KDE Rejected (LDN) at Z={z:.2f}")
                continue
                
            # 4. ML Sizing (Non-Linear Alpha)
            clf = params.get("ml_model")
            ml_mult = 1.0
            prob = 0.5
            if clf is not None:
                spread_std = last_bar["spread_std"]
                hour = sig.index[-1].hour + sig.index[-1].minute / 60.0
                features = np.array([[abs(z), vr, spread_std, hour]])
                try:
                    prob = clf.predict_proba(features)[0][1]
                    ml_mult = max(0.0, 2.0 * (prob - 0.5))
                except Exception:
                    pass
            
            if ml_mult <= 0:
                print(f"[{pair}] ML Rejected (Prob={prob:.2%}) at Z={z:.2f}")
                continue
                
            # EXECUTE VIRTUAL TRADE
            weight = weights[pair]
            direction = "LONG" if z < 0 else "SHORT"
            trades_taken.append({
                "time": sig.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                "pair": pair,
                "direction": direction,
                "zscore": round(z, 2),
                "prob_win": round(prob, 3),
                "size_mult": round(ml_mult * weight * len(active_pairs), 3)
            })
            print(f"🚀 [{pair}] SIGNAL {direction} Z={z:.2f} | P(Win)={prob:.0%} | Size={ml_mult:.2f}x")
            
    if trades_taken:
        df_trades = pd.DataFrame(trades_taken)
        hdr = not PAPER_LOG.exists()
        df_trades.to_csv(PAPER_LOG, mode='a', header=hdr, index=False)
        print(f"Logged {len(trades_taken)} trades to {PAPER_LOG.name}")
    else:
        print("No active entry signals on the last bar.")

if __name__ == "__main__":
    run_iteration()
