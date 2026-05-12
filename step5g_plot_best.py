import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATA_DIR, OUTPUT_DIR

def plot_pair(pair_name):
    t1, t2 = pair_name.split('-')
    
    # 1. Load 15-min data
    df = pd.read_csv(DATA_DIR / "closes_15min.csv", index_col=0, parse_dates=True)
    if t1 not in df.columns or t2 not in df.columns:
        print(f"Tickers {t1} or {t2} not in data.")
        return
    
    # 2. Get latest parameters for this pair from wfo_params
    params_df = pd.read_csv(DATA_DIR / "wfo_params.csv")
    pair_params = params_df[params_df['pair'] == pair_name].iloc[-1]
    
    beta = pair_params['beta']
    entry_z = pair_params['entry_z']
    exit_z = pair_params['exit_z']
    
    # 3. Build spread
    sub = df[[t1, t2]].dropna().tail(2000) # Last 2000 bars (~20 days)
    spread = sub[t1] - beta * sub[t2]
    
    # Rolling stats for Z-score
    window = 100 # matching WFO
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    zscore = (spread - mu) / sigma
    
    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Upper panel: Spread + Mu
    ax1.plot(spread.index, spread, label='Spread', color='steelblue', alpha=0.8)
    ax1.plot(mu.index, mu, label='Moving Mean', color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f"Real Spread: {pair_name} (Beta: {beta:.3f})")
    ax1.legend()
    
    # Lower panel: Z-score + Levels
    ax2.plot(zscore.index, zscore, label='Z-score', color='darkred')
    ax2.axhline(entry_z, color='green', linestyle='--', label=f'Entry (+{entry_z})')
    ax2.axhline(-entry_z, color='green', linestyle='--')
    ax2.axhline(exit_z, color='blue', linestyle=':', label=f'Exit ({exit_z})')
    ax2.axhline(0, color='black', lw=0.8)
    
    # Highlight signals
    longs = zscore[zscore < -entry_z]
    shorts = zscore[zscore > entry_z]
    ax2.scatter(longs.index, longs, color='green', marker='^', s=30, label='Long Signal')
    ax2.scatter(shorts.index, shorts, color='red', marker='v', s=30, label='Short Signal')
    
    ax2.set_title(f"Z-Score Signals (Window={window})")
    ax2.set_ylim(-6, 6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"signals_{pair_name}.png", dpi=150)
    print(f"Plot saved for {pair_name} to output/signals_{pair_name}.png")

if __name__ == "__main__":
    plot_pair("nsxusd-spxusd")
    plot_pair("usdjpy-usdnok")
