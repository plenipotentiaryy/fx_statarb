import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from config import DATA_DIR, OUTPUT_DIR, BARS_PER_DAY

def main():
    path = DATA_DIR / "closes_15min.csv"
    if not path.exists():
        print("Error: data/closes_15min.csv not found.")
        return

    print("Loading 15-min data for session analysis...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)

    # We'll analyze the pairs from pairs_selected.csv
    pairs_path = DATA_DIR / "pairs_selected.csv"
    if not pairs_path.exists():
        print("pairs_selected.csv not found. Using top 5 from exhaustive diag if exists...")
        # Fallback or just use a few common pairs
        pairs = ["eurusd-gbpusd", "audusd-nzdusd", "usdcad-usdchf"]
    else:
        pairs = pd.read_csv(pairs_path)["pair"].tolist()

    print(f"Analyzing {len(pairs)} pairs for optimal trading windows...")
    
    session_results = []

    for pair in pairs:
        t1, t2 = pair.split("-")
        if t1 not in df.columns or t2 not in df.columns: continue
        
        # Calculate spread
        # Simple rolling beta for session analysis
        s1, s2 = df[t1], df[t2]
        beta = s1.rolling(252*96).corr(s2) * (s1.rolling(252*96).std() / s2.rolling(252*96).std())
        beta = beta.fillna(method='bfill').fillna(1.0)
        spread = s1 - beta * s2
        
        # Z-score
        z = (spread - spread.rolling(100).mean()) / spread.rolling(100).std()
        
        # Calculate "Mean Reversion Velocity" per hour
        # How much does Z move towards 0 when it's outside 1.0?
        # dZ * -sign(Z)
        dz = z.diff().shift(-1)
        mr_speed = dz * -np.sign(z)
        mr_speed = mr_speed.where(z.abs() > 1.0) # only when stretched
        
        # Volatility
        vol = df[t1].pct_change().abs() + df[t2].pct_change().abs()

        # Group by hour (UTC)
        hourly_stats = pd.DataFrame({
            "mr_speed": mr_speed,
            "vol": vol
        })
        hourly_stats["hour"] = hourly_stats.index.hour
        
        grouped = hourly_stats.groupby("hour").mean()
        
        # Score = Speed * Vol (we want fast reversion in active markets)
        grouped["score"] = grouped["mr_speed"] * grouped["vol"] * 10000
        
        # Find best continuous 8-hour window
        best_start = 0
        max_score = -np.inf
        for h in range(24):
            # Rolling 8h sum
            score_8h = sum(grouped["score"].iloc[(h + i) % 24] for i in range(8))
            if score_8h > max_score:
                max_score = score_8h
                best_start = h
        
        best_end = (best_start + 8) % 24
        
        session_results.append({
            "pair": pair,
            "best_start_utc": best_start,
            "best_end_utc": best_end,
            "avg_vol": round(grouped["vol"].mean() * 10000, 2),
            "max_mr_hour": grouped["mr_speed"].idxmax()
        })
        
        print(f"  {pair:15s} | Best Window (UTC): {best_start:02d}:00 - {best_end:02d}:00")

    res_df = pd.DataFrame(session_results)
    res_df.to_csv(DATA_DIR / "pair_sessions.csv", index=False)
    print(f"\nSaved session profiles to data/pair_sessions.csv")

if __name__ == "__main__":
    main()
