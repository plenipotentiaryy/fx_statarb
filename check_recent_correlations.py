import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR

def get_top_correlations(df, days):
    cutoff = df.index.max() - pd.Timedelta(days=days)
    slice_df = df[df.index >= cutoff]
    
    # Drop columns with insufficient data in the slice
    slice_df = slice_df.dropna(axis=1, thresh=len(slice_df) * 0.9)
    
    log_ret = np.log(slice_df / slice_df.shift(1)).dropna(how='all')
    corr_matrix = log_ret.corr()
    
    # Extract pairs
    sol = (corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
                     .stack()
                     .sort_values(ascending=False))
    return sol

def main():
    path = DATA_DIR / "closes_daily.csv"
    if not path.exists():
        print("Error: data/closes_daily.csv not found.")
        return

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)

    print(f"Analyzing correlations for {len(df.columns)} assets...")
    
    corr_180 = get_top_correlations(df, 180)
    corr_90  = get_top_correlations(df, 90)
    
    print("\n" + "="*50)
    print("TOP CORRELATIONS (LAST 180 DAYS)")
    print("="*50)
    print(corr_180.head(15).to_string())

    print("\n" + "="*50)
    print("TOP CORRELATIONS (LAST 90 DAYS)")
    print("="*50)
    print(corr_90.head(15).to_string())

    # Find stable correlations (high in both)
    print("\n" + "="*50)
    print("STABLE HIGH CORRELATIONS (Both periods > 0.8)")
    print("="*50)
    
    stable = []
    for pair, val90 in corr_90.items():
        if val90 > 0.8:
            if pair in corr_180 and corr_180[pair] > 0.8:
                stable.append({
                    "pair": f"{pair[0]}-{pair[1]}",
                    "corr_180": round(corr_180[pair], 4),
                    "corr_90": round(val90, 4),
                    "diff": round(val90 - corr_180[pair], 4)
                })
    
    if stable:
        stable_df = pd.DataFrame(stable).sort_values("corr_90", ascending=False)
        print(stable_df.to_string(index=False))
    else:
        print("No pairs found with correlation > 0.8 in both periods.")

    # Check our selected pairs
    pairs_path = DATA_DIR / "pairs_selected.csv"
    if pairs_path.exists():
        print("\n" + "="*50)
        print("RECENT CORRELATION OF SELECTED PAIRS")
        print("="*50)
        selected = pd.read_csv(pairs_path)["pair"].tolist()
        results = []
        for p in selected:
            t1, t2 = p.split("-")
            c180 = corr_180.get((t1, t2), corr_180.get((t2, t1), np.nan))
            c90  = corr_90.get((t1, t2), corr_90.get((t2, t1), np.nan))
            results.append({"pair": p, "corr_180": c180, "corr_90": c90})
        
        print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
