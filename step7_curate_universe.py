import pandas as pd
from pathlib import Path

def curate_universe():
    params_path = Path("data/wfo_params.csv")
    if not params_path.exists():
        print("Error: data/wfo_params.csv not found. Run WFO first.")
        return

    df = pd.read_csv(params_path)
    
    # Aggregate stats per pair
    summary = df.groupby("pair").agg({
        "oos_pnl": "sum",
        "oos_trades": "sum",
        "oos_sharpe": "mean"
    }).sort_values("oos_pnl", ascending=False)
    
    print("Full Universe OOS Summary:")
    print(summary)
    
    # Filter winners: Positive P&L and at least 5 trades
    winners = summary[(summary["oos_pnl"] > 0) & (summary["oos_trades"] >= 5)]
    
    print("\nSelected Winners:")
    print(winners)
    
    # Load original pairs to get any extra metadata if needed
    original_pairs = pd.read_csv("data/pairs_selected.csv")
    
    # Create the live universe
    live_pairs = original_pairs[original_pairs["pair"].isin(winners.index)]
    
    live_path = Path("data/pairs_live.csv")
    live_pairs.to_csv(live_path, index=False)
    print(f"\nSaved {len(live_pairs)} winners to {live_path}")
    
    # Optional: Update pairs_selected.csv to narrow down future runs
    # live_pairs.to_csv("data/pairs_selected.csv", index=False)

if __name__ == "__main__":
    curate_universe()
