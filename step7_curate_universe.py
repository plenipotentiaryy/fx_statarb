import pandas as pd
from pathlib import Path


def curate_universe():
    zero_path = Path("data/pairs_zero_universe.csv")
    selected_path = Path("data/pairs_selected.csv")
    live_path = Path("data/pairs_live.csv")

    if zero_path.exists():
        df = pd.read_csv(zero_path)
        if df.empty:
            print(f"Error: {zero_path} is empty. Run step4a_backtest.py first.")
            return

        winner_mask = df["zero_universe_ok"].fillna(False).astype(bool) if "zero_universe_ok" in df.columns else pd.Series(False, index=df.index)
        winners = df[winner_mask].copy()
        if winners.empty:
            print("No Zero candidates passed the 60d universe gate.")
            return

        sort_cols = [c for c in ["max_60d_pnl", "net_pnl", "max_profit_days_60d", "trades"] if c in winners.columns]
        if sort_cols:
            winners = winners.sort_values(sort_cols, ascending=False)

        if selected_path.exists():
            original_pairs = pd.read_csv(selected_path)
            live_pairs = original_pairs[original_pairs["pair"].isin(winners["pair"])].copy()
        else:
            live_pairs = winners.copy()

        live_pairs.to_csv(live_path, index=False)
        print(f"Saved {len(live_pairs)} Zero candidates to {live_path}")
        return

    params_path = Path("data/wfo_params.csv")
    if not params_path.exists():
        print("Error: data/pairs_zero_universe.csv or data/wfo_params.csv not found.")
        return

    df = pd.read_csv(params_path)
    summary = df.groupby("pair").agg({
        "oos_pnl": "sum",
        "oos_trades": "sum",
        "oos_sharpe": "mean",
    }).sort_values("oos_pnl", ascending=False)

    print("Full Universe OOS Summary:")
    print(summary)

    winners = summary[(summary["oos_pnl"] > 0) & (summary["oos_trades"] >= 5)]
    print("\nSelected Winners:")
    print(winners)

    original_pairs = pd.read_csv(selected_path)
    live_pairs = original_pairs[original_pairs["pair"].isin(winners.index)]
    live_pairs.to_csv(live_path, index=False)
    print(f"\nSaved {len(live_pairs)} winners to {live_path}")


if __name__ == "__main__":
    curate_universe()
