import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import DATA_DIR, OUTPUT_DIR

def main():
    # Load daily closes
    path = DATA_DIR / "closes_daily.csv"
    if not path.exists():
        print(f"Error: {path} not found. Run step0b_generate_daily.py first.")
        return

    print(f"Loading {path}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # Calculate log-returns
    # Using recent data (e.g., last 2 years) for a more relevant correlation matrix
    recent_years = 3
    cutoff = df.index.max() - pd.DateOffset(years=recent_years)
    df_recent = df[df.index >= cutoff]
    
    print(f"Calculating correlations for the last {recent_years} years ({len(df_recent)} days)...")
    
    # Drop columns with too many NaNs in the period
    df_recent = df_recent.dropna(axis=1, thresh=len(df_recent) * 0.7)
    
    log_ret = np.log(df_recent / df_recent.shift(1))
    corr_matrix = log_ret.corr()
    
    # Plotting
    plt.figure(figsize=(24, 20))
    sns.set_theme(style="white")
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=False, fmt=".2f")
    
    plt.title(f"FX Correlation Matrix (Last {recent_years} Years)", fontsize=24, fontweight='bold')
    
    # Save the plot
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "correlation_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {out_path}")
    
    # Print top correlated pairs
    print("\nTop 15 Correlated Pairs:")
    # Extract lower triangle without diagonal
    sol = (corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
                     .stack()
                     .sort_values(ascending=False))
    
    for i, ((t1, t2), val) in enumerate(sol.head(15).items()):
        print(f"{i+1:2d}. {t1}-{t2:7s} : {val:.4f}")

if __name__ == "__main__":
    main()
