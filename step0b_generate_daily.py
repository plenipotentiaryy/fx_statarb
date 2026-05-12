import pandas as pd
from pathlib import Path
from config import DATA_DIR, CLOSES_FILE

def main():
    input_path = DATA_DIR / f"closes_15min.csv"
    if not input_path.exists():
        input_path = DATA_DIR / f"closes_15min.parquet"
        
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run step0_preprocess_fx.py first.")
        return
        
    print(f"Loading {input_path}...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        
    print("Resampling to daily closes...")
    # For FX, daily close is typically 5 PM ET, but we can just use the last available bar of the day.
    # Since we are in UTC, we can just resample to 'D'.
    daily = df.resample("D").last().dropna(how="all")
    
    output_path = DATA_DIR / "closes_daily.csv"
    print(f"Saving {len(daily)} days of data to {output_path}...")
    daily.to_csv(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
