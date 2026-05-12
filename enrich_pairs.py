import pandas as pd
import statsmodels.api as sm
import numpy as np
from config import DATA_DIR

# Load top 50 pairs
top50 = pd.read_csv(DATA_DIR / "pairs_top50_wfo.csv")
closes = pd.read_csv(DATA_DIR / "closes_daily.csv", index_col=0, parse_dates=True)

enriched = []
for idx, row in top50.iterrows():
    pair = row['pair']
    t1, t2 = pair.split('-')
    
    if t1 in closes.columns and t2 in closes.columns:
        # Get beta using last 252 days
        y = closes[t1].tail(252)
        x = closes[t2].tail(252)
        
        # Align and drop NaNs
        aligned = pd.concat([y, x], axis=1).dropna()
        if len(aligned) > 50:
            Y = aligned.iloc[:, 0]
            X = aligned.iloc[:, 1]
            X = sm.add_constant(X)
            
            # Check for infs
            if not np.any(np.isinf(X)) and not np.any(np.isinf(Y)):
                model = sm.OLS(Y, X).fit()
                beta = model.params.iloc[1]
                
                enriched.append({
                    "pair": pair,
                    "beta": beta
                })

df_enriched = pd.DataFrame(enriched)
df_enriched.to_csv(DATA_DIR / "pairs_top50_enriched.csv", index=False)
print(f"Enriched {len(df_enriched)} pairs with fresh beta.")
