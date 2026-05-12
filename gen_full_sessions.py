import pandas as pd
from itertools import combinations
from config import DATA_DIR

# Load tickers
daily = pd.read_csv(DATA_DIR / "closes_daily.csv", index_col=0, nrows=1)
tickers = sorted(daily.columns.tolist())
combos = list(combinations(tickers, 2))
all_pairs = [f"{t1}-{t2}" for t1, t2 in combos]

# Refined Logic from update_sessions.py
euro_currencies = ['eur', 'gbp', 'chf', 'czk', 'pln', 'huf', 'nok', 'sek', 'dkk']
asia_currencies = ['jpy', 'aud', 'nzd', 'sgd', 'hkd']
na_currencies   = ['usd', 'cad', 'mxn']

def get_session(pair):
    t1, t2 = pair.lower().split('-')
    is_e1 = any(c in t1 for c in euro_currencies); is_e2 = any(c in t2 for c in euro_currencies)
    is_a1 = any(c in t1 for c in asia_currencies); is_a2 = any(c in t2 for c in asia_currencies)
    is_n1 = any(c in t1 for c in na_currencies);   is_n2 = any(c in t2 for c in na_currencies)
    
    is_euro = is_e1 or is_e2
    is_asia = is_a1 or is_a2
    is_na   = is_n1 or is_n2
    
    if is_euro and is_asia: return 5, 18
    if is_euro and is_na:   return 8, 18
    if is_asia and is_na:   return 5, 16
    if is_euro:            return 8, 18
    if is_asia:            return 5, 16
    return 0, 24

data = []
for p in all_pairs:
    s, e = get_session(p)
    data.append({"pair": p, "best_start_utc": s, "best_end_utc": e})

df = pd.DataFrame(data)
df.to_csv(DATA_DIR / "pair_sessions.csv", index=False)
print(f"Generated sessions for {len(df)} pairs.")
