import numpy as np
import pandas as pd

# 1) parse_bids: adds player_id and random timing for tie-breaking
def parse_bids(raw_bids, player_ids):
    quantities, prices = raw_bids
    timing = np.random.rand(len(player_ids))
    return pd.DataFrame({
        'player_id': player_ids,
        'quantity':  quantities,
        'price':      prices,
        'timing':     timing
    })

# 2) compute_clearing_price: clamp bids above reserve * premium, discard below reserve
def compute_clearing_price(bids, supply, reserve_price, premium):
    df = bids.copy()

    # Clamp any bids above reserve_price * premium
    max_price = reserve_price * premium
    too_high = df.loc[df['price'] > max_price, 'player_id'].tolist()
    if too_high:
        print(f"Warning: capped bids too-high from {too_high} to {max_price}")
    df['price'] = df['price'].clip(upper=max_price)

    # Keep only bids at or above reserve, with quantity > 0
    valid = df[(df['price'] >= reserve_price) & (df['quantity'] > 0)].copy()
    if valid.empty:
        return 0.0, 0.0, valid

    # Sort by price desc, timing asc for tie-breaking
    valid.sort_values(['price', 'timing'], ascending=[False, True], inplace=True)

    # Compute cumulative quantity to find clearing price
    valid['cum_qty'] = valid['quantity'].cumsum()
    crossing = valid['cum_qty'] >= supply
    if not crossing.any():
        p_clear = valid['price'].min()
        cap = valid['cum_qty'].max() / supply
        return p_clear, cap, valid

    # Identify the first crossing bid
    idx = crossing.idxmax()
    p_clear = valid.at[idx, 'price']
    cap = valid['cum_qty'].iat[-1] / supply
    return p_clear, cap, valid

# 3) allocate_cores: allocate in sorted order up to supply
def allocate_cores(filtered_bids, supply, p_clear):
    if filtered_bids.empty:
        return pd.DataFrame(columns=['player_id','price','allocated']), 0, supply

    remaining = supply
    allocations = []
    for _, row in filtered_bids.iterrows():
        take = min(row['quantity'], remaining)
        allocations.append({
            'player_id': row['player_id'],
            'price':      row['price'],
            'allocated':  take
        })
        remaining -= take
        if remaining <= 0:
            break

    alloc_df = pd.DataFrame(allocations)
    sold = alloc_df['allocated'].sum()
    unsold = supply - sold
    return alloc_df, sold, unsold

# 4) apply_premium: market price = clearing price * (1 + premium)
def apply_premium(p_clear, premium):
    return p_clear * (1 + premium)

# 5) run_auction: one-shot auction
def run_auction(raw_bids, player_ids, supply, reserve_price, premium):
    bids_df = parse_bids(raw_bids, player_ids)
    p_clear, cap, valid = compute_clearing_price(bids_df, supply, reserve_price, premium)
    alloc_df, sold, unsold = allocate_cores(valid, supply, p_clear)
    p_market = apply_premium(p_clear, premium)
    return {
        'p_clear':     p_clear,
        'p_market':    p_market,
        'capacity':    cap,
        'sold':        sold,
        'unsold':      unsold,
        'allocations': alloc_df
    }

# 6) adjust_reserve_price: exponential-of-error update, floored at 1
def adjust_reserve_price(p_old, capacity, desired, k, p_min=1):
    p_new = p_old * np.exp(k * (capacity - desired))
    if capacity >= 1.0:
        p_new = max(p_new, p_old + 100)
    return max(p_new, p_min)
