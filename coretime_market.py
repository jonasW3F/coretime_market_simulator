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

# 2) compute_clearing_price: clamp bids above reserve* premium, discard below reserve
def compute_clearing_price(bids, supply, reserve_price, premium):
    # Copy so we don't mutate caller
    df = bids.copy()

    # 1) Clamp any bids above reserve_price * premium
    max_acc = reserve_price *  premium
    too_high = df.loc[df['price'] > max_acc, 'player_id'].tolist()
    if too_high:
        print(f"Warning: capped bids too-high from {too_high} to {max_acc}")
    df['price'] = df['price'].clip(upper=max_acc)

    # 2) Keep only bids at or above reserve, with quantity > 0
    valid = df[(df['price'] >= reserve_price) & (df['quantity'] > 0)].copy()
    if valid.empty:
        # No valid demand at all → clearing price is zero
        return 0.0, 0.0, valid

    # 3) Sort by price desc, timing asc and accumulate quantities
    valid = valid.sort_values(['price','timing'], ascending=[False,True])
    valid['cum_qty'] = valid['quantity'].cumsum()

    # 4) Find the index where cumulative qty ≥ supply
    idx = valid['cum_qty'].searchsorted(supply, side='left')

    if idx >= len(valid) or valid['cum_qty'].iloc[-1] < supply:
        # Demand < supply → clear at reserve price
        p_clear = reserve_price
        sold    = int(valid['quantity'].sum())
    else:
        # Clearing bid sets the price
        p_clear = float(valid.iloc[idx]['price'])
        sold    = int(min(supply, valid['cum_qty'].iloc[idx]))

    capacity = sold / supply
    return p_clear, capacity, valid



# 3) allocate_cores: allocate in sorted order up to supply
def allocate_cores(filtered_bids, supply, p_clear):
    if filtered_bids.empty:
        return pd.DataFrame(columns=['player_id','price','allocated']), 0, supply
    remaining = supply
    allocs = []
    for _, row in filtered_bids.iterrows():
        take = min(row['quantity'], remaining)
        allocs.append(take)
        remaining -= take
        if remaining <= 0:
            break
    df = filtered_bids.iloc[:len(allocs)][['player_id']].copy()
    df['price']     = p_clear
    df['allocated'] = allocs
    sold   = int(sum(allocs))
    return df, sold, supply - sold

# 4) apply_premium: market price = clearing price * (1 + premium)
def apply_premium(p_clear, premium):
    return p_clear * (1 + premium)

# 5) run_auction: end-to-end single-period auction
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
def adjust_reserve_price(p_old, capacity, desired, k=1.0, p_min=1):
    p_new = p_old * np.exp(k * (capacity - desired))
    # special rule to allow for recovery after very low prices. We have a minimum increment of e.g., 100 DOT if the system is on full capacity.
    if capacity >= 1.0:
        p_new = max(p_new, p_old + 100)

    return max(p_new, p_min)

# 7) simulate_market: runs multiple rounds, returns DataFrame
def simulate_market(raw_bids_list, player_ids, supply_vec,
                    reserve_init, premium, desired_capacity, k=1.0):
    results = []
    rp = reserve_init
    for t in range(len(raw_bids_list)):
        raw = raw_bids_list[t]
        out = run_auction(raw, player_ids, supply_vec[t], rp, premium)
        results.append({
            'round':         t+1,
            'reserve_price': rp,
            'p_clear':       out['p_clear'],
            'p_market':      out['p_market'],
            'capacity':      out['capacity'],
            'sold':          out['sold'],
            'unsold':        out['unsold']
        })
        rp = adjust_reserve_price(rp, out['capacity'], desired_capacity, k)
    return pd.DataFrame(results)
