import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from coretime_market import run_auction, adjust_reserve_price

# Title
st.title('Interactive Coretime Market Simulator')

# Sidebar configuration
st.sidebar.header('Market Parameters')
supply = st.sidebar.number_input('Cores available', value=10, min_value=1, key='supply')
premium_pct = st.sidebar.slider('Premium (%)', 100.0, 500.0, 200.0)
premium = premium_pct / 100.0
desired = st.sidebar.slider('Target Utilization', 0.0, 1.0, 0.9, key='desired')
k = st.sidebar.slider('Sensitivity k', 0.0, 5.0, 2.0, key='k')
players = int(st.sidebar.number_input('Number of bidders', 1, 50, 3, key='players'))

# Initialize session state on first run
if 'round' not in st.session_state:
    # Start at round 0 with initial reserve
    st.session_state.round = 0
    st.session_state.reserve_price = 1000.0
    st.session_state.initial_reserve = 1000.0
    # Seed history with initial point at round 0
    st.session_state.history = pd.DataFrame([{  
        'round': 0,
        'reserve_price': st.session_state.initial_reserve,
        'p_clear': 0.0,
        'capacity': 0.0,
        'sold': 0,
        'unsold': supply
    }])

# Collect bids input
st.write('Enter bids (quantity & price) for each player:')
bid_data = []
cols = st.columns(min(players, 3))
for i in range(players):
    with cols[i % len(cols)]:
        q = st.number_input(f'P{i+1} quantity', min_value=0, value=0, key=f'q{i}')
        p = st.number_input(f'P{i+1} price', min_value=0.0, step=0.01, key=f'p{i}')
    bid_data.append((q, p))

if st.button('Submit Bids'):
    # Run auction for this round using current reserve
    quantities = np.array([b[0] for b in bid_data])
    prices = np.array([b[1] for b in bid_data])
    player_ids = [f'P{i+1}' for i in range(players)]
    out = run_auction(
        (quantities, prices),
        player_ids,
        supply,
        st.session_state.reserve_price,
        premium
    )

    # Compute next reserve
    new_rp = adjust_reserve_price(
        st.session_state.reserve_price,
        out['capacity'],
        desired,
        k
    )

    # Advance to next round
    st.session_state.round += 1

    # Create a new DataFrame for the new row
    new_row = pd.DataFrame([{
        'round':         st.session_state.round,
        'reserve_price': new_rp,
        'p_clear':       out['p_clear'],
        'capacity':      out['capacity'],
        'sold':          out['sold'],
        'unsold':        out['unsold']
    }])

    # Append to history using concat
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

    # Update reserve price AFTER appending
    st.session_state.reserve_price = new_rp


# Now display current round, reserve, last clearing, max‐bid, and accumulated revenue
st.subheader(f'Round {st.session_state.round}')
st.write(f"Current reserve price: {st.session_state.reserve_price:.3f}")

# maximum bid = reserve * (1 + premium_frac)
current_max = st.session_state.reserve_price * (premium)
st.write(f"Current maximum price: {current_max:.3f}")

last_clear = st.session_state.history['p_clear'].iloc[-1]
st.write(f"Last clearing price: {last_clear:.3f}")

# accumulated revenue = sum over past (p_clear * sold)
acc_revenue = (st.session_state.history['p_clear'] * 
               st.session_state.history['sold']).sum()
st.write(f"Accumulated revenue: {acc_revenue:.3f}")


# Plot after first submission (history length > 1)
if st.session_state.history.shape[0] > 1:
    df = st.session_state.history.copy()
    df['round'] = df['round'].astype(int)
    # Create three subplots: clearing price, reserve price, capacity
    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    max_round = df['round'].max()

    # Clearing price plot
    upper_clear = max(df['p_clear'].max(), st.session_state.initial_reserve * 20)
    ax[0].plot(df['round'], df['p_clear'], marker='o', color='green')
    ax[0].set_title('Clearing Price Over Time')
    ax[0].set_xlabel('Round')
    ax[0].set_ylabel('Clearing Price')
    ax[0].set_ylim(0, upper_clear)
    ax[0].set_xlim(0, max_round)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid(True)

    # Reserve price plot
    upper_reserve = max(df['reserve_price'].max(), st.session_state.initial_reserve * 20)
    ax[1].plot(df['round'], df['reserve_price'], marker='o', color='blue')
    ax[1].set_title('Reserve Price Over Time')
    ax[1].set_xlabel('Round')
    ax[1].set_ylabel('Reserve Price')
    ax[1].set_ylim(0, upper_reserve)
    ax[1].set_xlim(0, max_round)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid(True)

    # Capacity usage plot
    ax[2].plot(df['round'], df['capacity'], marker='o', color='orange')
    ax[2].set_title('Capacity Usage Over Time')
    ax[2].set_xlabel('Round')
    ax[2].set_ylabel('Capacity (sold/supply)')
    ax[2].set_ylim(0, 1)
    ax[2].set_xlim(0, max_round)
    ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[2].grid(True)

    plt.tight_layout()
    st.pyplot(fig)
