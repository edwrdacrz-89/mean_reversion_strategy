"""
Mean Reversion Signal Generator
================================
Generates daily buy signals for a mean reversion strategy on S&P 500 stocks.

STRATEGY OVERVIEW:
==================
This strategy exploits the tendency of stocks to "bounce back" after down days.
We identify stocks with strong historical mean reversion characteristics and
buy them when they drop, expecting a recovery.

EXECUTION RULES (GUARDRAILS):
=============================
1. TIMING:
   - Signals generated at 3:45 PM ET (15 minutes before market close)
   - All orders are Market-On-Close (MOC) orders
   - GitHub Actions runs at 20:45 UTC (3:45 PM ET)

2. DAILY WORKFLOW:
   - Step 1: SELL all existing positions (MOC)
   - Step 2: Calculate total portfolio value from sales
   - Step 3: BUY new signals with equal weight allocation (MOC)
   - Positions are held for exactly ONE day (close-to-close)

3. POSITION SIZING:
   - Maximum 3 positions at any time
   - Equal weight: 33.33% per position
   - If fewer than 3 signals, remainder stays in cash
   - Example: 2 signals = 66.67% invested, 33.33% cash

4. SIGNAL CRITERIA:
   - Stock must be in the approved universe (S&P 500 subset)
   - Consistency (C) >= 55% (stock bounces after 55%+ of down days)
   - Must have dropped from previous close to current price
   - Minimum drop threshold: 0.20%

5. PRICE CALCULATION:
   - Drop % = (Previous Close - Current Price) / Previous Close
   - Uses previous day's close as baseline (not today's open)
   - This allows MOC orders to be placed at 3:45 PM

6. RANKING:
   - Calculate E*C score for each stock (elasticity * consistency)
   - Take top 6 candidates by score
   - Filter by minimum drop requirement
   - Select top 3 by drop percentage (deepest drops first)

TRACK RECORD:
=============
- All signals are logged to track_record.json
- Performance calculated as close-to-close returns
- DO NOT manually edit track_record.json
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path

# =============================================================================
# STRATEGY PARAMETERS (DO NOT MODIFY WITHOUT BACKTESTING)
# =============================================================================

# Lookback period for calculating E and C scores
# 378 trading days ≈ 1.5 years of data
ROLLING_WINDOW = 378

# Minimum consistency threshold
# Stock must bounce on at least 55% of down days historically
MIN_CONSISTENCY = 0.55

# Number of top-scoring candidates to consider each day
TOP_CANDIDATES = 6

# Maximum simultaneous positions
# Portfolio divided equally: 100% / MAX_POSITIONS per position
MAX_POSITIONS = 3

# Minimum price drop required to trigger a buy signal
# 0.20% = stock must have dropped at least 0.20% from previous close
MIN_DROP_PCT = 0.0020

# =============================================================================
# EXECUTION PARAMETERS
# =============================================================================

# Signal generation time (Eastern Time)
SIGNAL_TIME_ET = "15:45"  # 3:45 PM ET

# Order type for all trades
ORDER_TYPE = "MOC"  # Market-On-Close

# Holding period
HOLDING_PERIOD_DAYS = 1  # Close-to-close, sell next day

# Universe - hardcoded for cloud (no local files)
# These are S&P 500 stocks in our target industries
UNIVERSE = [
    # Semiconductors
    'NVDA', 'AMD', 'AVGO', 'INTC', 'QCOM', 'TXN', 'ADI', 'MCHP', 'NXPI', 'KLAC', 'LRCX', 'AMAT',
    # Software
    'MSFT', 'ORCL', 'CRM', 'ADBE', 'NOW', 'INTU', 'SNPS', 'CDNS', 'PANW', 'FTNT', 'CRWD',
    # IT Services
    'ACN', 'IBM', 'CTSH', 'IT', 'LDOS', 'GD',
    # Banks & Financial
    'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'USB', 'PNC', 'TFC',
    # Asset Management
    'BLK', 'SCHW', 'BX', 'KKR', 'APO', 'TROW', 'IVZ', 'BEN',
    # Credit Services
    'V', 'MA', 'AXP', 'COF', 'DFS', 'SYF',
    # Insurance P&C
    'PGR', 'CB', 'TRV', 'ALL', 'HIG', 'CNA',
    # Aerospace & Defense
    'LMT', 'RTX', 'BA', 'NOC', 'GD', 'LHX', 'TDG',
    # Industrial Machinery
    'CAT', 'DE', 'EMR', 'ROK', 'ETN', 'PH', 'ITW', 'IR', 'AME', 'GNRC',
    # Building/Construction
    'SHW', 'LEN', 'DHI', 'PHM', 'NVR', 'TOL', 'KBH',
    # Diagnostics/Research
    'TMO', 'DHR', 'A', 'ILMN', 'MTD', 'PKI', 'WAT', 'BIO',
]

# Current S&P 500 members (subset for filtering)
SP500_MEMBERS = set(UNIVERSE)  # Simplified - assume all are S&P 500

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_prices(tickers):
    """Get historical prices."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=ROLLING_WINDOW + 50)

    try:
        data = yf.download(
            tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )
        return data
    except Exception as e:
        print(f"Error downloading: {e}")
        return None

def get_close_prices(prices_df):
    """Extract close prices."""
    if prices_df is None or prices_df.empty:
        return None

    if isinstance(prices_df.columns, pd.MultiIndex):
        if 'Close' in prices_df.columns.get_level_values(0):
            return prices_df['Close']
        elif 'Close' in prices_df.columns.get_level_values(1):
            return prices_df.xs('Close', axis=1, level=1)
    return prices_df

def calculate_scores(close_prices):
    """Calculate E, C, and Final Score."""
    if close_prices is None or close_prices.empty:
        return pd.DataFrame()

    returns = close_prices.pct_change()
    scores = []

    for ticker in returns.columns:
        ret = returns[ticker].dropna().tail(ROLLING_WINDOW)
        if len(ret) < ROLLING_WINDOW // 2:
            continue

        down_days = ret < 0
        next_day = ret.shift(-1)
        e = next_day.where(down_days, 0).sum()

        bounces = (next_day > 0) & down_days
        down_count = down_days.sum()
        c = bounces.sum() / down_count if down_count > 0 else 0

        if pd.notna(e) and pd.notna(c):
            scores.append({
                'ticker': ticker,
                'E': e,
                'C': c,
                'score': e * c
            })

    return pd.DataFrame(scores)

def get_intraday_info(tickers):
    """Get today's prices and compare to previous close."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                current = hist['Close'].iloc[-1]
                drop = (prev_close - current) / prev_close if prev_close > 0 else 0
                data[ticker] = {
                    'prev_close': float(prev_close),
                    'current': float(current),
                    'drop_pct': float(drop * 100)
                }
        except:
            pass
    return data

def generate_signals(scores_df, intraday_data):
    """Generate trading signals."""
    if scores_df.empty:
        return []

    # Filter by consistency
    scores_df = scores_df[scores_df['C'] >= MIN_CONSISTENCY]
    if scores_df.empty:
        return []

    # Top candidates
    top = scores_df.nlargest(TOP_CANDIDATES, 'score')

    signals = []
    for _, row in top.iterrows():
        ticker = row['ticker']
        if ticker in intraday_data:
            info = intraday_data[ticker]
            if info['drop_pct'] >= MIN_DROP_PCT * 100:
                signals.append({
                    'ticker': ticker,
                    'price': info['current'],
                    'drop_pct': info['drop_pct'],
                    'score': float(row['score']),
                    'consistency': float(row['C'])
                })

    # Sort by drop and take top 3
    signals = sorted(signals, key=lambda x: x['drop_pct'], reverse=True)[:MAX_POSITIONS]
    return signals

def load_track_record():
    """Load existing track record."""
    path = Path('track_record.json')
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {'entries': [], 'start_date': None, 'performance': {'total_trades': 0, 'wins': 0, 'total_return': 0.0}}


def update_previous_returns(track_record):
    """Calculate actual returns for previous day's signals."""
    if not track_record['entries']:
        return track_record

    # Initialize performance stats if missing
    if 'performance' not in track_record:
        track_record['performance'] = {'total_trades': 0, 'wins': 0, 'total_return': 0.0}

    # Find entries without calculated returns
    for entry in track_record['entries']:
        if 'returns' in entry:  # Already calculated
            continue
        if not entry.get('signals'):  # No signals that day
            entry['returns'] = []
            entry['daily_return'] = 0.0
            continue

        entry_date = entry['date']

        # Get prices for signal tickers
        tickers = [s['ticker'] for s in entry['signals']]
        try:
            # Download a few days around the signal date
            start = (datetime.strptime(entry_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
            end = (datetime.strptime(entry_date, '%Y-%m-%d') + timedelta(days=5)).strftime('%Y-%m-%d')

            prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            if prices.empty:
                continue

            close = prices['Close'] if len(tickers) > 1 else prices['Close'].to_frame(tickers[0])

            # Find signal day and next trading day
            if entry_date not in close.index.strftime('%Y-%m-%d').tolist():
                continue

            signal_idx = close.index.strftime('%Y-%m-%d').tolist().index(entry_date)
            if signal_idx >= len(close) - 1:  # Next day not available yet
                continue

            signal_close = close.iloc[signal_idx]
            next_close = close.iloc[signal_idx + 1]

            # Calculate returns for each position
            returns = []
            for signal in entry['signals']:
                ticker = signal['ticker']
                if ticker in signal_close.index and ticker in next_close.index:
                    entry_price = signal_close[ticker]
                    exit_price = next_close[ticker]
                    pct_return = ((exit_price - entry_price) / entry_price) * 100
                    returns.append({
                        'ticker': ticker,
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                        'return_pct': float(pct_return)
                    })

                    # Update performance stats
                    track_record['performance']['total_trades'] += 1
                    track_record['performance']['total_return'] += pct_return / len(entry['signals'])
                    if pct_return > 0:
                        track_record['performance']['wins'] += 1

            entry['returns'] = returns
            entry['daily_return'] = sum(r['return_pct'] for r in returns) / len(returns) if returns else 0.0

        except Exception as e:
            print(f"Error calculating returns for {entry_date}: {e}")
            continue

    # Calculate win rate
    perf = track_record['performance']
    perf['win_rate'] = (perf['wins'] / perf['total_trades'] * 100) if perf['total_trades'] > 0 else 0.0

    return track_record

def save_signals(signals, track_record):
    """Save signals to files."""
    today = datetime.now().strftime('%Y-%m-%d')

    # Save current signals
    current = {
        'date': today,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'signals': signals,
        'allocation_per_position': 100 / MAX_POSITIONS,
        'cash_allocation': (MAX_POSITIONS - len(signals)) * (100 / MAX_POSITIONS)
    }

    with open('signals.json', 'w') as f:
        json.dump(current, f, indent=2)

    # Update track record
    existing_dates = [e['date'] for e in track_record['entries']]
    if today not in existing_dates:
        if track_record['start_date'] is None:
            track_record['start_date'] = today

        track_record['entries'].append({
            'date': today,
            'signals': signals
        })

        with open('track_record.json', 'w') as f:
            json.dump(track_record, f, indent=2)

    return current

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 50)
    print(f"Signal Generation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Check if trading day
    if datetime.now().weekday() >= 5:
        print("Weekend - skipping")
        return

    print(f"Universe: {len(UNIVERSE)} stocks")

    # Get prices
    print("Downloading prices...")
    prices = get_prices(UNIVERSE)
    close = get_close_prices(prices)

    # Calculate scores
    print("Calculating scores...")
    scores = calculate_scores(close)
    print(f"Scores for {len(scores)} stocks")

    # Get intraday for top candidates
    top_tickers = scores[scores['C'] >= MIN_CONSISTENCY].nlargest(TOP_CANDIDATES * 2, 'score')['ticker'].tolist()
    print(f"Getting intraday for {len(top_tickers)} candidates...")
    intraday = get_intraday_info(top_tickers)

    # Generate signals
    signals = generate_signals(scores, intraday)

    if signals:
        print(f"\nSIGNALS ({len(signals)}):")
        for s in signals:
            print(f"  BUY {s['ticker']} @ ${s['price']:.2f} (drop: {s['drop_pct']:.2f}%)")
    else:
        print("\nNO SIGNALS - Hold Cash")

    # Load and update track record
    track_record = load_track_record()

    # Calculate returns for previous signals (now that next-day close is available)
    print("Updating previous returns...")
    track_record = update_previous_returns(track_record)

    # Save new signals
    current = save_signals(signals, track_record)

    # Print performance summary
    perf = track_record.get('performance', {})
    print(f"\nSaved to signals.json")
    print(f"Track record: {len(track_record['entries'])} entries")
    if perf.get('total_trades', 0) > 0:
        print(f"\n--- LIVE PERFORMANCE ---")
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']:.1f}%")
        print(f"Cumulative Return: {perf['total_return']:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
