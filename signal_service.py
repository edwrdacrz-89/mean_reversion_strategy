"""
Mean Reversion Strategy - Automatic Signal Service
===================================================
Runs automatically at 3:45 PM ET every trading day.
Records signals to track_record.json for the dashboard to display.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import schedule
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

ROLLING_WINDOW = 378
MIN_CONSISTENCY = 0.55
TOP_CANDIDATES = 6
MAX_POSITIONS = 3
MIN_DROP_PCT = 0.0020

TRACK_RECORD_FILE = Path('track_record.json')
UNIVERSE_FILE = Path('universe_cache.csv')
SP500_DATA_DIR = Path('sp500_data')

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def load_universe():
    """Load trading universe."""
    if UNIVERSE_FILE.exists():
        return pd.read_csv(UNIVERSE_FILE)['ticker'].tolist()
    logger.error("Universe file not found!")
    return []

def load_sp500_members():
    """Load current S&P 500 members."""
    csv_files = list(SP500_DATA_DIR.glob('S&P 500 Historical Components*.csv'))
    if not csv_files:
        logger.error("S&P 500 data not found!")
        return set()
    df = pd.read_csv(sorted(csv_files)[-1])
    latest_tickers = df.iloc[-1]['tickers']
    return set(latest_tickers.split(','))

def get_prices(tickers: list):
    """Get historical prices for score calculation."""
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
        logger.error(f"Error downloading prices: {e}")
        return None

def get_close_prices(prices_df):
    """Extract close prices from yfinance data."""
    if prices_df is None or prices_df.empty:
        return None

    if isinstance(prices_df.columns, pd.MultiIndex):
        if 'Close' in prices_df.columns.get_level_values(0):
            return prices_df['Close']
        elif 'Close' in prices_df.columns.get_level_values(1):
            return prices_df.xs('Close', axis=1, level=1)
    return prices_df

def calculate_scores(close_prices):
    """Calculate E, C, and Final Score for all stocks."""
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

def get_intraday_info(tickers: list):
    """Get today's open/current price for each ticker."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if not hist.empty:
                open_p = hist['Open'].iloc[0]
                close_p = hist['Close'].iloc[-1]
                drop = (open_p - close_p) / open_p if open_p > 0 else 0
                data[ticker] = {
                    'open': open_p,
                    'current': close_p,
                    'drop_pct': drop * 100,
                    'name': stock.info.get('shortName', ticker)
                }
        except Exception as e:
            logger.warning(f"Could not get intraday for {ticker}: {e}")
    return data

def generate_signals(scores_df, intraday_data, sp500_members):
    """Generate trading signals."""
    if scores_df.empty:
        return []

    scores_df = scores_df[scores_df['ticker'].isin(sp500_members)]
    scores_df = scores_df[scores_df['C'] >= MIN_CONSISTENCY]

    if scores_df.empty:
        return []

    top = scores_df.nlargest(TOP_CANDIDATES, 'score')

    signals = []
    for _, row in top.iterrows():
        ticker = row['ticker']
        if ticker in intraday_data:
            info = intraday_data[ticker]
            if info['drop_pct'] >= MIN_DROP_PCT * 100:
                signals.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'price': info['current'],
                    'drop_pct': info['drop_pct'],
                    'score': row['score'],
                    'consistency': row['C']
                })

    signals = sorted(signals, key=lambda x: x['drop_pct'], reverse=True)[:MAX_POSITIONS]
    return signals

# =============================================================================
# TRACK RECORD
# =============================================================================

def load_track_record():
    """Load track record from file."""
    if TRACK_RECORD_FILE.exists():
        with open(TRACK_RECORD_FILE, 'r') as f:
            return json.load(f)
    return {'entries': [], 'start_date': None}

def save_track_record(record):
    """Save track record to file."""
    with open(TRACK_RECORD_FILE, 'w') as f:
        json.dump(record, f, indent=2)

def record_signals(signals):
    """Record signals to track record."""
    record = load_track_record()
    today = datetime.now().strftime('%Y-%m-%d')

    existing_dates = [e['date'] for e in record['entries']]
    if today in existing_dates:
        logger.info(f"Already recorded for {today}, skipping")
        return record

    if record['start_date'] is None:
        record['start_date'] = today

    record['entries'].append({
        'date': today,
        'signals': [{
            'ticker': s['ticker'],
            'price': s['price'],
            'drop_pct': s['drop_pct']
        } for s in signals],
        'recorded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    save_track_record(record)
    return record

# =============================================================================
# MAIN JOB
# =============================================================================

def is_trading_day():
    """Check if today is a trading day (weekday, not holiday)."""
    today = datetime.now()
    # Weekend check
    if today.weekday() >= 5:
        return False
    # Could add holiday check here
    return True

def run_signal_generation():
    """Main job - generate and record signals."""
    logger.info("=" * 50)
    logger.info("Starting signal generation...")

    if not is_trading_day():
        logger.info("Not a trading day, skipping")
        return

    # Load data
    universe = load_universe()
    if not universe:
        logger.error("No universe loaded")
        return

    sp500 = load_sp500_members()
    if not sp500:
        logger.error("No S&P 500 data")
        return

    logger.info(f"Universe: {len(universe)} stocks")
    logger.info(f"S&P 500: {len(sp500)} members")

    # Get prices and calculate scores
    logger.info("Downloading price data...")
    prices = get_prices(universe)
    close = get_close_prices(prices)
    scores = calculate_scores(close)

    if scores.empty:
        logger.error("No scores calculated")
        return

    logger.info(f"Calculated scores for {len(scores)} stocks")

    # Get intraday data for top candidates
    top_tickers = scores[scores['C'] >= MIN_CONSISTENCY].nlargest(TOP_CANDIDATES * 2, 'score')['ticker'].tolist()
    logger.info(f"Getting intraday data for {len(top_tickers)} candidates...")
    intraday = get_intraday_info(top_tickers)

    # Generate signals
    signals = generate_signals(scores, intraday, sp500)

    if signals:
        logger.info(f"Generated {len(signals)} signals:")
        for s in signals:
            logger.info(f"  BUY {s['ticker']} @ ${s['price']:.2f} (drop: {s['drop_pct']:.2f}%)")
    else:
        logger.info("No signals today - CASH")

    # Record
    record = record_signals(signals)
    logger.info(f"Track record now has {len(record['entries'])} entries")
    logger.info("Signal generation complete!")
    logger.info("=" * 50)

# =============================================================================
# SCHEDULER
# =============================================================================

def main():
    logger.info("Mean Reversion Strategy Signal Service Starting...")
    logger.info(f"Will run at 3:45 PM ET every trading day")

    # Schedule the job at 3:45 PM (adjust for your timezone if needed)
    schedule.every().monday.at("15:45").do(run_signal_generation)
    schedule.every().tuesday.at("15:45").do(run_signal_generation)
    schedule.every().wednesday.at("15:45").do(run_signal_generation)
    schedule.every().thursday.at("15:45").do(run_signal_generation)
    schedule.every().friday.at("15:45").do(run_signal_generation)

    # Also run once at startup if during market hours
    now = datetime.now()
    if now.weekday() < 5 and now.hour >= 15 and now.hour < 16:
        logger.info("Running immediately (within signal window)...")
        run_signal_generation()

    logger.info("Service running. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
