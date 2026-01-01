"""
Mean Reversion Strategy Dashboard
==================================
Daily signal generator with live track record and technical documentation.

See cloud_signal_generator.py for full strategy documentation and guardrails.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import json

# =============================================================================
# STRATEGY PARAMETERS (MUST MATCH cloud_signal_generator.py)
# =============================================================================

# Lookback period for calculating E and C scores (378 trading days = 1.5 years)
ROLLING_WINDOW = 378

# Minimum consistency threshold (stock must bounce 55%+ of down days)
MIN_CONSISTENCY = 0.55

# Number of top-scoring candidates to consider
TOP_CANDIDATES = 6

# Maximum positions (equal weight: 33.33% each)
MAX_POSITIONS = 3

# Minimum drop from previous close to trigger signal (0.20%)
MIN_DROP_PCT = 0.0020

# =============================================================================
# EXECUTION RULES (GUARDRAILS)
# =============================================================================

SIGNAL_TIME_ET = "15:45"  # 3:45 PM ET - when to generate signals
ORDER_TYPE = "MOC"  # Market-On-Close orders only
HOLDING_PERIOD = 1  # Days (close-to-close)

# Daily workflow:
# 1. SELL all existing positions (MOC)
# 2. Calculate total portfolio value
# 3. BUY new signals with equal weight (MOC)

TRACK_RECORD_FILE = Path("track_record.json")

# GitHub raw URLs for fetching latest data
GITHUB_REPO = "edwrdacrz-89/mean_reversion_strategy"
GITHUB_TRACK_RECORD_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/master/track_record.json"
GITHUB_SIGNALS_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/master/signals.json"

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Mean Reversion Strategy",
    page_icon="📈",
    layout="wide"
)

# Auto-refresh every 60 seconds for price data (smooth enough to not be jarring)
# Auto-refresh every 60 seconds for price data (smooth enough to not be jarring)
# We use query params to persist the pause state across reloads
query_params = st.query_params
initial_pause_state = query_params.get("paused", "false").lower() == "true"

pause_updates = st.sidebar.toggle("Pause Updates", value=initial_pause_state)

if pause_updates:
    st.query_params["paused"] = "true"
else:
    st.query_params["paused"] = "false"

if not pause_updates:
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=DSEG7+Classic:wght@400&display=swap');
    @font-face {
        font-family: 'Digital-7';
        src: url('https://cdn.jsdelivr.net/npm/digital-7-font@1.0.0/digital-7.ttf') format('truetype');
    }

    /* Base styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit elements - but keep sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Make sidebar toggle more visible */
    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }

    /* Style the header area to be minimal but functional */
    header[data-testid="stHeader"] {
        background: transparent;
        height: 2.5rem;
    }

    /* Full viewport, no scrolling */
    .block-container {
        padding-top: 0.25rem;
        padding-bottom: 0;
        max-width: 1600px;
    }

    /* Allow scrolling but minimize need for it */
    .main .block-container {
        padding-bottom: 0.5rem;
    }

    /* Glassmorphism card base */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
    }

    /* Position cards */
    .position-card {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.08) 0%, rgba(0, 200, 100, 0.03) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 136, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }

    .position-card:hover {
        border-color: rgba(0, 255, 136, 0.3);
        box-shadow: 0 4px 16px rgba(0, 255, 136, 0.1);
    }

    .position-card.negative {
        background: linear-gradient(135deg, rgba(255, 82, 82, 0.08) 0%, rgba(200, 60, 60, 0.03) 100%);
        border-color: rgba(255, 82, 82, 0.15);
    }

    .position-card.negative:hover {
        border-color: rgba(255, 82, 82, 0.3);
        box-shadow: 0 4px 16px rgba(255, 82, 82, 0.1);
    }

    /* Cash box - compact */
    .cash-box {
        background: linear-gradient(135deg, rgba(100, 100, 100, 0.1) 0%, rgba(80, 80, 80, 0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(150, 150, 150, 0.15);
        border-radius: 10px;
        padding: 12px;
        margin: 4px 0;
        text-align: center;
    }

    /* Portfolio summary */
    .portfolio-summary {
        background: linear-gradient(135deg, rgba(30, 40, 70, 0.8) 0%, rgba(20, 30, 50, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(100, 150, 255, 0.1);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 12px;
    }

    /* Metric styling - compact */
    .metric-label {
        font-size: 10px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 600;
        margin: 0;
    }

    .metric-green { color: #00ff88; }
    .metric-red { color: #ff5252; }

    .big-number {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }

    /* Info/Warning boxes */
    .info-box {
        background: rgba(74, 158, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(74, 158, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }

    .warning-box {
        background: rgba(255, 170, 0, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 170, 0, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 20, 35, 0.95) 0%, rgba(10, 15, 25, 0.98) 100%);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Headers */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }

    h2, h3 {
        font-weight: 600 !important;
    }

    /* Ticker symbol styling - compact */
    .ticker-symbol {
        font-size: 16px;
        font-weight: 700;
        color: #fff;
        letter-spacing: 0.5px;
    }

    .ticker-price {
        font-size: 18px;
        font-weight: 600;
    }

    .ticker-change {
        font-size: 14px;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
    }

    .ticker-change.positive {
        background: rgba(0, 255, 136, 0.15);
        color: #00ff88;
    }

    .ticker-change.negative {
        background: rgba(255, 82, 82, 0.15);
        color: #ff5252;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4a9eff 0%, #2d7dd2 100%);
        border: none;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #5aafff 0%, #3d8de2 100%);
        box-shadow: 0 4px 20px rgba(74, 158, 255, 0.3);
        transform: translateY(-1px);
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }

    /* Animation for cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-in {
        animation: fadeInUp 0.4s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def load_universe():
    """Load trading universe - same as cloud_signal_generator.py (125 stocks)."""
    return [
        # Diagnostics & Research
        'A', 'DGX', 'DHR', 'IDXX', 'ILMN', 'IQV', 'LH', 'MTD', 'TMO', 'WAT',
        # IT Services
        'ACN', 'BR', 'CTSH', 'DXC', 'FIS', 'IBM', 'IT', 'JKHY', 'LDOS', 'UIS', 'XRX',
        # Software
        'ADBE', 'ADP', 'ADSK', 'AKAM', 'CDNS', 'CRM', 'FFIV', 'FTNT', 'INTU', 'MSFT',
        'NTAP', 'ORCL', 'PAYX', 'PTC', 'ROP', 'S', 'SNPS', 'TDC', 'VRSN',
        # Semiconductors
        'ADI', 'AMD', 'AMAT', 'AVGO', 'INTC', 'IPGP', 'KLAC', 'LRCX', 'MCHP', 'MU',
        'NVDA', 'QCOM', 'QRVO', 'SWKS', 'TER', 'TXN',
        # Electronic Components
        'APH', 'GLW', 'JBL', 'SANM', 'TEL',
        # Insurance P&C
        'AIZ', 'ALL', 'CB', 'CINF', 'HIG', 'L', 'PGR', 'TRV',
        # Banks
        'BAC', 'BK', 'C', 'JPM', 'WFC',
        # Asset Management
        'AMG', 'AMP', 'BEN', 'BLK', 'IVZ', 'NTRS', 'PFG', 'PX', 'RJF', 'STT', 'TROW',
        # Credit Services
        'AXP', 'COF', 'MA', 'NAVI', 'PYPL', 'SLM', 'SYF', 'V', 'WU',
        # Aerospace & Defense
        'BA', 'GD', 'GE', 'HII', 'LMT', 'NOC', 'TDG', 'TXT',
        # Specialty Industrial Machinery
        'AME', 'AOS', 'CMI', 'CR', 'DOV', 'EMR', 'ETN', 'FLS', 'IR', 'ITT', 'ITW',
        'PH', 'PNR', 'ROK', 'XYL',
        # Building & Construction
        'DHI', 'JCI', 'KBH', 'LEN', 'LPX', 'MAS', 'PHM',
    ]

@st.cache_data(ttl=300)
def load_sp500_members():
    """Load current S&P 500 members."""
    csv_files = list(Path('sp500_data').glob('S&P 500 Historical Components*.csv'))
    if not csv_files:
        return set()
    df = pd.read_csv(sorted(csv_files)[-1])
    latest_tickers = df.iloc[-1]['tickers']
    return set(latest_tickers.split(','))

@st.cache_data(ttl=120)
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
    except:
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

@st.cache_data(ttl=60)
def get_intraday_chart_data(tickers: list):
    """Get intraday price data for charting current positions."""
    if not tickers:
        return {}

    intraday_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get 1-day intraday data with 5-minute intervals
            hist = stock.history(period='1d', interval='5m')
            if not hist.empty:
                intraday_data[ticker] = {
                    'times': hist.index.tolist(),
                    'prices': hist['Close'].tolist(),
                    'open': hist['Open'].iloc[0] if len(hist) > 0 else None
                }
        except:
            pass
    return intraday_data

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

        # Mean reversion: sum of next-day returns after down days
        down_days = ret < 0
        next_day = ret.shift(-1)
        e = next_day.where(down_days, 0).sum()

        # Consistency: % of bounces after down days
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

@st.cache_data(ttl=60)
def get_drops_batch(tickers: list):
    """Get today's drops for all tickers in one batch download (fast)."""
    try:
        data = yf.download(tickers, period='5d', auto_adjust=True, progress=False)
        if data.empty:
            return {}

        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data[['Close']]
            close.columns = [tickers[0]] if len(tickers) == 1 else tickers

        drops = {}
        for ticker in close.columns:
            prices = close[ticker].dropna()
            if len(prices) >= 2:
                prev_close = prices.iloc[-2]
                current = prices.iloc[-1]
                drop = (prev_close - current) / prev_close if prev_close > 0 else 0
                drops[ticker] = {
                    'prev_close': float(prev_close),
                    'current': float(current),
                    'drop_pct': float(drop * 100),
                    'name': ticker  # Name lookup is slow, skip for batch
                }
        return drops
    except:
        return {}

@st.cache_data(ttl=60)
def get_intraday_info(tickers: list):
    """Get current price vs previous close for each ticker (with names)."""
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
                    'prev_close': prev_close,
                    'current': current,
                    'drop_pct': drop * 100,
                    'name': stock.info.get('shortName', ticker)
                }
        except:
            pass
    return data

def generate_todays_signals(scores_df, intraday_data, sp500_members):
    """Generate today's trading signals."""
    if scores_df.empty:
        return []

    # Filter by S&P 500 membership
    scores_df = scores_df[scores_df['ticker'].isin(sp500_members)]

    # Filter by consistency threshold
    scores_df = scores_df[scores_df['C'] >= MIN_CONSISTENCY]

    if scores_df.empty:
        return []

    # Top candidates by score
    top = scores_df.nlargest(TOP_CANDIDATES, 'score')

    # Filter by drop from previous close
    signals = []
    for _, row in top.iterrows():
        ticker = row['ticker']
        if ticker in intraday_data:
            info = intraday_data[ticker]
            if info['drop_pct'] >= MIN_DROP_PCT * 100:
                signals.append({
                    'ticker': ticker,
                    'name': info['name'],
                    'prev_close': info['prev_close'],
                    'current_price': info['current'],
                    'change_pct': -info['drop_pct'],  # Negative for drops
                    'change_dollar': info['current'] - info['prev_close'],
                    'drop_pct': info['drop_pct'],
                    'score': row['score'],
                    'consistency': row['C']
                })

    # Sort by drop (deepest first) and take top 3
    signals = sorted(signals, key=lambda x: x['drop_pct'], reverse=True)[:MAX_POSITIONS]
    return signals

# =============================================================================
# TRACK RECORD
# =============================================================================

def fetch_from_github(url):
    """Fetch JSON data from GitHub raw URL. Returns (data, error_message)."""
    import requests
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"GitHub returned status {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "GitHub request timed out"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to GitHub"
    except Exception as e:
        return None, f"GitHub fetch error: {str(e)}"

def load_track_record():
    """Load track record from GitHub only."""
    data, error = fetch_from_github(GITHUB_TRACK_RECORD_URL)
    if data:
        return data, None
    return {'entries': [], 'start_date': None}, error

def save_track_record(record):
    """Save track record to file."""
    with open(TRACK_RECORD_FILE, 'w') as f:
        json.dump(record, f, indent=2)

def record_todays_signals(signals):
    """Record today's signals to track record."""
    record, _ = load_track_record()  # Ignore error for local recording
    today = datetime.now().strftime('%Y-%m-%d')

    # Don't duplicate
    existing_dates = [e['date'] for e in record['entries']]
    if today in existing_dates:
        return record

    if record['start_date'] is None:
        record['start_date'] = today

    record['entries'].append({
        'date': today,
        'signals': signals,
        'recorded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    save_track_record(record)
    return record

def calculate_track_record_performance(record):
    """Calculate performance from track record."""
    if not record['entries']:
        return None

    # Get historical prices for all tickers we've traded
    all_tickers = set()
    for entry in record['entries']:
        for sig in entry['signals']:
            all_tickers.add(sig['ticker'])

    if not all_tickers:
        return None

    # Download prices
    start = record['start_date']
    prices = yf.download(
        list(all_tickers),
        start=start,
        end=datetime.now().strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=False
    )

    if prices.empty:
        return None

    close = get_close_prices(prices)
    if close is None:
        return None

    returns = close.pct_change()

    # Calculate portfolio returns day by day
    portfolio_returns = []

    for i, entry in enumerate(record['entries']):
        date = entry['date']
        tickers = [s['ticker'] for s in entry['signals']]

        # Get next day's return (since we buy at close, sell next close)
        try:
            date_idx = returns.index.get_loc(pd.Timestamp(date))
            if date_idx + 1 < len(returns):
                next_date = returns.index[date_idx + 1]

                if tickers:
                    day_return = 0
                    weight = 1 / MAX_POSITIONS
                    for ticker in tickers:
                        if ticker in returns.columns:
                            ret = returns.loc[next_date, ticker]
                            if pd.notna(ret):
                                day_return += ret * weight

                    # Cash for empty slots
                    cash_slots = MAX_POSITIONS - len(tickers)
                    # Cash returns 0

                    portfolio_returns.append({
                        'date': str(next_date.date()),
                        'return': day_return,
                        'positions': tickers
                    })
        except:
            pass

    return portfolio_returns

# =============================================================================
# TECHNICAL DETAILS PAGE
# =============================================================================

def show_technical_details():
    """Display the technical details page."""
    st.title("Technical Details")
    st.caption("Complete documentation of the mean reversion strategy")

    st.header("Strategy Overview")
    st.markdown("""
    This strategy capitalizes on short-term price corrections in high-quality, systemically important stocks ("The Power Universe"). It is fundamentally designed to exploit behavioral inefficiencies and psychological overreactions in the market.

    Human market participants often over-correct to negative news, driven by fear and uncertainly. This strategy systematically identifies these moments of peak emotional selling and positions itself to capture the inevitable reversion when rationality returns.
    """)



    st.header("Execution Rules")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Timing")
        st.markdown(f"""
        - Signal Generation: {SIGNAL_TIME_ET} ET (3:45 PM)
        - Order Type: {ORDER_TYPE} (Market-On-Close)
        - Holding Period: {HOLDING_PERIOD} day (close-to-close)
        """)

        st.subheader("2. Daily Workflow")
        st.markdown("""
        1. SELL all existing positions (MOC)
        2. Calculate total portfolio value from sales
        3. BUY new signals with equal weight (MOC)

        *Positions are held for exactly ONE day (close-to-close)*
        """)

    with col2:
        st.subheader("3. Position Sizing")
        st.markdown(f"""
        - Max Positions: {MAX_POSITIONS}
        - Weight per Position: {100/MAX_POSITIONS:.2f}%
        - If fewer signals: Remainder stays in cash

        *Example: 2 signals = 66.67% invested, 33.33% cash*
        """)

        st.subheader("4. Signal Criteria")
        st.markdown(f"""
        - Universe: S&P 500 subset (Power Universe)
        - Min Consistency (C): {MIN_CONSISTENCY*100:.0f}%
        - Min Drop: {MIN_DROP_PCT*100:.2f}% from previous close
        """)

    st.header("Scoring System")

    st.subheader("E Score (Elasticity)")
    st.markdown(f"""
    What generally happens when this stock drops?
    This score measures the *strength* of a stock's historical recovery. A high Elasticity Score means that when the stock has dipped in the past, it has typically rebounded strongly the next day.
    ```
    E = Sum of next-day returns after all down days (over {ROLLING_WINDOW} trading days)
    ```
    - Higher E = Stock tends to bounce more aggressively after drops
    - Calculated over rolling {ROLLING_WINDOW} trading days (~1.5 years)
    """)

    st.subheader("C Score (Consistency)")
    st.markdown(f"""
    How often does this stock recover?
    This measures the *probability* of a rebound. We require a minimum 55% historical win rate. This ensures we are not just chasing volatile stocks, but ones that reliably revert to the mean.
    ```
    C = (Number of bounces after down days) / (Total down days)
    ```
    - C of 0.55 means stock goes up 55% of the time after a down day
    - Minimum threshold: {MIN_CONSISTENCY*100:.0f}%
    """)

    st.subheader("Final Score")
    st.markdown("""
    ```
    Final Score = E x C
    ```
    - Combines magnitude (E) with reliability (C)
    - Higher score = Better mean reversion candidate
    """)

    st.header("Price Drop Calculation")
    st.markdown("""
    IMPORTANT: Drop is calculated from previous day's close, NOT today's open.

    `Drop % = (Previous Close - Current Price) / Previous Close x 100`

    This allows MOC orders to be placed at 3:45 PM with known baseline.
    """, unsafe_allow_html=True)

    st.header("Signal Selection Process")
    st.markdown(f"""
    1. Calculate E and C scores for all stocks in universe
    2. Filter: Keep only stocks with C >= {MIN_CONSISTENCY*100:.0f}%
    3. Rank by Final Score (E x C)
    4. Take top {TOP_CANDIDATES} candidates
    5. Filter: Keep only stocks that dropped >= {MIN_DROP_PCT*100:.2f}% today
    6. Rank remaining by drop percentage (deepest first)
    7. Select top {MAX_POSITIONS} as final signals
    """)

    st.header("Strategy Parameters")
    params_df = pd.DataFrame({
        'Parameter': ['ROLLING_WINDOW', 'MIN_CONSISTENCY', 'TOP_CANDIDATES', 'MAX_POSITIONS', 'MIN_DROP_PCT'],
        'Value': [ROLLING_WINDOW, MIN_CONSISTENCY, TOP_CANDIDATES, MAX_POSITIONS, MIN_DROP_PCT],
        'Description': [
            'Days of history for E/C calculation',
            'Minimum bounce rate after down days',
            'Number of top scorers to consider',
            'Maximum simultaneous positions',
            'Minimum drop to trigger signal'
        ]
    })
    st.table(params_df)

    st.header("Universe Selection")
    st.markdown("""
    We focus exclusively on industries with deep structural ties to government spending and critical infrastructure. These companies are less likely to fail within the larger socio-economic-political system they exist in because of inextricable links to national stability. This ensures they benefit from consistent institutional capital flows and are effectively insulated from total collapse.

    Key Sectors:
    - Defense & Aerospace (Government contracts)
    - Financials (Systemic infrastructure)
    - Semiconductors (Critical supply chain)
    - Infrastructure & Construction

    Pre-Selected Industries:
    - Semiconductors - Critical infrastructure, government contracts
    - Software - Enterprise IT, cybersecurity
    - IT Services - Government/enterprise contracts
    - Banks & Financial - Financial system infrastructure
    - Asset Management - Institutional capital flows
    - Credit Services - Payment infrastructure
    - Insurance P&C - Regulated, stable demand
    - Aerospace & Defense - Government contracts
    - Industrial Machinery - Infrastructure, government spending
    - Building/Construction - Housing, infrastructure
    - Diagnostics/Research - Healthcare infrastructure
    """)

    st.header("Important Warnings")
    st.markdown("""
    DO NOT:
    - Modify strategy parameters without backtesting
    - Record signals at times other than 3:45 PM ET
    - Manually edit track_record.json
    - Execute orders other than MOC
    - Hold positions longer than 1 day
    """, unsafe_allow_html=True)

    st.header("Historical Performance (2015-2025)")
    st.markdown("""
    Backtest conducted from Jan 1, 2015 to Nov 28, 2025 on historical S&P 500 constituents.
    Results exclude commissions but include 0.05% slippage per trade.
    """)
    
    perf_data = pd.DataFrame({
        'Metric': ['Total Return', 'CAGR', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown', 'Avg Trades/Day'],
        'Value': ['4,048%', '40.8%', '1.94', '57.6%', '-24.9%', '1.86']
    })
    st.table(perf_data)

# =============================================================================
# SIGNALS PAGE
# =============================================================================

def show_signals_page():
    """Display the main signals page."""
    # Get Eastern time server-side
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo('America/New_York')
    now_et = datetime.now(eastern)

    # Check if market is open (9:30 AM - 4:00 PM ET, weekdays)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_weekday = now_et.weekday() < 5
    is_market_hours = market_open <= now_et <= market_close
    market_is_open = is_weekday and is_market_hours

    market_status = "MARKET OPEN" if market_is_open else "MARKET CLOSED"
    status_color = "#00ff00" if market_is_open else "#ff6b6b"

    # Header with title and clock inline
    st.markdown(f"""
    <link href="https://fonts.cdnfonts.com/css/digital-7-mono" rel="stylesheet">
    <div style="display:flex; justify-content:space-between; align-items:center; padding:0 0 16px 0; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:16px;">
        <div>
            <h1 style="margin:0; font-size:36px; font-weight:700;">Mean Reversion Strategy v1</h1>
            <p style="margin:4px 0 0 0; color:rgba(255,255,255,0.5); font-size:14px;">Daily MOC signals for S&P 500 stocks</p>
        </div>
        <div style="text-align:right; display:flex; align-items:center; gap:20px;">
            <div>
                <p style="margin:0; font-family:'Digital-7 Mono', 'DSEG7 Classic', 'Courier New', monospace; font-size:56px; color:#39ff14; letter-spacing:2px;">
                    {now_et.strftime('%H:%M')}
                </p>
                <p style="margin:4px 0 0 0; color:#666; font-size:12px; text-align:right;">Eastern Time</p>
            </div>
            <div style="padding:10px 16px; border-radius:10px; background:{'rgba(0,255,0,0.1)' if market_is_open else 'rgba(255,100,100,0.1)'}; border:1px solid {status_color}40;">
                <p style="margin:0; font-size:13px; color:{status_color}; font-weight:600;">● {market_status}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <p style="font-size: 11px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Strategy</p>
            <p style="font-size: 18px; font-weight: 600; color: #fff; margin: 0;">Mean Reversion</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-card" style="padding: 16px;">
            <p style="font-size: 11px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">Parameters</p>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Lookback</span>
                    <span style="color: #fff; font-weight: 500; font-size: 13px;">{ROLLING_WINDOW} days</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Min Consistency</span>
                    <span style="color: #fff; font-weight: 500; font-size: 13px;">{MIN_CONSISTENCY*100:.0f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Min Drop</span>
                    <span style="color: #fff; font-weight: 500; font-size: 13px;">{MIN_DROP_PCT*100:.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Max Positions</span>
                    <span style="color: #fff; font-weight: 500; font-size: 13px;">{MAX_POSITIONS}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Order Type</span>
                    <span style="color: #4a9eff; font-weight: 500; font-size: 13px;">{ORDER_TYPE}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        if st.button("Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Load data
    universe = load_universe()
    sp500 = load_sp500_members()

    if not universe:
        st.error("Run the backtester first to generate universe_cache.csv")
        return

    # Main content - two equal columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # Check if we have saved signals from GitHub
        saved_signals_date = None
        github_error = None
        saved_data, github_error = fetch_from_github(GITHUB_SIGNALS_URL)
        if saved_data:
            saved_signals_date = saved_data.get('date')

        # Show GitHub error if any
        if github_error:
            st.error(f"⚠️ Cannot fetch from GitHub: {github_error}")

        today_str = now_et.strftime('%Y-%m-%d')
        is_trading_day = now_et.weekday() < 5  # Monday-Friday

        # We'll determine the header after loading price data to know the actual date
        header_placeholder = st.empty()

        # OPTIMIZED: Check drops FIRST, then only calculate scores for dropped stocks
        with st.spinner("Checking today's drops..."):
            all_drops = get_drops_batch(universe)

        # Filter to stocks that dropped enough
        down_stocks = {t: d for t, d in all_drops.items() if d['drop_pct'] >= MIN_DROP_PCT * 100}

        if not down_stocks:
            scores = pd.DataFrame()
            close = None
            data_date_str = None
            data_date_display = None
        else:
            # Only download historical data for dropped stocks
            down_tickers = list(down_stocks.keys())
            with st.spinner(f"Calculating scores for {len(down_tickers)} dropped stocks..."):
                prices = get_prices(down_tickers)
                close = get_close_prices(prices)
                scores = calculate_scores(close)

        # Now determine the actual data date and show appropriate header
        data_date_str = None
        if close is not None and not close.empty:
            last_data_date = close.index[-1]
            data_date_str = last_data_date.strftime('%Y-%m-%d')
            data_date_display = last_data_date.strftime('%b %d, %Y')

        # Determine what to show in the header
        if saved_signals_date and saved_signals_date != today_str:
            # Signals are from a previous date
            from datetime import datetime as dt
            sig_date = dt.strptime(saved_signals_date, '%Y-%m-%d')
            date_display = sig_date.strftime('%b %d, %Y')
            header_placeholder.markdown(f"""
            <p style="font-size: 11px; font-weight: 500; color: rgba(255,170,0,0.8); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">⚠️ Signals from {date_display}</p>
            <p style="font-size: 26px; font-weight: 600; color: #fff; margin: 0 0 12px 0;">Recorded Signals</p>
            """, unsafe_allow_html=True)
        elif data_date_str and data_date_str != today_str:
            # Data is from a previous trading day
            header_placeholder.markdown(f"""
            <p style="font-size: 11px; font-weight: 500; color: rgba(255,170,0,0.8); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">⚠️ Data from {data_date_display} (last trading day)</p>
            <p style="font-size: 26px; font-weight: 600; color: #fff; margin: 0 0 12px 0;">Live Preview</p>
            """, unsafe_allow_html=True)
        elif not is_trading_day:
            # Weekend - market closed
            header_placeholder.markdown(f"""
            <p style="font-size: 11px; font-weight: 500; color: rgba(255,170,0,0.8); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">⚠️ Market Closed (Weekend)</p>
            <p style="font-size: 26px; font-weight: 600; color: #fff; margin: 0 0 12px 0;">Live Preview</p>
            """, unsafe_allow_html=True)
        elif not market_is_open:
            # Weekday but outside market hours
            header_placeholder.markdown(f"""
            <p style="font-size: 11px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Market Closed</p>
            <p style="font-size: 26px; font-weight: 600; color: #fff; margin: 0 0 12px 0;">Live Preview</p>
            """, unsafe_allow_html=True)
        else:
            header_placeholder.markdown("""
            <p style="font-size: 11px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Positions</p>
            <p style="font-size: 26px; font-weight: 600; color: #fff; margin: 0 0 12px 0;">Today's Signals</p>
            """, unsafe_allow_html=True)

        # Use the already-fetched drop data (no need to fetch again)
        # down_stocks already has all stocks that dropped >= MIN_DROP_PCT
        signals = generate_todays_signals(scores, down_stocks, sp500) if not scores.empty else []

        # Display signals
        if signals:
            allocation = 100 / MAX_POSITIONS

            # Calculate total portfolio change (weighted by allocation)
            invested_pct = len(signals) * allocation / 100
            total_change_pct = sum(sig['change_pct'] for sig in signals) / len(signals) * invested_pct
            total_color = "#ff5252" if total_change_pct < 0 else "#00ff88"
            total_sign = "" if total_change_pct < 0 else "+"
            border_color = "rgba(255, 82, 82, 0.3)" if total_change_pct < 0 else "rgba(0, 255, 136, 0.3)"

            # Portfolio summary header
            st.markdown(f"""
            <div class="portfolio-summary animate-in" style="border-color: {border_color}; padding: 18px 24px; margin-bottom: 12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; gap:32px; align-items:center;">
                        <div>
                            <p class="metric-label" style="font-size:10px;">Positions</p>
                            <p style="margin:0; font-size:22px; font-weight:600; color:#fff;">{len(signals)}<span style="color:rgba(255,255,255,0.4);"> / {MAX_POSITIONS}</span></p>
                        </div>
                        <div>
                            <p class="metric-label" style="font-size:10px;">Invested</p>
                            <p style="margin:0; font-size:22px; font-weight:600; color:#fff;">{len(signals) * allocation:.0f}%</p>
                        </div>
                        <div>
                            <p class="metric-label" style="font-size:10px;">Cash</p>
                            <p style="margin:0; font-size:22px; font-weight:600; color:rgba(255,255,255,0.5);">{(MAX_POSITIONS - len(signals)) * allocation:.0f}%</p>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <p class="metric-label" style="font-size:10px;">Day Change</p>
                        <p style="margin:0; font-size:32px; font-weight:600; color:{total_color};">{total_sign}{total_change_pct:.2f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Position cards
            for i, sig in enumerate(signals):
                change_color = "#ff5252" if sig['change_pct'] < 0 else "#00ff88"
                change_sign = "" if sig['change_pct'] < 0 else "+"
                card_class = "position-card negative" if sig['change_pct'] < 0 else "position-card"
                change_badge_class = "ticker-change negative" if sig['change_pct'] < 0 else "ticker-change positive"

                st.markdown(f"""
                <div class="{card_class} animate-in" style="animation-delay: {i * 0.05}s; padding: 16px 20px; margin: 8px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="display:flex; align-items:center; gap:18px;">
                            <span style="font-size:20px; font-weight:700; color:#fff; letter-spacing:0.5px;">{sig['ticker']}</span>
                            <span style="color:rgba(255,255,255,0.5); font-size:15px;">${sig['prev_close']:.2f} → <span style="color:{change_color};">${sig['current_price']:.2f}</span></span>
                        </div>
                        <div style="display:flex; align-items:center; gap:16px;">
                            <span style="color:{change_color}; font-size:15px; font-weight:500;">{change_sign}${abs(sig['change_dollar']):.2f}</span>
                            <span class="{change_badge_class}" style="font-size:14px; padding:5px 12px;">{change_sign}{sig['change_pct']:.2f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Show why there are no signals
            if not down_stocks:
                reason = f"No stocks dropped ≥ {MIN_DROP_PCT*100:.2f}% today"
            elif scores.empty:
                reason = "No dropped stocks passed scoring filter"
            else:
                reason = "No stocks met all criteria"

            st.markdown(f"""
            <div class="cash-box animate-in" style="padding:16px;">
                <p style="margin:0; color:rgba(255,255,255,0.5); font-size:14px;">No Active Positions</p>
                <p style="margin:8px 0 0 0; font-size:28px; font-weight:600; color:rgba(255,255,255,0.4);">100% Cash</p>
                <p style="margin:8px 0 0 0; color:rgba(255,255,255,0.3); font-size:12px;">{reason}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <p style="font-size: 11px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Performance</p>
        <p style="font-size: 26px; font-weight: 600; color: #fff; margin: 0 0 12px 0;">Intraday Activity</p>
        """, unsafe_allow_html=True)

        # Intraday chart for current positions
        if signals:
            signal_tickers = [sig['ticker'] for sig in signals]
            with st.spinner("Loading intraday data..."):
                intraday_chart_data = get_intraday_chart_data(signal_tickers)

            if intraday_chart_data:
                # Create multi-line chart
                fig_intraday = go.Figure()

                colors = ['#00ff88', '#4a9eff', '#ff9f43']  # Green, Blue, Orange
                for idx, ticker in enumerate(signal_tickers):
                    if ticker in intraday_chart_data:
                        data = intraday_chart_data[ticker]
                        times = data['times']
                        prices = data['prices']

                        # Normalize to percentage change from open
                        if data['open'] and data['open'] > 0:
                            pct_changes = [(p - data['open']) / data['open'] * 100 for p in prices]
                        else:
                            pct_changes = [0] * len(prices)

                        color = colors[idx % len(colors)]
                        fig_intraday.add_trace(go.Scatter(
                            x=times,
                            y=pct_changes,
                            mode='lines',
                            name=ticker,
                            line=dict(color=color, width=2),
                            hovertemplate=f'{ticker}<br>%{{x|%H:%M}}<br>%{{y:.2f}}%<extra></extra>'
                        ))

                # Add zero line
                fig_intraday.add_hline(y=0, line_dash="dash", line_color="#444", line_width=1)

                fig_intraday.update_layout(
                    height=200,
                    margin=dict(t=5, b=25, l=40, r=10),
                    xaxis_title="",
                    yaxis_title="",
                    template='plotly_dark',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                        font=dict(size=12)
                    ),
                    yaxis=dict(ticksuffix="%", tickfont=dict(size=11), zeroline=False),
                    xaxis=dict(tickformat="%H:%M", tickfont=dict(size=11)),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_intraday, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 20px;">
                    <p style="color: rgba(255,255,255,0.5); font-size:12px;">Intraday data not available</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 20px;">
                <p style="color: rgba(255,255,255,0.5); font-size:12px;">No positions to track</p>
            </div>
            """, unsafe_allow_html=True)

        # Trading History section
        st.markdown("""
        <p style="font-size: 11px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 4px 0;">Track Record</p>
        <p style="font-size: 22px; font-weight: 600; color: #fff; margin: 0 0 10px 0;">Trading History</p>
        """, unsafe_allow_html=True)

        record, track_error = load_track_record()

        if track_error:
            st.error(f"⚠️ Cannot fetch track record: {track_error}")

        if not record['entries']:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 16px 20px;">
                <p style="margin:0; color:rgba(255,255,255,0.5); font-size:14px;">📊 No history yet · Recorded at 3:45 PM ET</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Calculate performance
            with st.spinner("Calculating performance..."):
                perf = calculate_track_record_performance(record)

            if perf and len(perf) > 0:
                # Build full data for date range filtering
                cumulative = 1
                all_dates = []
                all_values = []
                all_returns = []

                for p in perf:
                    cumulative *= (1 + p['return'])
                    all_dates.append(datetime.strptime(p['date'], '%Y-%m-%d'))
                    all_values.append((cumulative - 1) * 100)
                    all_returns.append(p['return'])

                # Date range selector
                date_range = st.selectbox(
                    "Date Range",
                    ["All Time", "Last 30 Days", "Last 7 Days", "Last 90 Days"],
                    index=0,
                    key="date_range"
                )

                # Filter data by date range
                today = datetime.now()
                if date_range == "Last 7 Days":
                    cutoff = today - timedelta(days=7)
                elif date_range == "Last 30 Days":
                    cutoff = today - timedelta(days=30)
                elif date_range == "Last 90 Days":
                    cutoff = today - timedelta(days=90)
                else:
                    cutoff = datetime.min

                # Get filtered indices
                filtered_indices = [i for i, d in enumerate(all_dates) if d >= cutoff]

                if filtered_indices:
                    # Recalculate cumulative for filtered period
                    filtered_dates = [all_dates[i] for i in filtered_indices]
                    filtered_returns = [all_returns[i] for i in filtered_indices]

                    period_cumulative = 1
                    filtered_values = []
                    for r in filtered_returns:
                        period_cumulative *= (1 + r)
                        filtered_values.append((period_cumulative - 1) * 100)

                    period_return = (period_cumulative - 1) * 100
                    avg_return = np.mean(filtered_returns) * 100
                    win_rate = sum(1 for r in filtered_returns if r > 0) / len(filtered_returns) * 100

                    # Display metrics in compact glass card
                    period_color = "#00ff88" if period_return >= 0 else "#ff5252"
                    avg_color = "#00ff88" if avg_return >= 0 else "#ff5252"
                    win_color = "#00ff88" if win_rate >= 50 else "#ff5252"

                    st.markdown(f"""
                    <div class="glass-card" style="padding: 10px 14px;">
                        <div style="display: flex; justify-content: space-between;">
                            <div style="text-align: center; flex: 1;">
                                <p class="metric-label" style="font-size:9px;">Return</p>
                                <p style="margin:0; font-size:18px; font-weight:600; color:{period_color};">{period_return:+.1f}%</p>
                            </div>
                            <div style="width: 1px; background: rgba(255,255,255,0.1);"></div>
                            <div style="text-align: center; flex: 1;">
                                <p class="metric-label" style="font-size:9px;">Avg/Day</p>
                                <p style="margin:0; font-size:18px; font-weight:600; color:{avg_color};">{avg_return:+.2f}%</p>
                            </div>
                            <div style="width: 1px; background: rgba(255,255,255,0.1);"></div>
                            <div style="text-align: center; flex: 1;">
                                <p class="metric-label" style="font-size:9px;">Win Rate</p>
                                <p style="margin:0; font-size:18px; font-weight:600; color:{win_color};">{win_rate:.0f}%</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Chart - compact
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_dates,
                        y=filtered_values,
                        mode='lines',
                        line=dict(color='#00ff88' if filtered_values[-1] >= 0 else '#ff5252', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0,255,136,0.1)' if filtered_values[-1] >= 0 else 'rgba(255,82,82,0.1)',
                        hovertemplate='%{x|%m/%d}<br>%{y:.2f}%<extra></extra>'
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1)
                    fig.update_layout(
                        height=120,
                        margin=dict(t=5, b=20, l=35, r=5),
                        xaxis_title="",
                        yaxis_title="",
                        template='plotly_dark',
                        showlegend=False,
                        yaxis=dict(ticksuffix="%", tickfont=dict(size=9)),
                        xaxis=dict(tickformat="%m/%d", tickfont=dict(size=9)),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # Trading history table header - compact
                    st.markdown("""
                    <p style="font-size: 10px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin: 6px 0 4px 0;">Recent Trades</p>
                    """, unsafe_allow_html=True)

                    # Get last 3 entries with performance - very compact
                    recent_entries = list(reversed(record['entries'][-3:]))
                    trade_data = []

                    for i, entry in enumerate(recent_entries):
                        tickers = [s['ticker'] for s in entry['signals']] or ['CASH']
                        # Find corresponding performance
                        entry_date = entry['date']
                        perf_match = next((p for p in perf if p['date'] == entry_date), None)
                        daily_return = perf_match['return'] * 100 if perf_match else None

                        trade_data.append({
                            'Date': entry_date,
                            'Positions': ', '.join(tickers),
                            'Return': daily_return
                        })

                    trade_df = pd.DataFrame(trade_data)

                    def color_return(val):
                        if val is None:
                            return 'color: #888'
                        if val < 0:
                            return 'color: #ff6b6b'
                        elif val > 0:
                            return 'color: #00ff00'
                        return 'color: #888'

                    def format_return(val):
                        if val is None:
                            return 'Pending'
                        sign = "" if val < 0 else "+"
                        return f"{sign}{val:.2f}%"

                    styled_trades = trade_df.style.applymap(color_return, subset=['Return']).format({'Return': format_return})
                    st.dataframe(
                        styled_trades,
                        use_container_width=True,
                        hide_index=True,
                        height=115
                    )
                else:
                    st.markdown("""
                    <div class="glass-card" style="text-align: center; padding: 30px;">
                        <p style="color: rgba(255,255,255,0.5); font-size:16px;">No trades in selected date range</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 40px 24px;">
                    <p style="font-size: 28px; margin: 0 0 10px 0;">⏳</p>
                    <p style="margin:0; color:rgba(255,255,255,0.7); font-weight: 500; font-size:16px;">Awaiting First Trade</p>
                    <p style="margin:10px 0 0 0; color:rgba(255,255,255,0.4); font-size:14px;">
                        Performance calculated after first close
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <p style="font-size: 12px; font-weight: 500; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.5px; margin: 16px 0 12px 0;">Pending</p>
                """, unsafe_allow_html=True)

                for entry in reversed(record['entries'][-3:]):
                    tickers = [s['ticker'] for s in entry['signals']] or ['CASH']
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: rgba(255,255,255,0.5); font-size: 14px;">{entry['date']}</span>
                        <span style="color: #fff; font-weight: 500; font-size: 14px;">{', '.join(tickers)}</span>
                    </div>
                    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APP WITH NAVIGATION
# =============================================================================

def main():
    """Main app with navigation."""
    page = st.sidebar.radio(
        "Navigation",
        ["Signals", "Technical Details"],
        label_visibility="collapsed"
    )

    if page == "Signals":
        show_signals_page()
    else:
        show_technical_details()

if __name__ == "__main__":
    main()
