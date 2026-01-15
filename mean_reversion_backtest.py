"""
Mean Reversion Strategy Backtester
===================================
Mean reversion strategy targeting high-beta stocks in specific industries.

Captures bounce-back behavior after down days using:
- Reversion Score (E): Sum of next-day returns after down days
- Consistency Score (C): % of times stock bounces after a drop
- Final Score = E × C

Daily execution at 3:45 PM, MOC orders, max 3 positions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Target Industries - SYSTEMIC POWER UNIVERSE
# Focus: Government contracts, regulated monopolies, too-big-to-fail institutions
TARGET_INDUSTRIES = [
    'Aerospace & Defense',
    'Banks - Diversified',
    'Banks - Regional',
    'Insurance - Property & Casualty',
    'Insurance - Life',
    'Insurance - Diversified',
    'Healthcare Plans',
    'Medical Care Facilities',
    'Utilities - Regulated Electric',
    'Utilities - Diversified',
    'Utilities - Independent Power Producers',
    'Telecom Services',
    'Information Technology Services',
    'Waste Management',
    'Engineering & Construction',
]

# Yahoo Finance uses slightly different industry names - create mapping
INDUSTRY_ALIASES = {
    'Aerospace & Defense': ['Aerospace & Defense', 'Aerospace/Defense'],
    'Banks - Diversified': ['Banks - Diversified', 'Diversified Banks', 'Banks—Diversified', 'Money Center Banks'],
    'Banks - Regional': ['Banks - Regional', 'Regional Banks', 'Banks—Regional'],
    'Insurance - Property & Casualty': ['Insurance - Property & Casualty', 'Property & Casualty Insurance'],
    'Insurance - Life': ['Insurance - Life', 'Life Insurance'],
    'Insurance - Diversified': ['Insurance - Diversified', 'Multi-line Insurance', 'Insurance—Diversified'],
    'Healthcare Plans': ['Healthcare Plans', 'Health Care Plans', 'Managed Healthcare'],
    'Medical Care Facilities': ['Medical Care Facilities', 'Health Care Facilities', 'Hospitals'],
    'Utilities - Regulated Electric': ['Utilities - Regulated Electric', 'Utilities—Regulated Electric', 'Electric Utilities'],
    'Utilities - Diversified': ['Utilities - Diversified', 'Utilities—Diversified', 'Multi-Utilities'],
    'Utilities - Independent Power Producers': ['Utilities - Independent Power Producers', 'Independent Power Producers'],
    'Telecom Services': ['Telecom Services', 'Telecommunications Services', 'Wireless Telecommunications'],
    'Information Technology Services': ['Information Technology Services', 'IT Services'],
    'Waste Management': ['Waste Management', 'Environmental Services'],
    'Engineering & Construction': ['Engineering & Construction', 'Construction & Engineering'],
}

# Strategy Parameters
ROLLING_WINDOW = 378  # Days for E and C calculation
MIN_CONSISTENCY = 0.55  # 55% minimum bounce rate
TOP_CANDIDATES = 6  # Top stocks by score
MAX_POSITIONS = 3  # Final portfolio size
MIN_DROP_PCT = 0.0020  # 0.20% minimum intraday drop
POSITION_SIZE = 1/3  # Equal weight

# Backtest Parameters
START_DATE = '2015-01-01'
END_DATE = '2025-11-30'
INITIAL_CAPITAL = 100000

# Transaction Costs (realistic for MOC orders on liquid S&P 500 stocks)
SLIPPAGE_PCT = 0.0005  # 0.05% slippage per trade (5 bps)
COMMISSION_PER_TRADE = 0.0  # $0 commission (most brokers now)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_sp500_constituents(data_path: str) -> pd.DataFrame:
    """Load historical S&P 500 constituents from CSV."""
    csv_files = list(Path(data_path).glob('S&P 500 Historical Components*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No S&P 500 constituents CSV found in {data_path}")

    # Use the most recent file
    csv_file = sorted(csv_files)[-1]
    print(f"Loading constituents from: {csv_file.name}")

    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df


def get_constituents_for_date(constituents_df: pd.DataFrame, date: pd.Timestamp) -> set:
    """Get S&P 500 members for a specific date."""
    # Find the most recent date <= target date
    valid_dates = constituents_df.index[constituents_df.index <= date]
    if len(valid_dates) == 0:
        return set()

    closest_date = valid_dates[-1]
    tickers_str = constituents_df.loc[closest_date, 'tickers']
    return set(tickers_str.split(','))


def get_stock_info(ticker: str) -> dict:
    """Get stock info including industry from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'ticker': ticker,
            'industry': info.get('industry', ''),
            'sector': info.get('sector', ''),
            'name': info.get('shortName', ticker)
        }
    except Exception as e:
        return {'ticker': ticker, 'industry': '', 'sector': '', 'name': ticker}


def matches_target_industry(industry: str) -> bool:
    """Check if stock's industry matches our target list."""
    if not industry:
        return False

    industry_lower = industry.lower()
    for target, aliases in INDUSTRY_ALIASES.items():
        for alias in aliases:
            if alias.lower() in industry_lower or industry_lower in alias.lower():
                return True
    return False


def build_universe(constituents_df: pd.DataFrame, reference_date: str = None) -> pd.DataFrame:
    """
    Build the trading universe: S&P 500 stocks in target industries.
    Caches industry info to avoid repeated API calls.
    """
    cache_file = Path('universe_cache.csv')

    if cache_file.exists():
        print("Loading cached universe...")
        universe = pd.read_csv(cache_file)
        return universe

    print("Building universe (this may take a few minutes)...")

    # Get all unique tickers from history
    all_tickers = set()
    for tickers_str in constituents_df['tickers']:
        all_tickers.update(tickers_str.split(','))

    print(f"Found {len(all_tickers)} unique historical S&P 500 tickers")

    # Get industry info for each
    universe_data = []
    for i, ticker in enumerate(sorted(all_tickers)):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(all_tickers)}...")

        info = get_stock_info(ticker)
        if matches_target_industry(info['industry']):
            universe_data.append(info)

    universe = pd.DataFrame(universe_data)
    print(f"Universe contains {len(universe)} stocks in target industries")

    # Cache for future runs
    universe.to_csv(cache_file, index=False)
    return universe


# =============================================================================
# PRICE DATA
# =============================================================================

def download_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Download OHLCV data for all tickers."""
    cache_file = Path('price_cache.pkl')

    if cache_file.exists():
        print("Loading cached price data...")
        prices = pd.read_pickle(cache_file)
        return prices

    print(f"Downloading price data for {len(tickers)} tickers...")

    # Download in batches to avoid timeouts
    batch_size = 50
    all_data = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")

        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_data = data.copy()
                    else:
                        ticker_data = data.xs(ticker, axis=1, level=1)

                    if not ticker_data.empty:
                        all_data[ticker] = ticker_data
                except:
                    pass
        except Exception as e:
            print(f"  Error in batch: {e}")

    prices = pd.concat(all_data, axis=1)
    prices.to_pickle(cache_file)
    print(f"Cached price data for {len(all_data)} tickers")

    return prices


# =============================================================================
# MEAN REVERSION CALCULATIONS
# =============================================================================

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from close prices."""
    close = prices.xs('Close', axis=1, level=1) if 'Close' in prices.columns.get_level_values(1) else prices
    returns = close.pct_change()
    return returns


def calculate_reversion_score(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """
    Calculate Reversion Score (E):
    Sum of next-day returns following down days over rolling window.
    """
    # Identify down days
    down_days = returns < 0

    # Next day returns
    next_day_returns = returns.shift(-1)

    # Sum of next-day returns after down days (rolling)
    down_day_next_returns = next_day_returns.where(down_days, 0)

    reversion = down_day_next_returns.rolling(window=window, min_periods=window//2).sum()
    return reversion


def calculate_consistency_score(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """
    Calculate Consistency Score (C):
    Percentage of times stock bounces (positive return) after a down day.
    """
    # Identify down days
    down_days = returns < 0

    # Next day returns
    next_day_returns = returns.shift(-1)

    # Bounces: next day positive after down day
    bounces = (next_day_returns > 0) & down_days

    # Rolling counts
    bounce_count = bounces.astype(int).rolling(window=window, min_periods=window//2).sum()
    down_count = down_days.astype(int).rolling(window=window, min_periods=window//2).sum()

    consistency = bounce_count / down_count.replace(0, np.nan)
    return consistency


def calculate_final_score(returns: pd.Series, window: int = ROLLING_WINDOW) -> tuple:
    """Calculate E, C, and Final Score = E × C."""
    e = calculate_reversion_score(returns, window)
    c = calculate_consistency_score(returns, window)
    score = e * c
    return e, c, score


# =============================================================================
# DAILY EXECUTION LOGIC
# =============================================================================

def get_intraday_drop(open_price: float, current_price: float) -> float:
    """Calculate intraday drop percentage."""
    if open_price == 0 or pd.isna(open_price):
        return 0
    return (open_price - current_price) / open_price


def select_daily_portfolio(
    scores: pd.Series,
    consistency: pd.Series,
    open_prices: pd.Series,
    close_prices: pd.Series,
    sp500_members: set,
    top_n: int = TOP_CANDIDATES,
    max_positions: int = MAX_POSITIONS,
    min_consistency: float = MIN_CONSISTENCY,
    min_drop: float = MIN_DROP_PCT
) -> list:
    """
    Daily portfolio selection logic:
    1. Filter by consistency >= 60%
    2. Get top 6 by score
    3. Keep only those down >= 0.20% intraday
    4. Select top 3 (tiebreaker: deepest drop)
    """
    # Start with valid scores
    valid = scores.dropna()

    # Filter: must be current S&P 500 member
    valid = valid[valid.index.isin(sp500_members)]

    # Filter: consistency >= 60%
    valid_consistency = consistency.reindex(valid.index)
    valid = valid[valid_consistency >= min_consistency]

    if len(valid) == 0:
        return []

    # Step 1: Top 6 by score
    top_candidates = valid.nlargest(top_n)

    # Step 2: Filter by intraday drop
    candidates_with_drop = []
    for ticker in top_candidates.index:
        open_p = open_prices.get(ticker, np.nan)
        close_p = close_prices.get(ticker, np.nan)

        if pd.isna(open_p) or pd.isna(close_p):
            continue

        drop = get_intraday_drop(open_p, close_p)

        if drop >= min_drop:  # Down at least 0.20%
            candidates_with_drop.append((ticker, drop, top_candidates[ticker]))

    if len(candidates_with_drop) == 0:
        return []

    # Step 3: Sort by drop (deepest first) for tiebreaker
    candidates_with_drop.sort(key=lambda x: x[1], reverse=True)

    # Select top 3
    selected = [c[0] for c in candidates_with_drop[:max_positions]]
    return selected


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(
    prices: pd.DataFrame,
    constituents_df: pd.DataFrame,
    universe_tickers: list,
    start_date: str,
    end_date: str,
    initial_capital: float = INITIAL_CAPITAL
) -> pd.DataFrame:
    """
    Run the mean reversion strategy backtest.
    """
    print("\nRunning backtest...")

    # Get close and open prices
    close_prices = prices.xs('Close', axis=1, level=1) if isinstance(prices.columns, pd.MultiIndex) else prices
    open_prices = prices.xs('Open', axis=1, level=1) if isinstance(prices.columns, pd.MultiIndex) else prices

    # Filter to universe
    close_prices = close_prices[[c for c in close_prices.columns if c in universe_tickers]]
    open_prices = open_prices[[c for c in open_prices.columns if c in universe_tickers]]

    # Calculate returns
    returns = close_prices.pct_change()

    # Pre-calculate all scores
    print("Calculating mean reversion scores...")
    all_e = {}
    all_c = {}
    all_scores = {}

    for ticker in returns.columns:
        e, c, score = calculate_final_score(returns[ticker])
        all_e[ticker] = e
        all_c[ticker] = c
        all_scores[ticker] = score

    scores_df = pd.DataFrame(all_scores)
    consistency_df = pd.DataFrame(all_c)

    # Backtest loop
    dates = returns.loc[start_date:end_date].index

    # Need warmup period
    warmup_end = dates[0]
    warmup_start = warmup_end - timedelta(days=ROLLING_WINDOW + 50)

    results = []
    current_positions = []
    capital = initial_capital
    total_trades = 0
    total_slippage = 0

    for i, date in enumerate(dates):
        if i % 50 == 0:
            print(f"  Processing {date.strftime('%Y-%m-%d')} ({i}/{len(dates)})")

        # Get S&P 500 members for this date
        sp500_members = get_constituents_for_date(constituents_df, date)

        # Get today's scores and prices
        try:
            day_scores = scores_df.loc[date]
            day_consistency = consistency_df.loc[date]
            day_open = open_prices.loc[date]
            day_close = close_prices.loc[date]
        except KeyError:
            continue

        # Calculate returns for current positions
        daily_return = 0
        if current_positions:
            for ticker in current_positions:
                if ticker in returns.columns:
                    ret = returns.loc[date, ticker]
                    if not pd.isna(ret):
                        daily_return += ret * POSITION_SIZE

        # Update capital
        capital *= (1 + daily_return)

        # Select new portfolio for tomorrow
        new_positions = select_daily_portfolio(
            scores=day_scores,
            consistency=day_consistency,
            open_prices=day_open,
            close_prices=day_close,
            sp500_members=sp500_members
        )

        # Calculate transaction costs (slippage on entry and exit)
        # Positions being sold (in current but not in new)
        exits = set(current_positions) - set(new_positions)
        # Positions being bought (in new but not in current)
        entries = set(new_positions) - set(current_positions)

        num_trades = len(exits) + len(entries)
        if num_trades > 0:
            # Slippage applies to the portion of capital being traded
            # Each position is POSITION_SIZE of capital
            traded_capital_pct = num_trades * POSITION_SIZE
            slippage_cost = traded_capital_pct * SLIPPAGE_PCT
            capital *= (1 - slippage_cost)
            total_trades += num_trades
            total_slippage += slippage_cost * capital

        # Record results
        results.append({
            'date': date,
            'capital': capital,
            'daily_return': daily_return,
            'positions': current_positions.copy(),
            'num_positions': len(current_positions),
            'new_positions': new_positions,
            'trades': num_trades
        })

        # Update positions for next day
        current_positions = new_positions

    print(f"\nTotal trades: {total_trades}")
    print(f"Avg trades per day: {total_trades/len(dates):.2f}")

    results_df = pd.DataFrame(results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.set_index('date')

    return results_df


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_metrics(results: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> dict:
    """Calculate strategy performance metrics."""
    returns = results['daily_return']
    capital = results['capital']

    # Basic metrics
    total_return = (capital.iloc[-1] / initial_capital - 1) * 100
    trading_days = len(returns)
    years = trading_days / 252

    # Annualized return
    cagr = ((capital.iloc[-1] / initial_capital) ** (1/years) - 1) * 100

    # Volatility
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100

    # Sharpe Ratio (assuming 0% risk-free rate)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Drawdown
    peak = capital.cummax()
    drawdown = (capital - peak) / peak
    max_drawdown = drawdown.min() * 100

    # Win rate
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    total_trading_days = winning_days + losing_days
    win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0

    # Average position count
    avg_positions = results['num_positions'].mean()

    return {
        'Total Return (%)': round(total_return, 2),
        'CAGR (%)': round(cagr, 2),
        'Annual Volatility (%)': round(annual_vol, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Trading Days': trading_days,
        'Avg Positions': round(avg_positions, 2)
    }


def print_results(metrics: dict, results: pd.DataFrame):
    """Print backtest results."""
    print("\n" + "="*60)
    print("MEAN REVERSION STRATEGY BACKTEST RESULTS")
    print("="*60)

    for key, value in metrics.items():
        print(f"{key:.<30} {value}")

    print("\n" + "-"*60)
    print("EQUITY CURVE SUMMARY")
    print("-"*60)

    capital = results['capital']
    print(f"Start:  ${INITIAL_CAPITAL:,.0f}")
    print(f"End:    ${capital.iloc[-1]:,.0f}")
    print(f"High:   ${capital.max():,.0f}")
    print(f"Low:    ${capital.min():,.0f}")

    # Monthly returns
    print("\n" + "-"*60)
    print("MONTHLY RETURNS (%)")
    print("-"*60)

    monthly = results['daily_return'].resample('M').apply(lambda x: (1+x).prod()-1) * 100
    monthly_pivot = monthly.to_frame('return')
    monthly_pivot['year'] = monthly_pivot.index.year
    monthly_pivot['month'] = monthly_pivot.index.month
    monthly_table = monthly_pivot.pivot(index='year', columns='month', values='return')
    monthly_table.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    print(monthly_table.round(1).to_string())


def save_results(results: pd.DataFrame, metrics: dict):
    """Save results to files."""
    # Save equity curve
    results[['capital', 'daily_return', 'num_positions']].to_csv('equity_curve.csv')

    # Save metrics
    pd.Series(metrics).to_csv('metrics.csv')

    # Save positions history
    positions_history = results[['positions', 'new_positions']].copy()
    positions_history['positions'] = positions_history['positions'].apply(lambda x: ','.join(x) if x else '')
    positions_history['new_positions'] = positions_history['new_positions'].apply(lambda x: ','.join(x) if x else '')
    positions_history.to_csv('positions_history.csv')

    print("\nResults saved to:")
    print("  - equity_curve.csv")
    print("  - metrics.csv")
    print("  - positions_history.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("MEAN REVERSION STRATEGY BACKTESTER")
    print("="*60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Rolling Window: {ROLLING_WINDOW} days")
    print(f"Max Positions: {MAX_POSITIONS}")
    print()

    # Load constituents
    constituents_df = load_sp500_constituents('sp500_data')

    # Build universe
    universe = build_universe(constituents_df)
    universe_tickers = universe['ticker'].tolist()

    print(f"\nUniverse: {len(universe_tickers)} stocks in target industries")
    print(f"Industries: {universe['industry'].value_counts().head(10).to_dict()}")

    # Download price data (with buffer for warmup)
    buffer_start = (pd.Timestamp(START_DATE) - timedelta(days=ROLLING_WINDOW + 100)).strftime('%Y-%m-%d')
    prices = download_price_data(universe_tickers, buffer_start, END_DATE)

    # Run backtest
    results = run_backtest(
        prices=prices,
        constituents_df=constituents_df,
        universe_tickers=universe_tickers,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # Calculate and print metrics
    metrics = calculate_metrics(results)
    print_results(metrics, results)

    # Save results
    save_results(results, metrics)

    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
