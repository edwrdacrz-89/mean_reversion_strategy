"""
Mean Reversion Strategy Dashboard
==================================
Displays recorded signals and track record from GitHub.
All calculations are done by cloud_signal_generator.py on GitHub Actions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import json
import requests
import base64

# =============================================================================
# CONFIG
# =============================================================================

MAX_POSITIONS = 3
GITHUB_REPO = "edwrdacrz-89/mean_reversion_strategy"
GITHUB_TRACK_RECORD_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/track_record.json"
GITHUB_SIGNALS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/signals.json"

st.set_page_config(page_title="Mean Reversion Strategy", page_icon="📈", layout="wide")

# =============================================================================
# STYLES
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, .stDeployButton { display: none; }
    .block-container { padding-top: 1rem; max-width: 1400px; }

    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 10px;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=60)
def fetch_github(url):
    """Fetch JSON from GitHub API."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            content = base64.b64decode(resp.json()['content']).decode('utf-8')
            return json.loads(content), None
        return None, f"GitHub error: {resp.status_code}"
    except Exception as e:
        return None, str(e)

# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo('America/New_York'))

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:16px; border-bottom:1px solid rgba(255,255,255,0.1); margin-bottom:20px;">
        <div>
            <h1 style="margin:0; font-size:32px;">Mean Reversion Strategy</h1>
            <p style="margin:4px 0 0 0; color:rgba(255,255,255,0.5);">Daily MOC signals · Updated at 3:45 PM ET</p>
        </div>
        <div style="text-align:right;">
            <p style="margin:0; font-size:42px; font-family:monospace; color:#39ff14;">{now_et.strftime('%H:%M')}</p>
            <p style="margin:0; color:#666; font-size:11px;">Eastern Time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch data from GitHub
    signals_data, signals_err = fetch_github(GITHUB_SIGNALS_URL)
    record_data, record_err = fetch_github(GITHUB_TRACK_RECORD_URL)

    col1, col2 = st.columns([1, 1])

    # === LEFT COLUMN: TODAY'S SIGNALS ===
    with col1:
        st.markdown("### Today's Signals")

        if signals_err:
            st.error(f"Cannot fetch signals: {signals_err}")
        elif not signals_data:
            st.info("No signals data available")
        else:
            sig_date = signals_data.get('date', 'Unknown')
            signals = signals_data.get('signals', [])
            cash_pct = signals_data.get('cash_allocation', 100)

            # Show date
            st.caption(f"Recorded: {sig_date}")

            if signals:
                for sig in signals:
                    drop_pct = sig.get('drop_pct', 0)
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:20px; font-weight:700;">{sig['ticker']}</span>
                            <div style="text-align:right;">
                                <span style="color:#ff5252;">-{drop_pct:.2f}%</span>
                                <p style="margin:0; font-size:12px; color:rgba(255,255,255,0.4);">@ ${sig.get('price', 0):.2f}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <p style="color:rgba(255,255,255,0.4); font-size:12px; margin-top:12px;">
                    {len(signals)} positions · {100-cash_pct:.0f}% invested · {cash_pct:.0f}% cash
                </p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding:24px;">
                    <p style="font-size:24px; color:rgba(255,255,255,0.4); margin:0;">100% Cash</p>
                    <p style="color:rgba(255,255,255,0.3); font-size:12px; margin-top:8px;">No signals met criteria</p>
                </div>
                """, unsafe_allow_html=True)

    # === RIGHT COLUMN: TRACK RECORD ===
    with col2:
        st.markdown("### Track Record")

        if record_err:
            st.error(f"Cannot fetch track record: {record_err}")
        elif not record_data:
            st.info("No track record available")
        else:
            perf = record_data.get('performance', {})
            entries = record_data.get('entries', [])

            total_return = perf.get('total_return', 0)
            total_trades = perf.get('total_trades', 0)
            win_rate = perf.get('win_rate', 0)

            # Performance summary
            if total_trades > 0:
                ret_color = "#00ff88" if total_return >= 0 else "#ff5252"
                wr_color = "#00ff88" if win_rate >= 50 else "#ff5252"

                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <p class="metric-label">Compounded Return</p>
                            <p style="margin:0; font-size:32px; font-weight:700; color:{ret_color};">{total_return:+.2f}%</p>
                        </div>
                        <div style="display:flex; gap:24px;">
                            <div style="text-align:center;">
                                <p class="metric-label">Trades</p>
                                <p style="margin:0; font-size:20px; font-weight:600;">{total_trades}</p>
                            </div>
                            <div style="text-align:center;">
                                <p class="metric-label">Win Rate</p>
                                <p style="margin:0; font-size:20px; font-weight:600; color:{wr_color};">{win_rate:.1f}%</p>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Equity curve from entries
            perf_entries = [e for e in entries if 'portfolio_return' in e]
            if perf_entries:
                cumulative = 1
                dates = []
                values = []
                for e in perf_entries:
                    cumulative *= (1 + e['portfolio_return'] / 100)
                    dates.append(datetime.strptime(e['date'], '%Y-%m-%d'))
                    values.append((cumulative - 1) * 100)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=values, mode='lines',
                    line=dict(color='#00ff88' if values[-1] >= 0 else '#ff5252', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,136,0.1)' if values[-1] >= 0 else 'rgba(255,82,82,0.1)'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="#333")
                fig.update_layout(
                    height=150, margin=dict(t=10, b=30, l=40, r=10),
                    template='plotly_dark', showlegend=False,
                    yaxis=dict(ticksuffix="%", tickfont=dict(size=10)),
                    xaxis=dict(tickformat="%m/%d", tickfont=dict(size=10)),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Recent trades
            if entries:
                st.markdown("**Recent Trades**")
                for entry in reversed(entries[-5:]):
                    tickers = [s['ticker'] for s in entry.get('signals', [])] or ['CASH']
                    ret = entry.get('portfolio_return')
                    ret_str = f"{ret:+.2f}%" if ret is not None else "Pending"
                    ret_color = "#00ff88" if ret and ret > 0 else "#ff5252" if ret and ret < 0 else "#888"

                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="color:rgba(255,255,255,0.5); font-size:13px;">{entry['date']}</span>
                        <span style="font-size:13px;">{', '.join(tickers)}</span>
                        <span style="color:{ret_color}; font-size:13px;">{ret_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Export CSV button
                csv_rows = []
                cumulative = 1
                portfolio_value = 100.0
                for entry in entries:
                    tickers = [s['ticker'] for s in entry.get('signals', [])] or ['CASH']
                    ret = entry.get('portfolio_return')
                    if ret is not None:
                        cumulative *= (1 + ret / 100)
                        portfolio_value = 100.0 * cumulative
                    csv_rows.append({
                        'Date': entry['date'],
                        'Positions': ', '.join(tickers),
                        'Return (%)': f"{ret:.2f}" if ret is not None else '',
                        'Cumulative (%)': f"{(cumulative - 1) * 100:.2f}",
                        'Portfolio Value': f"{portfolio_value:.2f}"
                    })
                csv_df = pd.DataFrame(csv_rows)
                csv_data = csv_df.to_csv(index=False)
                st.download_button(
                    "Export CSV",
                    csv_data,
                    "trade_history.csv",
                    "text/csv",
                    use_container_width=True
                )
            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center;">
                    <p style="color:rgba(255,255,255,0.5);">No history yet</p>
                </div>
                """, unsafe_allow_html=True)

def show_technical_details():
    """Display the technical details page."""
    st.title("Technical Details")
    st.caption("Complete documentation of the mean reversion strategy")

    st.header("Strategy Overview")
    st.markdown("""
    This strategy capitalizes on short-term price corrections in high-quality, systemically important stocks.
    We identify stocks with strong historical mean reversion characteristics and buy them when they drop, expecting a recovery.
    """)

    st.header("Execution Rules")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Timing")
        st.markdown("""
        - Signal Generation: 3:45 PM ET
        - Order Type: MOC (Market-On-Close)
        - Holding Period: 1 day (close-to-close)
        """)

        st.subheader("2. Daily Workflow")
        st.markdown("""
        1. SELL all existing positions (MOC)
        2. Calculate total portfolio value
        3. BUY new signals with equal weight (MOC)
        """)

    with col2:
        st.subheader("3. Position Sizing")
        st.markdown("""
        - Max Positions: 3
        - Weight per Position: 33.33%
        - If fewer signals: Remainder stays in cash
        """)

        st.subheader("4. Signal Criteria")
        st.markdown("""
        - Universe: 125 stocks (Power Universe)
        - Min Consistency (C): 55%
        - Min Drop: 0.20% from previous close
        """)

    st.header("Scoring System")

    st.subheader("E Score (Elasticity)")
    st.markdown("""
    Measures the *strength* of historical recovery after down days.
    ```
    E = Sum of next-day returns after all down days (over 378 trading days)
    ```
    Higher E = Stock bounces more aggressively after drops.
    """)

    st.subheader("C Score (Consistency)")
    st.markdown("""
    Measures the *probability* of a rebound.
    ```
    C = (Bounces after down days) / (Total down days)
    ```
    We require C ≥ 55% to filter out volatile stocks.
    """)

    st.subheader("Final Score")
    st.markdown("""
    ```
    Final Score = E × C
    ```
    Combines magnitude (E) with reliability (C).
    """)

    st.header("Signal Selection Process")
    st.markdown("""
    1. Check which stocks dropped ≥ 0.20% today (fast batch check)
    2. Download historical data only for dropped stocks
    3. Calculate E×C scores for dropped stocks
    4. Filter: Keep only stocks with C ≥ 55%
    5. Take top 6 by score
    6. Select top 3 by drop percentage (deepest drops)
    """)

    st.header("Strategy Parameters")
    params_df = pd.DataFrame({
        'Parameter': ['ROLLING_WINDOW', 'MIN_CONSISTENCY', 'TOP_CANDIDATES', 'MAX_POSITIONS', 'MIN_DROP_PCT'],
        'Value': ['378 days', '55%', '6', '3', '0.20%'],
        'Description': [
            'Days of history for E/C calculation (~1.5 years)',
            'Minimum bounce rate after down days',
            'Number of top scorers to consider',
            'Maximum simultaneous positions',
            'Minimum drop to trigger signal'
        ]
    })
    st.table(params_df)

    st.header("Universe Selection")
    st.markdown("""
    125 stocks from industries with structural ties to government and critical infrastructure:
    - **Defense & Aerospace** - Government contracts
    - **Financials** - Systemic infrastructure
    - **Semiconductors** - Critical supply chain
    - **Software & IT Services** - Enterprise/government
    - **Insurance & Credit** - Regulated, stable demand
    - **Industrial Machinery** - Infrastructure spending
    - **Building & Construction** - Housing, infrastructure
    """)

    st.header("Important Warnings")
    st.markdown("""
    **DO NOT:**
    - Modify strategy parameters without backtesting
    - Record signals at times other than 3:45 PM ET
    - Manually edit track_record.json
    - Execute orders other than MOC
    - Hold positions longer than 1 day
    """)

# =============================================================================
# MAIN APP
# =============================================================================

def app():
    with st.sidebar:
        page = st.radio("", ["Signals", "Technical Details"], label_visibility="collapsed")
        st.markdown("---")
        if st.button("Refresh", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if page == "Signals":
        main()
    else:
        show_technical_details()

if __name__ == "__main__":
    app()
