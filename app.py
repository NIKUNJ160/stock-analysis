import streamlit as st
import subprocess
import os
import time
import json
import pandas as pd
import glob

from src.utils.helpers import safe_read_json

st.set_page_config(page_title="Quant Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📈 Live Quantitative Trading Dashboard")
st.markdown("Monitor and control your AI-powered trading algorithms.")

# ═══════════ Sidebar ═══════════
st.sidebar.header("⚙️ Architecture Engine")

if "engine_process" not in st.session_state:
    st.session_state.engine_process = None

def start_engine():
    if st.session_state.engine_process is None or st.session_state.engine_process.poll() is not None:
        venv_python = os.path.join("venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join("venv", "bin", "python")
        if not os.path.exists(venv_python):
            venv_python = "python"
        os.makedirs("data", exist_ok=True)
        log_file = open(os.path.join("data", "engine_logs.txt"), "w", encoding="utf-8")
        st.session_state.log_file = log_file
        st.session_state.engine_process = subprocess.Popen(
            [venv_python, "-u", "main.py"],
            stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        st.sidebar.success("Engine started!")

def stop_engine():
    if st.session_state.engine_process is not None and st.session_state.engine_process.poll() is None:
        st.session_state.engine_process.terminate()
        st.session_state.engine_process = None
        if "log_file" in st.session_state and st.session_state.log_file:
            st.session_state.log_file.close()
        st.sidebar.warning("Engine stopped!")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start", type="primary"):
        start_engine()
with col2:
    if st.button("⏹️ Stop"):
        stop_engine()

st.sidebar.divider()
st.sidebar.subheader("System Status")
if st.session_state.engine_process is None or st.session_state.engine_process.poll() is not None:
    st.sidebar.error("Engine OFFLINE")
else:
    st.sidebar.success("Engine RUNNING")

# Check auto-refresh
if st.session_state.engine_process is not None and st.session_state.engine_process.poll() is None:
    time.sleep(1)

# ═══════════ Tabs ═══════════
tab_signals, tab_portfolio, tab_backtest, tab_chart, tab_logs = st.tabs(
    ["⚡ Signals", "💰 Portfolio", "📊 Backtest", "📈 Market Data", "📝 Logs"]
)

# ═══════════ Tab 1: Signals ═══════════
with tab_signals:
    st.header("⚡ Real-Time ML Signals")
    
    signals = safe_read_json(os.path.join("data", "latest_signals.json"))
    
    if signals:
        cols = st.columns(len(signals))
        for i, (symbol, data) in enumerate(signals.items()):
            with cols[i]:
                strength = data.get("strength", "HOLD")
                if "BUY" in strength:
                    emoji = "🟢"
                    delta_color = "normal"
                elif "SELL" in strength:
                    emoji = "🔴"
                    delta_color = "inverse"
                else:
                    emoji = "⚪"
                    delta_color = "off"
                
                st.metric(
                    label=f"{symbol}",
                    value=f"{emoji} {strength}",
                    delta=f"Conf: {data.get('confidence', 'N/A')}",
                    delta_color=delta_color
                )
                
                # Details expander
                with st.expander("Details"):
                    st.write(f"**Model Conf:** {data.get('model_confidence', 'N/A')}")
                    st.write(f"**RSI:** {data.get('rsi', 'N/A')}")
                    st.write(f"**Trend:** {data.get('trend', 'N/A')}")
                    st.write(f"**Filter:** {'✅' if data.get('filter_passed') else '❌'}")
                    
                    if data.get('risk_approved') is not None:
                        st.write(f"**Risk:** {'✅ Approved' if data.get('risk_approved') else '❌ Rejected'}")
                        st.write(f"**Position Size:** {data.get('position_size', 'N/A')}")
                        st.write(f"**SL:** {data.get('stop_loss', 'N/A')}")
                        st.write(f"**TP:** {data.get('take_profit', 'N/A')}")
                    
                    reasons = data.get('reasons', [])
                    if reasons:
                        st.write("**Reasons:**")
                        for r in reasons:
                            st.write(f"  • {r}")
                
                st.caption(f"Updated: {data.get('timestamp', 'N/A')}")
    else:
        st.info("No signals yet. Start the Engine to begin predicting.")

# ═══════════ Tab 2: Portfolio ═══════════
with tab_portfolio:
    st.header("💰 Portfolio & Risk Dashboard")
    
    portfolio = safe_read_json(os.path.join("data", "portfolio.json"))
    
    if portfolio and 'open_positions' in portfolio:
        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open Positions", f"{portfolio.get('open_positions', 0)}/{portfolio.get('max_positions', 5)}")
        c2.metric("Daily P&L", f"₹{portfolio.get('daily_pnl', 0):,.2f}")
        c3.metric("Total P&L", f"₹{portfolio.get('total_pnl', 0):,.2f}")
        c4.metric("Exposure", f"{portfolio.get('exposure_pct', 0):.1%}")
        
        # Positions table
        positions = portfolio.get('positions', {})
        if positions:
            st.subheader("Open Positions")
            pos_data = []
            for sym, pos in positions.items():
                pos_data.append({
                    "Symbol": sym,
                    "Side": pos.get('side', ''),
                    "Entry": f"₹{pos.get('entry', 0):,.2f}",
                    "Qty": pos.get('qty', 0),
                    "SL": f"₹{pos.get('sl', 0):,.2f}",
                    "TP": f"₹{pos.get('tp', 0):,.2f}",
                    "P&L": f"₹{pos.get('pnl', 0):,.2f}",
                })
            st.table(pd.DataFrame(pos_data))
        else:
            st.info("No open positions.")
    else:
        st.info("No portfolio data. Start the Engine.")

# ═══════════ Tab 3: Backtest ═══════════
with tab_backtest:
    st.header("📊 Backtest Results")
    
    bt_results = safe_read_json(os.path.join("data", "backtest_results", "backtest_summary.json"))
    
    if bt_results:
        for symbol, metrics in bt_results.items():
            with st.expander(f"📊 {symbol}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
                c2.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
                c3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                c4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Total Trades", metrics.get('total_trades', 0))
                c6.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                c7.metric("Avg Win", f"₹{metrics.get('avg_win', 0):,.0f}")
                c8.metric("Avg Loss", f"₹{metrics.get('avg_loss', 0):,.0f}")
    else:
        st.info("No backtest results. Run `python -m src.backtesting.backtester`")

# ═══════════ Tab 4: Market Data ═══════════
with tab_chart:
    st.header("📈 Market Data")
    
    data_dir = os.path.join("data", "processed")
    if os.path.exists(data_dir):
        csv_files = glob.glob(os.path.join(data_dir, "*_cleaned.csv"))
        if csv_files:
            symbols_map = {os.path.basename(f).split('_')[0]: f for f in csv_files}
            selected = st.selectbox("Select Asset", list(symbols_map.keys()))
            
            file_path = symbols_map.get(selected)
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not df.empty:
                    st.line_chart(df['close'].tail(150))
                    with st.expander("Volume"):
                        st.bar_chart(df['volume'].tail(150))
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("No processed data available.")

# ═══════════ Tab 5: Logs ═══════════
with tab_logs:
    st.header("📝 Engine Logs")
    
    log_path = os.path.join("data", "engine_logs.txt")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                logs = f.readlines()
            if logs:
                st.code("".join(logs[-30:]), language="bash")
            else:
                st.info("Logs empty...")
        except Exception as e:
            st.error(f"Could not read logs: {e}")
    else:
        st.info("No logs. Start the Engine.")

# Auto-refresh
if st.session_state.engine_process is not None and st.session_state.engine_process.poll() is None:
    st.rerun()
