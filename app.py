import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Future Eye Pro - AI Institutional Engine", layout="wide")

# Custom Styling for Traffic Lights & Cards
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #2a2e39;
    }
    .signal-green {
        background-color: #0d3820;
        color: #26a69a;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        border: 2px solid #26a69a;
    }
    .signal-yellow {
        background-color: #3b3207;
        color: #ffd166;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        border: 2px solid #ffd166;
    }
    .signal-red {
        background-color: #3d1419;
        color: #ef5350;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        border: 2px solid #ef5350;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Future Eye Pro: Institutional Multi-Factor Engine")

# ==================== SIDEBAR & 3000+ STOCK SELECTOR ====================
st.sidebar.header("Asset Selection")

popular_stocks = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "ITC": "ITC.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "L&T": "LT.NS",
    "ZOMATO": "ZOMATO.NS"
}

dropdown_choice = st.sidebar.selectbox("Popular Index / Stock Select Karein", list(popular_stocks.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Search Any Listed Share (3000+)")
custom_ticker_input = st.sidebar.text_input("NSE Symbol Type Karein (e.g. SUZLON, IRFC, BEL):", "").strip().upper()

if custom_ticker_input:
    selected_ticker = f"{custom_ticker_input}.NS"
    display_name = custom_ticker_input
else:
    selected_ticker = popular_stocks[dropdown_choice]
    display_name = dropdown_choice

# ==================== GLOBAL INTERMARKET ENGINE ====================
@st.cache_data(ttl=900)
def fetch_global_macro():
    """US Indices, Crude Oil, DXY, and Bond Yields fetch karta hai."""
    macro_tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Brent Crude": "BZ=F",
        "US Dollar Index (DXY)": "DX-Y.NYB"
    }
    macro_data = {}
    macro_changes = {}

    for name, sym in macro_tickers.items():
        try:
            hist = yf.download(sym, period="5d", interval="1d", progress=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [col[0] for col in hist.columns]
            if len(hist) >= 2:
                last = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                chg = ((last - prev) / prev) * 100
                macro_data[name] = last
                macro_changes[name] = chg
        except Exception:
            macro_data[name] = 0.0
            macro_changes[name] = 0.0

    return macro_data, macro_changes

# ==================== ASSET DATA & TECHNICALS ====================
@st.cache_data(ttl=900)
def fetch_asset_data(symbol):
    df = yf.download(symbol, period="300d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if df.empty or len(df) < 50:
        return None
    df.dropna(inplace=True)

    # Returns & Volatility
    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # ATR
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # RSI
    diff = df["Close"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # Moving Averages
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # Volume Z-Score (Institutional Flow Proxy)
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_SMA20"] + 1e-9)

    df.dropna(inplace=True)
    return df

# Run Pipelines
macro_vals, macro_chgs = fetch_global_macro()
data = fetch_asset_data(selected_ticker)

# ==================== MACRO STRIP ====================
st.subheader("🌐 Global Intermarket Macro Pulse")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

sp_chg = macro_chgs.get("S&P 500", 0.0)
nas_chg = macro_chgs.get("Nasdaq", 0.0)
crude_chg = macro_chgs.get("Brent Crude", 0.0)
dxy_chg = macro_chgs.get("US Dollar Index (DXY)", 0.0)

m_col1.metric("S&P 500", f"{macro_vals.get('S&P 500', 0):.1f}", f"{sp_chg:+.2f}%")
m_col2.metric("Nasdaq", f"{macro_vals.get('Nasdaq', 0):.1f}", f"{nas_chg:+.2f}%")
m_col3.metric("Brent Crude", f"${macro_vals.get('Brent Crude', 0):.2f}", f"{crude_chg:+.2f}%", delta_color="inverse")
m_col4.metric("US Dollar (DXY)", f"{macro_vals.get('US Dollar Index (DXY)', 0):.2f}", f"{dxy_chg:+.2f}%", delta_color="inverse")

# Calculate Macro Bias Score (Scale: -30 to +30)
macro_score = 0
if sp_chg > 0.3: macro_score += 10
elif sp_chg < -0.3: macro_score -= 10

if nas_chg > 0.5: macro_score += 8
elif nas_chg < -0.5: macro_score -= 8

# Falling crude & falling dollar are bullish for India
if crude_chg < -0.5: macro_score += 6
elif crude_chg > 0.5: macro_score -= 6

if dxy_chg < -0.2: macro_score += 6
elif dxy_chg > 0.2: macro_score -= 6

st.divider()

if data is None:
    st.error(f"'{selected_ticker}' ka data load nahi ho saka. Kripya check karein ki symbol sahi hai (jaise TATAMOTORS, HDFCBANK, INFY).")
else:
    last_close = float(data["Close"].iloc[-1])
    last_rsi = float(data["RSI"].iloc[-1])
    sma20 = float(data["SMA_20"].iloc[-1])
    sma50 = float(data["SMA_50"].iloc[-1])
    vol_ratio = float(data["Vol_Ratio"].iloc[-1])
    last_atr = float(data["ATR"].iloc[-1])

    # ==================== QUANTITATIVE SCORING (0 to 100) ====================
    # 1. Technical Score (Base 50)
    tech_score = 50
    if last_close > sma20: tech_score += 10
    else: tech_score -= 10
    
    if sma20 > sma50: tech_score += 8
    else: tech_score -= 8

    if 45 <= last_rsi <= 65: tech_score += 7  # Healthy bullish zone
    elif last_rsi > 75: tech_score -= 5       # Overbought exhaustion
    elif last_rsi < 35: tech_score -= 10      # Deep bearish breakdown

    # 2. Institutional Volume Proxy Score
    if vol_ratio > 1.5 and last_close > data["Open"].iloc[-1]:
        tech_score += 10 # Strong accumulation
    elif vol_ratio > 1.5 and last_close < data["Open"].iloc[-1]:
        tech_score -= 10 # Heavy distribution/dump

    # Final Composite Probability Score
    composite_score = np.clip(tech_score + macro_score, 5, 95)

    # Monte Carlo simulation for tomorrow
    daily_drift = float(data["Log_Ret"].mean())
    daily_vol = float(data["Log_Ret"].std())
    shocks = np.random.normal(daily_drift, daily_vol, 8000)
    sim_closes = last_close * np.exp(shocks)
    p10 = np.percentile(sim_closes, 10)
    p50 = np.percentile(sim_closes, 50)
    p90 = np.percentile(sim_closes, 90)

    # ==================== TRAFFIC SIGNAL & CORE VERDICT ====================
    sig_col, stat1, stat2, stat3 = st.columns([1.5, 1, 1, 1])

    with sig_col:
        if composite_score >= 63:
            st.markdown('<div class="signal-green">🟢 STRONG BUY SIGNAL</div>', unsafe_allow_html=True)
            trade_bias = "BULLISH ACCUMULATION"
        elif composite_score <= 38:
            st.markdown('<div class="signal-red">🔴 STRONG SELL / CAUTION</div>', unsafe_allow_html=True)
            trade_bias = "BEARISH DISTRIBUTION"
        else:
            st.markdown('<div class="signal-yellow">🟡 WAIT / CHOPPY ZONE</div>', unsafe_allow_html=True)
            trade_bias = "NEUTRAL / SIDEWAYS CONSOLIDATION"

    stat1.metric("Asset Price", f"₹{last_close:.2f}")
    stat2.metric("AI Probability Score", f"{composite_score:.1f} / 100")
    stat3.metric("Expected Move (1-Day)", f"±₹{last_atr * 0.85:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================== REASON TO TRADE (EXPLAINABLE AI) ====================
    st.subheader(f"📑 Trade Kyun Karein? (Detailed Trade Thesis for {display_name})")
    
    macro_verdict_text = "Favourable (US positive, Crude/DXY stable)" if macro_score >= 5 else (
        "Adverse (High Crude/Dollar rally putting pressure on INR assets)" if macro_score <= -5 else "Neutral Global Setup"
    )

    volume_verdict_text = f"High Institutional Interest ({vol_ratio:.2f}x of 20-day average)" if vol_ratio > 1.2 else "Normal Retail Volume"

    st.markdown(f"""
    * **1. Global Intermarket Alignment:** {macro_verdict_text}. US markets aur commodity prices is trade ke liye {'anukool (supportive)' if macro_score >= 0 else 'pratikool (headwind)'} hain.
    * **2. Institutional Flow & Volume Signature:** {volume_verdict_text}. Price action ke sath volume {'accumulation' if last_close > sma20 else 'distribution'} ki taraf ishara kar raha hai.
    * **3. Trend & Momentum Structure:** RSI **{last_rsi:.1f}** par hai aur asset apne 20-day SMA (₹{sma20:.2f}) ke **{'UPAR' if last_close > sma20 else 'NICHE'}** trade kar raha hai, jo structure ko {'mazboot' if last_close > sma20 else 'kamzor'} banata hai.
    * **4. Risk-to-Reward Execution Matrix:**
        * **Recommended Entry Zone:** ₹{last_close:.2f} ke aas-paas
        * **Invalidation / Stop-Loss Level:** ₹{last_close - (last_atr * 1.0):.2f} (Strict volatility-adjusted SL)
        * **Target 1 (Base Statistical Move):** ₹{p50:.2f}
        * **Target 2 (Extended Trend Target):** ₹{p90 if composite_score >= 50 else p10:.2f}
    """)

    st.divider()

    # ==================== VISUAL CHART & LEVELS ====================
    c_left, c_right = st.columns([2, 1])

    with c_left:
        st.subheader("Price Structure & Statistical Corridor")
        recent = data.tail(45)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=recent.index,
            open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"],
            name="Candles"
        ))
        fig.add_hline(y=p90, line_dash="dash", line_color="#26a69a", annotation_text=f"P90 Target: ₹{p90:.2f}")
        fig.add_hline(y=p10, line_dash="dash", line_color="#ef5350", annotation_text=f"P10 Support: ₹{p10:.2f}")
        fig.update_layout(xaxis_rangeslider_visible=False, height=420, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("Intraday Pivot Roadmap")
        rng = float(data.iloc[-1]["High"]) - float(data.iloc[-1]["Low"])
        h4 = last_close + (rng * 1.1 / 2.0)
        h3 = last_close + (rng * 1.1 / 4.0)
        l3 = last_close - (rng * 1.1 / 4.0)
        l4 = last_close - (rng * 1.1 / 2.0)

        st.info(f"**Long Breakout (H4):** ₹{h4:.2f}\n*(Iske upar sustain kare toh big momentum)*")
        st.warning(f"**Supply Resistance (H3):** ₹{h3:.2f}")
        st.warning(f"**Demand Support (L3):** ₹{l3:.2f}")
        st.error(f"**Short Breakdown (L4):** ₹{l4:.2f}\n*(Iske niche slide kare toh fast fall)*")
