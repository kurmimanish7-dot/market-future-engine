import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Future Eye - Market Forecast Engine", layout="wide")
st.title("🔮 Future Eye: Quantitative Probability Engine")

# Sidebar Controls
st.sidebar.header("Asset Selection")
ticker_map = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS"
}
selected_label = st.sidebar.selectbox("Index / Stock Select Karein", list(ticker_map.keys()))
ticker = ticker_map[selected_label]

@st.cache_data(ttl=1800)
def load_and_process_data(symbol):
    df = yf.download(symbol, period="500d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)

    df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_10"] = df["Log_Ret"].rolling(10).std()
    df["Volatility_30"] = df["Log_Ret"].rolling(30).std()

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

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df.dropna(inplace=True)
    return df

try:
    data = load_and_process_data(ticker)
    last_close = float(data["Close"].iloc[-1])
    last_atr = float(data["ATR"].iloc[-1])

    # ML Probability
    features = ["Log_Ret", "Volatility_10", "Volatility_30", "ATR", "RSI"]
    X = data[features].iloc[:-1]
    y = data["Target"].iloc[:-1]
    latest_feat = data[features].iloc[-1:].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lat_s = scaler.transform(latest_feat)

    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X_s, y)
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42).fit(X_s, y)
    bullish_prob = float((rf.predict_proba(lat_s)[0][1] + gb.predict_proba(lat_s)[0][1]) / 2)

    # Monte Carlo (10,000 runs)
    daily_drift = float(data["Log_Ret"].mean())
    daily_vol = float(data["Log_Ret"].std())
    shocks = np.random.normal(daily_drift, daily_vol, 10000)
    sim_closes = last_close * np.exp(shocks)

    p10 = np.percentile(sim_closes, 10)
    p50 = np.percentile(sim_closes, 50)
    p90 = np.percentile(sim_closes, 90)

    # UI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Reference Close", f"{last_close:.2f}")
    col2.metric("Bullish Upside Probability", f"{bullish_prob * 100:.1f}%")
    col3.metric("Projected Median (P50)", f"{p50:.2f}")
    col4.metric("1-Sigma Expected Range", f"±{last_atr * 0.85:.1f} pts")

    st.divider()

    # Candlestick Chart with Probability Bands
    col_chart, col_levels = st.columns([2, 1])

    with col_chart:
        st.subheader("Recent Price Action & Projected Bands")
        recent_df = data.tail(60)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=recent_df.index,
            open=recent_df["Open"], high=recent_df["High"],
            low=recent_df["Low"], close=recent_df["Close"],
            name="Candles"
        ))
        fig.add_hline(y=p90, line_dash="dot", line_color="green", annotation_text="P90 Target")
        fig.add_hline(y=p10, line_dash="dot", line_color="red", annotation_text="P10 Support")
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_levels:
        st.subheader("Key Levels for Tomorrow")
        last_row = data.iloc[-1]
        rng = float(last_row["High"]) - float(last_row["Low"])
        
        st.write(f"🟢 **Breakout Buy (H4):** `{last_close + (rng * 1.1 / 2):.2f}`")
        st.write(f"🟡 **Resistance Zone (H3):** `{last_close + (rng * 1.1 / 4):.2f}`")
        st.write(f"⚪ **Central Baseline:** `{p50:.2f}`")
        st.write(f"🟡 **Support Zone (L3):** `{last_close - (rng * 1.1 / 4):.2f}`")
        st.write(f"🔴 **Breakdown Sell (L4):** `{last_close - (rng * 1.1 / 2):.2f}`")

except Exception as e:
    st.error(f"Data load karne me problem aayi: {e}")
