import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


class QuantitativeFutureEngine:
    def __init__(self, ticker: str = "^NSEI", lookback_days: int = 500):
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.df = None

    def fetch_data(self):
        """Historical OHLCV data fetch karta hai."""
        print(f"[*] Fetching historical data for {self.ticker}...")
        data = yf.download(self.ticker, period=f"{self.lookback_days}d", interval="1d", progress=False)
        
        if data.empty or len(data) < 100:
            raise ValueError("Insufficient data returned from provider.")
        
        # Flatten MultiIndex columns agar yfinance return kare
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data.dropna(inplace=True)
        self.df = data
        return self.df

    def engineer_features(self):
        """Technical factors, momentum, aur volatility signals create karta hai."""
        df = self.df.copy()

        # Returns & Volatility
        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility_10"] = df["Log_Ret"].rolling(window=10).std()
        df["Volatility_30"] = df["Log_Ret"].rolling(window=30).std()

        # ATR (Average True Range)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = true_range.rolling(14).mean()

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # Momentum & Moving Average Ratios
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["Distance_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
        df["Distance_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]

        # Target: Agla din positive (1) ya negative (0) close hoga
        df["Target_Direction"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

        df.dropna(inplace=True)
        self.df = df
        return self.df

    def ml_directional_probability(self):
        """Random Forest + Gradient Boosting Ensemble directional probability calculate karta hai."""
        feature_cols = [
            "Log_Ret", "Volatility_10", "Volatility_30", "ATR_14", 
            "RSI_14", "Distance_SMA20", "Distance_SMA50"
        ]
        
        X = self.df[feature_cols].iloc[:-1]  # Exclude last row (target unknown)
        y = self.df["Target_Direction"].iloc[:-1]

        latest_features = self.df[feature_cols].iloc[-1:].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        latest_scaled = scaler.transform(latest_features)

        # Dual Ensemble Models
        rf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)

        prob_rf = rf.predict_proba(latest_scaled)[0][1]
        prob_gb = gb.predict_proba(latest_scaled)[0][1]

        # Blended Probability
        ensemble_prob = (prob_rf * 0.5) + (prob_gb * 0.5)
        return float(ensemble_prob)

    def monte_carlo_simulation(self, simulations: int = 10000):
        """Geometric Brownian Motion (GBM) se agle din ke OHLC distribution generate karta hai."""
        last_close = float(self.df["Close"].iloc[-1])
        daily_drift = float(self.df["Log_Ret"].mean())
        daily_vol = float(self.df["Log_Ret"].std())

        # Random shock generations
        random_shocks = np.random.normal(daily_drift, daily_vol, simulations)
        simulated_closes = last_close * np.exp(random_shocks)

        # Simulating Intraday Wicks using historical High/Low dispersion
        avg_high_spread = ((self.df["High"] - self.df["Close"]) / self.df["Close"]).tail(30).mean()
        avg_low_spread = ((self.df["Close"] - self.df["Low"]) / self.df["Close"]).tail(30).mean()

        simulated_highs = simulated_closes * (1 + np.abs(np.random.normal(avg_high_spread, avg_high_spread * 0.4, simulations)))
        simulated_lows = simulated_closes * (1 - np.abs(np.random.normal(avg_low_spread, avg_low_spread * 0.4, simulations)))

        return {
            "P10_Bearish_Boundary": np.percentile(simulated_closes, 10),
            "P50_Expected_Close": np.percentile(simulated_closes, 50),
            "P90_Bullish_Boundary": np.percentile(simulated_closes, 90),
            "Probable_High": np.percentile(simulated_highs, 65),
            "Probable_Low": np.percentile(simulated_lows, 35)
        }

    def structural_pivot_levels(self):
        """Camarilla & Classic Pivots ke zariye critical intraday levels nikalta hai."""
        last_row = self.df.iloc[-1]
        H = float(last_row["High"])
        L = float(last_row["Low"])
        C = float(last_row["Close"])
        rng = H - L

        # Camarilla Breakout/Reversal Math
        h4 = C + (rng * 1.1 / 2.0)  # Long Breakout
        h3 = C + (rng * 1.1 / 4.0)  # Reversal Resistance
        l3 = C - (rng * 1.1 / 4.0)  # Reversal Support
        l4 = C - (rng * 1.1 / 2.0)  # Short Breakdown

        # Standard ATR Expected Move (1 Sigma)
        atr = float(self.df["ATR_14"].iloc[-1])
        expected_upper = C + (atr * 0.85)
        expected_lower = C - (atr * 0.85)

        return {
            "Camarilla_Long_Breakout_H4": h4,
            "Camarilla_Sell_Reversal_H3": h3,
            "Camarilla_Buy_Reversal_L3": l3,
            "Camarilla_Short_Breakdown_L4": l4,
            "ATR_Upper_1Sigma": expected_upper,
            "ATR_Lower_1Sigma": expected_lower
        }

    def run_prediction_pipeline(self):
        """Poore quantitative framework ko execute karta hai."""
        self.fetch_data()
        self.engineer_features()

        last_close = float(self.df["Close"].iloc[-1])
        last_date = self.df.index[-1].strftime("%Y-%m-%d")

        bullish_prob = self.ml_directional_probability()
        mc_results = self.monte_carlo_simulation()
        pivots = self.structural_pivot_levels()

        print("\n" + "="*60)
        print(f" QUANTITATIVE PROBABILITY MATRIX: {self.ticker}")
        print(f" Reference Base: Close={last_close:.2f} on {last_date}")
        print("="*60)

        # 1. Ensemble Probability
        bias = "STRONGLY BULLISH" if bullish_prob > 0.60 else (
            "BULLISH BIAS" if bullish_prob > 0.52 else (
                "BEARISH BIAS" if bullish_prob < 0.48 else "NEUTRAL / CHOPPY"
            )
        )
        print(f"\n[1] DIRECTIONAL FORECAST ENGINE")
        print(f"  • Probabilistic Upside: {bullish_prob * 100:.1f}%")
        print(f"  • Probabilistic Downside: {(1 - bullish_prob) * 100:.1f}%")
        print(f"  • Consensus Structural Bias: {bias}")

        # 2. Monte Carlo Distribution Table
        print(f"\n[2] MONTE CARLO DISTRIBUTION (10,000 Iterations)")
        mc_table = [
            ["P90 (Bullish Boundary / Optimistic)", f"{mc_results['P90_Bullish_Boundary']:.2f}"],
            ["Probable Session High", f"{mc_results['Probable_High']:.2f}"],
            ["P50 (Median Projected Close)", f"{mc_results['P50_Expected_Close']:.2f}"],
            ["Probable Session Low", f"{mc_results['Probable_Low']:.2f}"],
            ["P10 (Bearish Boundary / Pessimistic)", f"{mc_results['P10_Bearish_Boundary']:.2f}"]
        ]
        print(tabulate(mc_table, headers=["Trajectory Metric", "Price Point"], tablefmt="grid"))

        # 3. Actionable Boundary Triggers
        print(f"\n[3] INTRADAY PIVOT & EXECUTION CORRIDOR")
        pivot_table = [
            ["Breakout Trigger (H4)", f"{pivots['Camarilla_Long_Breakout_H4']:.2f}", "Trend Day: Go Long above this"],
            ["Upper Reversal Zone (H3)", f"{pivots['Camarilla_Sell_Reversal_H3']:.2f}", "Profit booking / Short zone"],
            ["1-Sigma ATR Upper Bound", f"{pivots['ATR_Upper_1Sigma']:.2f}", "Standard Statistical Ceiling"],
            ["Lower Reversal Zone (L3)", f"{pivots['Camarilla_Buy_Reversal_L3']:.2f}", "Dip Buying / Bounce zone"],
            ["1-Sigma ATR Lower Bound", f"{pivots['ATR_Lower_1Sigma']:.2f}", "Standard Statistical Floor"],
            ["Breakdown Trigger (L4)", f"{pivots['Camarilla_Short_Breakdown_L4']:.2f}", "Trend Day: Go Short below this"]
        ]
        print(tabulate(pivot_table, headers=["Level Type", "Price", "Actionable Context"], tablefmt="grid"))


if __name__ == "__main__":
    # Ticker selection:
    # Nifty 50: '^NSEI' | Bank Nifty: '^NSEBANK' | Reliance: 'RELIANCE.NS'
    selected_ticker = "^NSEI"
    if len(sys.argv) > 1:
        selected_ticker = sys.argv[1]

    engine = QuantitativeFutureEngine(ticker=selected_ticker)
    engine.run_prediction_pipeline()
