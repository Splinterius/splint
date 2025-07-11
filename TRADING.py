import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import plotly.graph_objects as go

# ─────────────────── Page config ──────────────────────────────
st.set_page_config(page_title="Stock Forecast + Live Feed", layout="wide")

# ─────────────────── Sidebar controls ─────────────────────────
ticker = st.sidebar.text_input("Ticker symbol", "AAPL").upper()
period_days = st.sidebar.slider("History window (days)", 30, 730, 180)
look_back = st.sidebar.slider("GRU look‑back (hrs)", 12, 168, 24)
horizon = st.sidebar.slider("Forecast horizon (hrs)", 1, 48, 12)
model_choice = st.sidebar.selectbox("Forecast model", ["GRU", "Prophet"])
refresh_rate = st.sidebar.number_input("Auto‑refresh (sec)", 10, step=1, value=60)

# ─────────────────── Auto‑refresh ─────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
elif time.time() - st.session_state.last_refresh > refresh_rate:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ─────────────────── Data loaders ─────────────────────────────
@st.cache_data(ttl=refresh_rate - 1)
def load_hourly(tkr: str, days: int) -> pd.DataFrame:
    df = yf.download(tkr, period=f"{days}d", interval="1h", auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    if 'Datetime' not in df.columns:
        df.rename(columns={"index": "Datetime"}, inplace=True)
    return df

@st.cache_data(ttl=30)
def load_live(tkr: str):
    live = yf.download(tkr, period="1d", interval="1m", auto_adjust=True, progress=False)
    live.reset_index(inplace=True)
    if 'Datetime' not in live.columns:
        live.rename(columns={"index": "Datetime"}, inplace=True)
    return live

@st.cache_data(ttl=300)
def load_news(tkr: str, limit: int = 6):
    try:
        return yf.Ticker(tkr).news[:limit]
    except Exception:
        return []

# ─────────────────── Pull and validate data ───────────────────
hourly_df = load_hourly(ticker, period_days)

# SAFETY CHECK
if not isinstance(hourly_df, pd.DataFrame):
    st.error(f"⚠️ Failed to load data for “{ticker}”.")
    st.stop()

if hourly_df.empty:
    st.error(f"⚠️ No data found for “{ticker}”. Try another symbol or a different time range.")
    st.stop()

if "Close" not in hourly_df.columns:
    st.error(f"⚠️ Missing 'Close' column in the data for “{ticker}”.")
    st.stop()

clean_close = hourly_df["Close"].dropna()
if clean_close.empty:
    st.error(f"⚠️ All 'Close' values are missing for “{ticker}”. Try a different time range or stock.")
    st.stop()

# Safe to use 'Close' now
latest_close = clean_close.iloc[-1]
st.metric("Latest close", f"${latest_close:.2f}")

live_df = load_live(ticker)
news_items = load_news(ticker)

# ─────────────────── Tabs (Forecast | Live) ───────────────────
tab_forecast, tab_live = st.tabs(["📈 Forecast", "📰 Live & News"])

# ░░ FORECAST TAB ░░
with tab_forecast:
    if model_choice == "Prophet":
        df_prophet = hourly_df[["Datetime", "Close"]].rename(columns={"Datetime": "ds", "Close": "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)
        with st.spinner("Fitting Prophet…"):
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
        future = model.make_future_dataframe(periods=horizon, freq="H")
        forecast = model.predict(future)
        st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)
        with st.expander("Forecast tail"):
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))
    else:
        # GRU model
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(hourly_df["Close"].values.reshape(-1, 1))

        def make_xy(data, win, horiz):
            X, y = [], []
            for i in range(len(data) - win - horiz + 1):
                X.append(data[i : i + win, 0])
                y.append(data[i + win : i + win + horiz, 0])
            return np.array(X), np.array(y)

        X, y = make_xy(scaled, look_back, horizon)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        split = int(0.9 * len(X))
        Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

        with st.spinner("Training GRU…"):
            gru = Sequential([GRU(32, input_shape=(look_back, 1)), Dense(horizon)])
            gru.compile(optimizer="adam", loss="mse")
            gru.fit(Xtr, ytr, epochs=8, batch_size=32, verbose=0)

        last_window = scaled[-look_back:].reshape(1, look_back, 1)
        pred_scaled = gru.predict(last_window)[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        future_times = pd.date_range(start=hourly_df["Datetime"].iloc[-1] + pd.Timedelta(hours=1), periods=horizon, freq="H")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_df["Datetime"], y=hourly_df["Close"], name="Historical"))
        fig.add_trace(go.Scatter(x=future_times, y=pred, name="GRU Forecast"))
        st.plotly_chart(fig, use_container_width=True)

# ░░ LIVE & NEWS TAB ░░
with tab_live:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live 1‑minute price")
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(x=live_df["Datetime"], y=live_df["Close"], mode="lines"))
        fig_live.update_layout(height=500)
        st.plotly_chart(fig_live, use_container_width=True)

    with col2:
        st.subheader("Latest news")
        if not news_items:
            st.info("No news returned by Yahoo.")
        else:
            for item in news_items:
                timestamp = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc).strftime("%Y‑%m‑%d %H:%M UTC")
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"{item['publisher']} · {timestamp}")
                st.divider()

# Optional raw preview
with st.expander("Raw hourly data (tail)"):
    st.dataframe(hourly_df.tail(24 * 7))
