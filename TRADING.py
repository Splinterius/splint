import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

from prophet import Prophet
from prophet.plot import plot_plotly

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

import plotly.graph_objects as go

# ───────────────────────────────────────────────────────────────
# Streamlit page config
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Forecast + Live Feed", layout="wide")

# ── Sidebar controls ───────────────────────────────────────────
ticker = st.sidebar.text_input("Ticker symbol", "AAPL").upper()
period_days = st.sidebar.slider("History window (days)", 30, 730, 180)
look_back = st.sidebar.slider("GRU look‑back (hrs)", 12, 168, 24)
horizon = st.sidebar.slider("Forecast horizon (hrs)", 1, 48, 12)
model_choice = st.sidebar.selectbox("Forecast model", ["GRU", "Prophet"])
refresh_rate = st.sidebar.number_input("Auto‑refresh (sec)", 10, step=1, value=60)

# ── Auto‑refresh logic ─────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
elif time.time() - st.session_state.last_refresh > refresh_rate:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ── Cached data loaders ────────────────────────────────────────
@st.cache_data(ttl=refresh_rate - 1, show_spinner=False)
def load_hourly(tkr: str, days: int) -> pd.DataFrame:
    df = yf.download(
        tkr,
        period=f"{days}d",
        interval="1h",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Datetime"}, inplace=True)
    return df

@st.cache_data(ttl=30, show_spinner=False)
def load_live(tkr: str) -> pd.DataFrame:
    """1‑minute intraday for current day (live line chart)."""
    live = yf.download(
        tkr,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    live.reset_index(inplace=True)
    live.rename(columns={"index": "Datetime"}, inplace=True)
    return live

@st.cache_data(ttl=300, show_spinner=False)
def load_news(tkr: str, limit: int = 6):
    """Return top `limit` news items for ticker (via yfinance)."""
    try:
        items = yf.Ticker(tkr).news[:limit]
    except Exception:
        items = []
    return items

# ── Main data pulls ────────────────────────────────────────────
hourly_df = load_hourly(ticker, period_days)
live_df = load_live(ticker)
news_items = load_news(ticker)

st.metric("Latest close", f"${hourly_df['Close'].iloc[-1]:.2f}")

# ── TABS: Forecast | Live & News ───────────────────────────────
tab_forecast, tab_live = st.tabs(["📈 Forecast", "📰 Live & News"])

# ── ░░ FORECAST TAB ░░ ────────────────────────────────────────
with tab_forecast:
    st.subheader(f"{model_choice} forecast")

    # ── Prophet branch ─────────────────────────────────────────
    if model_choice == "Prophet":
        df_prophet = hourly_df[["Datetime", "Close"]].rename(
            columns={"Datetime": "ds", "Close": "y"}
        )
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)

        with st.spinner("Fitting Prophet…"):
            m = Prophet(daily_seasonality=True)
            m.fit(df_prophet)

        future = m.make_future_dataframe(periods=horizon, freq="H")
        forecast = m.predict(future)

        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Forecast tail"):
            st.dataframe(
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
            )

    # ── GRU branch ────────────────────────────────────────────
    else:
        st.caption("Training GRU (scaled Close)…")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(hourly_df["Close"].values.reshape(-1, 1))

        def make_supervised(data, window, horiz):
            X, y = [], []
            for i in range(len(data) - window - horiz + 1):
                X.append(data[i : i + window, 0])
                y.append(data[i + window : i + window + horiz, 0])
            return np.array(X), np.array(y)

        X, y = make_supervised(scaled_close, look_back, horizon)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = int(0.9 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        with st.spinner("Fitting GRU…"):
            gru = Sequential(
                [GRU(32, input_shape=(look_back, 1)), Dense(horizon)]
            )
            gru.compile(optimizer="adam", loss="mse")
            gru.fit(
                X_train,
                y_train,
                epochs=8,
                batch_size=32,
                verbose=0,
                validation_data=(X_test, y_test),
            )

        last_window = scaled_close[-look_back:].reshape(1, look_back, 1)
        pred_scaled = gru.predict(last_window)[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        future_times = pd.date_range(
            start=hourly_df["Datetime"].iloc[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq="H",
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hourly_df["Datetime"],
                y=hourly_df["Close"],
                name="Historical",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=pred,
                name="GRU forecast",
                mode="lines",
            )
        )
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# ── ░░ LIVE & NEWS TAB ░░ ─────────────────────────────────────
with tab_live:
    col1, col2 = st.columns([2, 1])

    # ── Live price chart ───────────────────────────────────────
    with col1:
        st.subheader("Live 1‑minute price")
        fig_live = go.Figure()
        fig_live.add_trace(
            go.Scatter(
                x=live_df["Datetime"],
                y=live_df["Close"],
                mode="lines",
                name="1‑min close",
            )
        )
        fig_live.update_layout(height=500, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig_live, use_container_width=True)

    # ── News headlines ────────────────────────────────────────
    with col2:
        st.subheader("Latest news")
        if not news_items:
            st.info("No news found (Yahoo may return empty).")
        else:
            for item in news_items:
                published = datetime.fromtimestamp(
                    item["providerPublishTime"], tz=timezone.utc
                ).strftime("%Y‑%m‑%d %H:%M UTC")
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"{item['publisher']} · {published}")
                st.markdown("---")

# ── Raw data preview toggle ───────────────────────────────────
with st.expander("Raw hourly data (last week)"):
    st.dataframe(hourly_df.tail(24 * 7))
