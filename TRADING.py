import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
import time

# ───────────────────────────────────────────────────────────────
# Streamlit page config
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Stock Projections", layout="wide")

# Sidebar controls
ticker = st.sidebar.text_input("Ticker symbol", "AAPL").upper()
period = st.sidebar.selectbox(
    "Historical window", ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)
refresh_rate = st.sidebar.number_input("Auto‑refresh (sec)", 5, step=1, value=30)

# ───────────────────────────────────────────────────────────────
# Data loading (cached)
# ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=refresh_rate - 1, show_spinner=False)
def load_data(tkr: str, prd: str) -> pd.DataFrame:
    df = yf.Ticker(tkr).history(period=prd, interval="1d")
    df.reset_index(inplace=True)
    return df

raw = load_data(ticker, period)

# Latest price metric
st.metric("Latest close", f"${raw['Close'].iloc[-1]:.2f}")

# ───────────────────────────────────────────────────────────────
# Prepare dataframe for Prophet
# ───────────────────────────────────────────────────────────────
train = raw.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]]
train["ds"] = pd.to_datetime(train["ds"]).dt.tz_localize(None)

# ───────────────────────────────────────────────────────────────
# Prophet model & forecast
# ───────────────────────────────────────────────────────────────
model = Prophet(daily_seasonality=True)
model.fit(train)

future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

# ───────────────────────────────────────────────────────────────
# Plot forecast
# ───────────────────────────────────────────────────────────────
fig = plot_plotly(model, forecast)
st.plotly_chart(fig, use_container_width=True)

# Optional data preview
with st.expander("Raw price data (tail)"):
    st.dataframe(raw.tail(20))

with st.expander("Forecast data (tail)"):
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon))

# ───────────────────────────────────────────────────────────────
# Auto-refresh after initial render
# ───────────────────────────────────────────────────────────────
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
elif time.time() - st.session_state.last_refresh > refresh_rate:
    st.session_state.last_refresh = time.time()
    st.rerun()  # ← this is now safe to call

