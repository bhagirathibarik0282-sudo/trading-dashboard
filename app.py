"""
Trading Dashboard
------------------
A Streamlit dashboard that shows live positions, holdings, orders and P&L
from either Zerodha (Kite Connect) or Dhan (DhanHQ). If no broker
credentials are configured it automatically falls back to a DEMO mode with
simulated data, so the app is runnable out of the box.

Configure credentials via environment variables (see .env.example) or
Streamlit secrets (.streamlit/secrets.toml):

    KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN
    DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_secret(name: str) -> str:
    """Read a credential from env vars first, then Streamlit secrets (if a
    secrets.toml happens to exist). Env vars are the primary path here since
    credentials are supplied as Railway variables, not a secrets file."""
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------
# Broker adapters - each exposes the same interface:
#   get_positions() -> DataFrame
#   get_holdings()  -> DataFrame
#   get_orders()    -> DataFrame
#   get_margins()   -> dict
# --------------------------------------------------------------------------
class KiteBroker:
    name = "Zerodha (Kite Connect)"

    def __init__(self, api_key, access_token):
        from kiteconnect import KiteConnect

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

    def get_positions(self) -> pd.DataFrame:
        data = self.kite.positions().get("net", [])
        return pd.DataFrame(data)

    def get_holdings(self) -> pd.DataFrame:
        return pd.DataFrame(self.kite.holdings())

    def get_orders(self) -> pd.DataFrame:
        return pd.DataFrame(self.kite.orders())

    def get_margins(self) -> dict:
        return self.kite.margins()


class DhanBroker:
    name = "Dhan (DhanHQ)"

    def __init__(self, client_id, access_token):
        from dhanhq import dhanhq

        self.dhan = dhanhq(client_id, access_token)

    def get_positions(self) -> pd.DataFrame:
        resp = self.dhan.get_positions()
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        return pd.DataFrame(data)

    def get_holdings(self) -> pd.DataFrame:
        resp = self.dhan.get_holdings()
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        return pd.DataFrame(data)

    def get_orders(self) -> pd.DataFrame:
        resp = self.dhan.get_order_list()
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        return pd.DataFrame(data)

    def get_margins(self) -> dict:
        resp = self.dhan.get_fund_limits()
        return resp.get("data", resp) if isinstance(resp, dict) else resp


class DemoBroker:
    """Simulated broker so the dashboard works with zero configuration."""

    name = "Demo (simulated data)"

    _SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "ITC", "LT"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_positions(self) -> pd.DataFrame:
        rows = []
        for sym in self.rng.choice(self._SYMBOLS, size=5, replace=False):
            qty = int(self.rng.integers(1, 50))
            avg = round(float(self.rng.uniform(200, 3500)), 2)
            ltp = round(avg * (1 + self.rng.uniform(-0.06, 0.06)), 2)
            rows.append(
                {
                    "tradingsymbol": sym,
                    "quantity": qty,
                    "average_price": avg,
                    "last_price": ltp,
                    "pnl": round((ltp - avg) * qty, 2),
                }
            )
        return pd.DataFrame(rows)

    def get_holdings(self) -> pd.DataFrame:
        rows = []
        for sym in self._SYMBOLS:
            qty = int(self.rng.integers(5, 100))
            avg = round(float(self.rng.uniform(150, 4000)), 2)
            ltp = round(avg * (1 + self.rng.uniform(-0.15, 0.20)), 2)
            rows.append(
                {
                    "tradingsymbol": sym,
                    "quantity": qty,
                    "average_price": avg,
                    "last_price": ltp,
                    "pnl": round((ltp - avg) * qty, 2),
                    "value": round(ltp * qty, 2),
                }
            )
        return pd.DataFrame(rows)

    def get_orders(self) -> pd.DataFrame:
        rows = []
        statuses = ["COMPLETE", "COMPLETE", "COMPLETE", "OPEN", "CANCELLED", "REJECTED"]
        now = datetime.now()
        for i in range(12):
            sym = self.rng.choice(self._SYMBOLS)
            rows.append(
                {
                    "order_id": f"DEMO{1000 + i}",
                    "tradingsymbol": sym,
                    "transaction_type": self.rng.choice(["BUY", "SELL"]),
                    "quantity": int(self.rng.integers(1, 30)),
                    "price": round(float(self.rng.uniform(150, 3500)), 2),
                    "status": self.rng.choice(statuses),
                    "order_timestamp": (now - timedelta(minutes=int(self.rng.integers(1, 600)))),
                }
            )
        df = pd.DataFrame(rows).sort_values("order_timestamp", ascending=False)
        return df.reset_index(drop=True)

    def get_margins(self) -> dict:
        available = round(float(self.rng.uniform(50000, 500000)), 2)
        used = round(available * float(self.rng.uniform(0.1, 0.6)), 2)
        return {
            "equity": {
                "available": {"cash": available, "live_balance": available - used},
                "utilised": {"debits": used},
            }
        }

    def get_equity_curve(self, days: int = 90) -> pd.DataFrame:
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        steps = self.rng.normal(loc=0.0015, scale=0.012, size=days)
        curve = 100000 * np.cumprod(1 + steps)
        return pd.DataFrame({"date": dates, "equity": curve})


# --------------------------------------------------------------------------
# Sidebar - broker selection & connection status
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Broker Settings")

kite_key = get_secret("KITE_API_KEY")
kite_secret = get_secret("KITE_API_SECRET")
kite_token_env = get_secret("KITE_ACCESS_TOKEN")
dhan_client = get_secret("DHAN_CLIENT_ID")
dhan_token = get_secret("DHAN_ACCESS_TOKEN")

# --------------------------------------------------------------------------
# Kite Connect login flow
# --------------------------------------------------------------------------
# Zerodha access tokens expire daily. If a `request_token` shows up in the
# URL (Zerodha redirects back here after login), exchange it for a fresh
# access token and keep it in this browser session's state. Otherwise fall
# back to KITE_ACCESS_TOKEN from the environment (useful right after you've
# generated one manually and set it as a Railway variable).
if "kite_access_token" not in st.session_state:
    st.session_state.kite_access_token = None

request_token = st.query_params.get("request_token")
if request_token and kite_key and kite_secret and not st.session_state.kite_access_token:
    try:
        from kiteconnect import KiteConnect

        _kite_login = KiteConnect(api_key=kite_key)
        session_data = _kite_login.generate_session(request_token, api_secret=kite_secret)
        st.session_state.kite_access_token = session_data["access_token"]
        st.query_params.clear()
        st.sidebar.success("Kite login successful — access token refreshed for this session.")
    except Exception as e:
        st.sidebar.error(f"Kite login failed: {e}")

kite_token = st.session_state.kite_access_token or kite_token_env

available_brokers = ["Demo"]
if kite_key and kite_token:
    available_brokers.append("Zerodha (Kite Connect)")
if dhan_client and dhan_token:
    available_brokers.append("Dhan (DhanHQ)")

broker_choice = st.sidebar.selectbox("Active broker", available_brokers, index=len(available_brokers) - 1)

if broker_choice == "Zerodha (Kite Connect)":
    st.sidebar.success("Connected via Kite Connect credentials")
elif broker_choice == "Dhan (DhanHQ)":
    st.sidebar.success("Connected via Dhan credentials")
else:
    st.sidebar.info("No broker credentials found — showing simulated demo data.\nSee .env.example to connect a real account.")

# Always show a Kite login link when we at least have an API key, since the
# token expires every day and needs refreshing regardless of which broker
# is currently active.
if kite_key and kite_secret:
    from kiteconnect import KiteConnect

    _login_url = KiteConnect(api_key=kite_key).login_url()
    st.sidebar.markdown(f"[🔑 Login with Kite (refresh today's token)]({_login_url})")
elif kite_key and not kite_secret:
    st.sidebar.caption("Set KITE_API_SECRET to enable one-click Kite login/token-refresh.")

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def get_broker(choice: str):
    # NOTE: deliberately NOT cached with st.cache_resource. That cache is
    # keyed only on `choice` (a fixed string), so if kite_token/dhan_token
    # aren't part of the key, a stale broker built with an old/expired
    # token would be reused forever after a fresh login — showing data
    # that doesn't match the real account. Constructing these client
    # objects is cheap and makes no network call, so there's no need to
    # cache it; the cost we actually care about (the API calls) is cached
    # separately below, keyed correctly.
    if choice == "Zerodha (Kite Connect)":
        return KiteBroker(kite_key, kite_token)
    if choice == "Dhan (DhanHQ)":
        return DhanBroker(dhan_client, dhan_token)
    return DemoBroker()


try:
    broker = get_broker(broker_choice)
except Exception as e:
    st.sidebar.error(f"Failed to connect: {e}")
    broker = DemoBroker()


@st.cache_data(ttl=30, show_spinner=False)
def load_data(broker_name: str, token_fingerprint: str, _broker):
    # `_broker` (leading underscore) is excluded from Streamlit's cache key,
    # so broker_name + token_fingerprint are passed explicitly to make sure
    # the cache actually invalidates when the active broker or its token
    # changes (e.g. right after a fresh Kite login). Without this, stale
    # data from a previous broker/token can keep being served.
    try:
        positions = _broker.get_positions()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch positions: {e}")
        positions = pd.DataFrame()
    try:
        holdings = _broker.get_holdings()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch holdings: {e}")
        holdings = pd.DataFrame()
    try:
        orders = _broker.get_orders()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch orders: {e}")
        orders = pd.DataFrame()
    try:
        margins = _broker.get_margins()
    except Exception:
        margins = {}
    return positions, holdings, orders, margins


_token_fingerprint = kite_token if broker_choice == "Zerodha (Kite Connect)" else (
    dhan_token if broker_choice == "Dhan (DhanHQ)" else "demo"
)
positions_df, holdings_df, orders_df, margins = load_data(broker_choice, _token_fingerprint, broker)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📈 Trading Dashboard")
st.caption(f"Data source: **{broker.name}**")

total_pnl = 0.0
if not positions_df.empty and "pnl" in positions_df.columns:
    total_pnl += positions_df["pnl"].sum()
if not holdings_df.empty and "pnl" in holdings_df.columns:
    total_pnl += holdings_df["pnl"].sum()

holdings_value = holdings_df["value"].sum() if not holdings_df.empty and "value" in holdings_df.columns else 0.0

available_cash = None
if isinstance(margins, dict):
    try:
        available_cash = margins.get("equity", {}).get("available", {}).get("cash")
    except Exception:
        available_cash = None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
col2.metric("Holdings Value", f"₹{holdings_value:,.2f}")
col3.metric("Open Positions", f"{len(positions_df)}")
col4.metric("Available Margin", f"₹{available_cash:,.2f}" if available_cash is not None else "—")

st.markdown("---")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_positions, tab_holdings, tab_orders = st.tabs(
    ["Overview", "Positions", "Holdings", "Orders"]
)

with tab_overview:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Equity Curve")
        if isinstance(broker, DemoBroker):
            curve = broker.get_equity_curve()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=curve["date"], y=curve["equity"], mode="lines", fill="tozeroy", name="Equity"
                )
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
                yaxis_title="Portfolio Value (₹)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Equity curve history requires broker trade-book data — not shown for live accounts in this view.")

    with right:
        st.subheader("Allocation")
        if not holdings_df.empty and "value" in holdings_df.columns:
            fig = px.pie(
                holdings_df, names="tradingsymbol", values="value", hole=0.45
            )
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No holdings to display.")

    st.subheader("P&L by Symbol")
    pnl_source = pd.concat(
        [df for df in [positions_df, holdings_df] if not df.empty and "pnl" in df.columns],
        ignore_index=True,
    ) if (not positions_df.empty or not holdings_df.empty) else pd.DataFrame()

    if not pnl_source.empty:
        grouped = pnl_source.groupby("tradingsymbol", as_index=False)["pnl"].sum()
        colors = np.where(grouped["pnl"] >= 0, "#2ca02c", "#d62728")
        fig = go.Figure(go.Bar(x=grouped["tradingsymbol"], y=grouped["pnl"], marker_color=colors))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, yaxis_title="P&L (₹)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No P&L data to display.")

with tab_positions:
    st.subheader("Net Positions")
    if positions_df.empty:
        st.info("No open positions.")
    else:
        st.dataframe(positions_df, use_container_width=True, hide_index=True)

with tab_holdings:
    st.subheader("Holdings")
    if holdings_df.empty:
        st.info("No holdings.")
    else:
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

with tab_orders:
    st.subheader("Order Book")
    if orders_df.empty:
        st.info("No orders found.")
    else:
        status_options = sorted(orders_df["status"].dropna().unique().tolist()) if "status" in orders_df.columns else []
        selected = st.multiselect("Filter by status", status_options, default=status_options)
        filtered = orders_df[orders_df["status"].isin(selected)] if status_options else orders_df
        st.dataframe(filtered, use_container_width=True, hide_index=True)
