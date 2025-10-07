# =============================================================== 
# Power Service Solutions GmbH — DCF + WACC Valuation App (Secure)
# ===============================================================

import os, io, json, datetime as dt
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ------------------------
# LOAD SECRETS / ENV
# ------------------------
load_dotenv()
USERS = {
    os.getenv("USER1"): os.getenv("USER1_PWD"),
    os.getenv("USER2"): os.getenv("USER2_PWD"),
    os.getenv("USER3"): os.getenv("USER3_PWD"),
    os.getenv("USER4"): os.getenv("USER4_PWD"),
    os.getenv("USER5"): os.getenv("USER5_PWD"),
    os.getenv("USER6"): os.getenv("USER6_PWD"),
}

# Friendly display names
USER_NAMES = {
    "d.garcia": "Daniel Garcia Rey",
    "t.held": "Thomas Held",
    "b.arrieta": "Borja Arrieta",
    "m.peter": "Michel Peter",
    "c.bahn": "Cristoph Bahn",
    "tgv": "Tomas Garcia Villanueva",
}

# ------------------------
# SIMPLE LOGIN SYSTEM
# ------------------------
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    st.title("🔐 PSS Corporate Valuation Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in USERS and pwd == USERS[user]:
            st.session_state["auth"] = True
            st.session_state["user"] = user
            name = USER_NAMES.get(user, user)
            st.success(f"Welcome back, {name.split()[0]}!")
        else:
            st.error("Invalid credentials. Please try again.")
    st.stop()

# ------------------------
# CONFIG
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def ts_folder(root):
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(root, stamp)
    os.makedirs(path, exist_ok=True)
    return path

# ------------------------
# BASE DATA
# ------------------------
years = [2025, 2026, 2027, 2028, 2029]
data = {
    "Sales_kEUR": [55650, 92408, 113161, 120765, 123180],
    "EBIT_kEUR":  [600, 2106, 5237, 7456, 7641],
    "Net_kEUR":   [535, 2135, 4845, 5218, 5322],
    "Equity_kEUR":[12596, 14731, 19219, 24750, 25850],
    "Cash_kEUR":  [8176, 11205, 20394, 28367, 36000],
    "FCF_kEUR":   [-6884, 2749, 9054, 7716, 5322],
}
df_base = pd.DataFrame(data, index=years)

# ------------------------
# FUNCTIONS
# ------------------------
def capm_cost_equity(rf, mrp, beta):
    return rf + beta * mrp

def compute_wacc(E, D, Re, Rd, tax):
    if (E + D) == 0:
        return Re
    return (E / (E + D)) * Re + (D / (E + D)) * Rd * (1 - tax)

def pv(values, r):
    return [v / ((1 + r) ** (i + 1)) for i, v in enumerate(values)]

def safe_irr(cash_flows):
    """Robust IRR calculation that returns NaN safely if undefined."""
    cf = np.array(cash_flows, dtype=float)
    if np.any(cf < 0) and np.any(cf > 0):
        try:
            return np.irr(cf)
        except Exception:
            return np.nan
    return np.nan

# ------------------------
# STREAMLIT APP UI
# ------------------------
st.set_page_config(page_title="PSS Valuation", layout="wide")
st.title("💼 Power Service Solutions GmbH — DCF & WACC Model")

display_name = USER_NAMES.get(st.session_state["user"], st.session_state["user"])
login_time = dt.datetime.now().strftime("%H:%M")
st.markdown(f"👤 Logged in as **{display_name}** | Session started at **{login_time}**")

st.subheader("Excel Extract — Key Lines (FY 2025–2029)")
st.dataframe(
    df_base.style.format({
        "Sales_kEUR": "{:,.0f}",
        "EBIT_kEUR": "{:,.0f}",
        "Net_kEUR": "{:,.0f}",
        "Equity_kEUR": "{:,.0f}",
        "Cash_kEUR": "{:,.0f}",
        "FCF_kEUR": "€{:,.0f}",
    }),
    use_container_width=True,
)

# ------------------------
# SIDEBAR INPUTS
# ------------------------
st.sidebar.header("Capital & Risk Assumptions")
rf = st.sidebar.number_input("Risk-free rate (Rf)", value=0.027, step=0.001, format="%.4f")
mrp = st.sidebar.number_input("Market risk premium (MRP)", value=0.04, step=0.001, format="%.4f")
beta = st.sidebar.number_input("Equity beta (β)", value=1.2, step=0.05, format="%.2f")
tax = st.sidebar.number_input("Tax rate (T)", value=0.30, step=0.01, format="%.2f", min_value=-1.0, max_value=1.0)
g = st.sidebar.number_input("Terminal growth (g)", value=0.02, step=0.001, format="%.4f", min_value=-1.0, max_value=1.0)

st.sidebar.markdown("---")
st.sidebar.header("Operational Assumptions")
dep_pct = st.sidebar.number_input("Depreciation % of Sales", value=0.01, step=0.001, format="%.4f", min_value=-1.0, max_value=1.0)
capex_pct = st.sidebar.number_input("CapEx % of Sales", value=0.01, step=0.001, format="%.4f", min_value=-1.0, max_value=1.0)
use_nwc = st.sidebar.checkbox("Include ΔNWC adjustment", value=True)
nwc_pct = 0.03
if use_nwc:
    nwc_pct = st.sidebar.number_input("ΔNWC % of ΔSales", value=0.03, step=0.005, format="%.4f", min_value=-1.0, max_value=1.0)
sales_growth = st.sidebar.number_input("Sales growth for 2029", value=0.02, step=0.005, format="%.4f", min_value=-1.0, max_value=1.0)

st.sidebar.markdown("---")
st.sidebar.header("Debt & Financing")
debt_amount = st.sidebar.number_input("Debt (€)", value=0.0, step=1_000_000.0, format="%.2f", min_value=-1e9, max_value=1e9)
rd = st.sidebar.number_input("Cost of Debt (Rd)", value=0.04, step=0.005, format="%.4f", min_value=-1.0, max_value=1.0)

st.sidebar.markdown("---")
st.sidebar.header("Acquisition & IRR Settings")
assumed_price_mdkb = st.sidebar.number_input(
    "Assumed Price for MDKB (€)", value=0.0, step=100_000.0, format="%.0f", min_value=0.0, max_value=1e9
)

# ------------------------
# CALCULATIONS
# ------------------------
E = df_base["Equity_kEUR"].iloc[-1] * 1000
D = debt_amount
Re = capm_cost_equity(rf, mrp, beta)
WACC = compute_wacc(E, D, Re, rd, tax)

sales_eur = df_base["Sales_kEUR"].values * 1000
ebit_eur = df_base["EBIT_kEUR"].values * 1000
fcfs = []

for i, y in enumerate(years):
    s = sales_eur[i]
    prev_s = sales_eur[i - 1] if i > 0 else s
    e = ebit_eur[i]
    dep = s * dep_pct
    capex = s * capex_pct
    dNWC = (s - prev_s) * nwc_pct if (use_nwc and i > 0) else 0
    fcf = (e * (1 - tax)) + dep - capex - dNWC
    if y == 2029:
        fcf = df_base.loc[y, "FCF_kEUR"] * 1000
    fcfs.append(fcf)

cash = df_base.loc[2029, "Cash_kEUR"] * 1000

pv_fcfs = pv(fcfs, WACC)
tv = fcfs[-1] * (1 + g) / (WACC - g) if WACC > g else np.nan
pv_tv = tv / ((1 + WACC) ** len(fcfs)) if not np.isnan(tv) else 0
EV = sum(pv_fcfs) + pv_tv
equity_value = EV + cash - D

# ------------------------
# IRR CALCULATION (SPA Instalments + MDKB Adjustment)
# ------------------------
total_purchase_price = 13_300_000
ratio = max(0, (total_purchase_price - assumed_price_mdkb) / total_purchase_price)

instalments = {
    2025: -500_000 * ratio,
    2026: -2_500_000 * ratio,
    2027: -3_500_000 * ratio,
    2028: -6_800_000 * ratio,
}

irr_cash_flows = [
    instalments.get(2025, 0) + fcfs[0],
    instalments.get(2026, 0) + fcfs[1],
    instalments.get(2027, 0) + fcfs[2],
    instalments.get(2028, 0) + fcfs[3],
    fcfs[4] + pv_tv,
]
IRR = safe_irr(irr_cash_flows)

# ------------------------
# METRICS DISPLAY
# ------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Cost of Equity (Re)", f"{Re*100:.2f}%")
c2.metric("Cost of Debt (Rd)", f"{rd*100:.2f}%")
c3.metric("WACC", f"{WACC*100:.2f}%")
c4.metric("Enterprise Value (EV)", f"€{EV:,.0f}")
c5.metric("Equity Value", f"€{equity_value:,.0f}")
c6.metric("IRR (Unlevered)", f"{IRR*100:.2f}%" if not np.isnan(IRR) else "N/A")

# ------------------------
# DCF TABLE
# ------------------------
df_results = pd.DataFrame({
    "Year": years,
    "Sales (€)": sales_eur,
    "EBIT (€)": ebit_eur,
    "Net (€)": df_base["Net_kEUR"].values * 1000,
    "FCF (€)": fcfs,
    "PV(FCF)": pv_fcfs,
})
st.subheader("DCF Inputs & Results (FY 2025–2029)")
st.dataframe(df_results.style.format({
    "Sales (€)": "€{:,.0f}",
    "EBIT (€)": "€{:,.0f}",
    "Net (€)": "€{:,.0f}",
    "FCF (€)": "€{:,.0f}",
    "PV(FCF)": "€{:,.0f}",
}), use_container_width=True)
