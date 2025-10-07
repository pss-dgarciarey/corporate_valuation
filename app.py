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
st.set_page_config(page_title="PSS Valuation", layout="wide")
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

def npv_from_rate(rate, cashflows):
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

def irr_bisection(cashflows, low=-0.9999, high=10.0, max_iter=200, tol=1e-8):
    cf = np.array(cashflows, dtype=float)
    if not (np.any(cf < 0) and np.any(cf > 0)):
        return float("nan")
    f_low = npv_from_rate(low, cf)
    f_high = npv_from_rate(high, cf)
    expand = 0
    while f_low * f_high > 0 and high < 1e6 and expand < 50:
        high *= 2
        f_high = npv_from_rate(high, cf)
        expand += 1
    if f_low * f_high > 0:
        return float("nan")
    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = npv_from_rate(mid, cf)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return mid

def adjust_instalments_absolute_deduction(base, deduction):
    """
    Adjust instalments by absolute deduction (tail-first). Ensures at least one
    negative cash outflow exists for IRR calculation.
    """
    total = sum(a for _, a in base)
    deduction = min(max(deduction, 0.0), total)
    remaining = deduction
    adjusted = []
    for year, amt in reversed(base):
        red = min(amt, remaining)
        new_amt = amt - red
        adjusted.append((year, -new_amt))
        remaining -= red
    adjusted.sort(key=lambda x: x[0])
    # Ensure at least one negative outflow
    if all(v == 0 for _, v in adjusted):
        adjusted[0] = (adjusted[0][0], -1.0)
    return dict(adjusted), total

# ------------------------
# UI HEADER
# ------------------------
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
tax = st.sidebar.number_input("Tax rate (T)", value=0.30, step=0.01, format="%.2f")
g = st.sidebar.number_input("Terminal growth (g)", value=0.02, step=0.001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("Operational Assumptions")
dep_pct = st.sidebar.number_input("Depreciation % of Sales", value=0.01, step=0.001, format="%.4f")
capex_pct = st.sidebar.number_input("CapEx % of Sales", value=0.01, step=0.001, format="%.4f")
use_nwc = st.sidebar.checkbox("Include ΔNWC adjustment", value=True)
nwc_pct = 0.10
if use_nwc:
    nwc_pct = st.sidebar.number_input("ΔNWC % of ΔSales", value=0.10, step=0.005, format="%.4f")
sales_growth = st.sidebar.number_input("Sales growth for 2029", value=0.02, step=0.005, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("Debt & Financing")
debt_amount = st.sidebar.number_input("Debt (€)", value=0.0, step=1_000_000.0, format="%.2f")
rd = st.sidebar.number_input("Cost of Debt (Rd)", value=0.04, step=0.005, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("Acquisition & IRR Settings")
assumed_price_mdkb = st.sidebar.number_input(
    "Assumed Price for MDKB (€)",
    value=0.0, step=100_000.0, format="%.0f"
)

# ------------------------
# DCF CALCULATIONS
# ------------------------
E = df_base["Equity_kEUR"].iloc[-1] * 1000
D = debt_amount
Re = capm_cost_equity(rf, mrp, beta)
WACC = compute_wacc(E, D, Re, rd, tax)

sales_eur = df_base["Sales_kEUR"].values * 1000
ebit_eur  = df_base["EBIT_kEUR"].values * 1000
net_eur   = df_base["Net_kEUR"].values * 1000

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
# METRICS DISPLAY
# ------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cost of Equity (Re)", f"{Re*100:.2f}%")
c2.metric("Cost of Debt (Rd)", f"{rd*100:.2f}%")
c3.metric("WACC", f"{WACC*100:.2f}%")
c4.metric("Enterprise Value (EV)", f"€{EV:,.0f}")
c5.metric("Equity Value", f"€{equity_value:,.0f}")

# ------------------------
# TABLES & DCF GRAPH
# ------------------------
df_results = pd.DataFrame({
    "Year": years,
    "Sales (€)": sales_eur,
    "EBIT (€)": ebit_eur,
    "Net (€)": net_eur,
    "FCF (€)": fcfs,
    "PV(FCF)": pv_fcfs,
})
st.subheader("DCF Inputs & Results (FY 2025–2029)")
st.dataframe(df_results.style.format("€{:,.0f}"), use_container_width=True)

fig = plt.figure(figsize=(9, 4.5))
plt.plot(years, fcfs, "o-", label="FCF (€)")
plt.plot(years, pv_fcfs, "o-", label="PV(FCF)")
plt.axhline(0, color="gray", lw=0.8)
plt.legend()
plt.title("Free Cash Flow and Present Value (FY 2025–2029)")
plt.xlabel("Year")
plt.ylabel("EUR")
st.pyplot(fig)

# ------------------------
# IRR SECTION (after valuation graph)
# ------------------------
st.subheader("💰 Internal Rate of Return (IRR) Analysis")

# --- SPA Instalments adjusted by MDKB
base_instalments = [(2025, 500_000), (2026, 2_500_000), (2027, 3_500_000), (2028, 6_800_000)]
adjusted_outflows, total_purchase_price = adjust_instalments_absolute_deduction(base_instalments, assumed_price_mdkb)

# --- Ensure minimum initial outflow for realism (avoid NaN)
total_investment = sum(abs(v) for v in adjusted_outflows.values())
if total_investment == 0:
    adjusted_outflows[2025] = -1_000_000  # fallback investment
    total_investment = 1_000_000

# --- IRR 1: FCF-based
irr_cash_flows = []
irr_rows = []
for i, y in enumerate(years):
    instal = adjusted_outflows.get(y, 0.0)
    inflow = fcfs[i] + (tv if y == 2029 and not np.isnan(tv) else 0)
    net_cf = instal + inflow
    irr_cash_flows.append(net_cf)
    irr_rows.append([y, instal, fcfs[i], (tv if y == 2029 else 0), net_cf])
IRR_fcf = irr_bisection(irr_cash_flows)

# --- IRR 2: Net Profit-based
irr_cash_flows_net = []
irr_rows_net = []
for i, y in enumerate(years):
    instal = adjusted_outflows.get(y, 0.0)
    inflow = net_eur[i] + (tv if y == 2029 and not np.isnan(tv) else 0)
    net_cf = instal + inflow
    irr_cash_flows_net.append(net_cf)
    irr_rows_net.append([y, instal, net_eur[i], (tv if y == 2029 else 0), net_cf])

# sanity check: ensure there’s at least one sign change
if not (np.any(np.array(irr_cash_flows_net) < 0) and np.any(np.array(irr_cash_flows_net) > 0)):
    irr_cash_flows_net[0] -= total_investment  # inject the "investment" effect
IRR_net = irr_bisection(irr_cash_flows_net)

# --- Display IRRs
col1, col2 = st.columns(2)
col1.metric("IRR (FCF-based)", f"{IRR_fcf*100:.2f}%" if not np.isnan(IRR_fcf) else "N/A")
col2.metric("IRR (Net Profit-based)", f"{IRR_net*100:.2f}%" if not np.isnan(IRR_net) else "N/A")

# --- Comparison bar chart
fig_irr = plt.figure(figsize=(5.5, 4))
plt.bar(["FCF-based", "Net Profit-based"], [IRR_fcf*100, IRR_net*100], color=["#0073e6", "#00b386"])
plt.ylabel("IRR (%)")
plt.title("IRR Comparison")
for i, val in enumerate([IRR_fcf*100, IRR_net*100]):
    if not np.isnan(val):
        plt.text(i, val + 0.5, f"{val:.2f}%", ha="center", fontsize=10)
st.pyplot(fig_irr)


# ------------------------
# SENSITIVITY MATRIX
# ------------------------
st.subheader("📊 Sensitivity Analysis — EV by WACC & Terminal Growth")
wacc_range = np.arange(max(0.05, WACC - 0.02), WACC + 0.025, 0.005)
g_range = np.arange(g - 0.01, g + 0.015, 0.005)
matrix = []
for w in wacc_range:
    row = []
    for gg in g_range:
        tv_test = fcfs[-1] * (1 + gg) / (w - gg) if w > gg else np.nan
        ev_test = sum(pv(fcfs, w)) + (tv_test / ((1 + w) ** len(fcfs)) if not np.isnan(tv_test) else 0)
        row.append(ev_test)
    matrix.append(row)
df_sens = pd.DataFrame(matrix, index=[f"{x*100:.1f}%" for x in wacc_range],
                       columns=[f"{y*100:.1f}%" for y in g_range])
st.dataframe(df_sens.style.format("€{:,.0f}"), use_container_width=True)

# ------------------------
# EXPORT
# ------------------------
st.markdown("### 📦 Export Options")
st.info("Choose how to export your results to local folder or browser.")

if st.button("💾 Export locally"):
    out = ts_folder(RESULTS_DIR)
    df_results.to_csv(os.path.join(out, "valuation_summary.csv"), index=False)
    df_sens.to_csv(os.path.join(out, "sensitivity_matrix.csv"))
    with open(os.path.join(out, "assumptions.json"), "w") as f:
        json.dump({
            "rf": rf, "mrp": mrp, "beta": beta, "Re": Re, "Rd": rd, "tax": tax,
            "g": g, "dep_pct": dep_pct, "capex_pct": capex_pct, "use_nwc": use_nwc,
            "nwc_pct": nwc_pct, "debt": D, "EV": EV, "EquityValue": equity_value,
            "WACC": WACC, "fcf": fcfs, "pv_fcfs": pv_fcfs, "tv": tv,
            "pv_tv": pv_tv, "IRR_FCF": IRR_fcf, "IRR_NetProfit": IRR_net,
            "assumed_price_mdkb": assumed_price_mdkb
        }, f, indent=2)
    fig.savefig(os.path.join(out, "DCF_chart.png"), dpi=150)
    fig_irr.savefig(os.path.join(out, "IRR_chart.png"), dpi=150)
    st.success(f"✅ Exported locally to: {out}")
