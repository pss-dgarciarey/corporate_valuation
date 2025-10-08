# ===============================================================
# Multi-Company DCF + WACC Valuation App (PSS & MDKB) — Secure
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
    st.title("🔐 Corporate Valuation Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user in USERS and pwd == USERS[user]:
            st.session_state["auth"] = True
            st.session_state["user"] = user
            name = USER_NAMES.get(user, user)
            welcome_name = "Thomas" if user == "t.held" else name.split()[0]
            st.success(f"Welcome back, {welcome_name}!")
        else:
            st.error("Invalid credentials. Please try again.")
    st.stop()

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(page_title="PSS / MDKB Valuation", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def ts_folder(root):
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(root, stamp)
    os.makedirs(path, exist_ok=True)
    return path

# ------------------------
# PSS BASE DATA (kEUR, 2025–2029)
# ------------------------
years_pss = [2025, 2026, 2027, 2028, 2029]
data_pss = {
    "Sales_kEUR": [55650, 92408, 113161, 120765, 123180],
    "EBIT_kEUR":  [600, 2106, 5237, 7456, 7641],
    "Net_kEUR":   [535, 2135, 4845, 5218, 5322],
    "Equity_kEUR":[12596, 14731, 19219, 24750, 25850],
    "Cash_kEUR":  [8176, 11205, 20394, 28367, 36000],
    "FCF_kEUR":   [-6884, 2749, 9054, 7716, 5322],
}
df_pss = pd.DataFrame(data_pss, index=years_pss)

# ------------------------
# MDKB DATA (from your screenshots, EUR m → converted to kEUR)
# Given for FY24–FY28; we align to FY25–FY29 and synthesize FY29
# as per your rule (use 2028 margins; grow Sales & FCF by sidebar %).
# ------------------------
# Source literals (EUR m), VERIFY if any number differs:
sales_m_24_28 = [15.3, 12.7, 11.7, 12.0, 12.3]
ebit_m_24_28  = [0.4,  0.8,  0.9,  1.0,  1.0]
net_m_24_28   = [0.6,  0.6,  0.7,  0.7,  0.7]
cash_m_24_28  = [2.9, 2.2, 2.5, 2.9, 3.3]
fcf_m_24_28   = [-2.9, -0.7, 0.3, 0.2, 0.3]      # after taxes
equity_m_year = {2025:12.1, 2026:12.6, 2027:13.3, 2028:14.0, 2029:14.6}

def m_to_k(seq): return [int(round(v*1000)) for v in seq]

# Build FY25–FY28 directly from provided FY25–FY28 rows (index 1..4 of 24–28 arrays)
years_mdkb_base = [2025, 2026, 2027, 2028]
sales_m_25_28 = sales_m_24_28[1:]
ebit_m_25_28  = ebit_m_24_28[1:]
net_m_25_28   = net_m_24_28[1:]
cash_m_25_28  = cash_m_24_28[1:]
fcf_m_25_28   = fcf_m_24_28[1:]

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
    PSS-only: Adjust instalments by absolute deduction (tail-first).
    Returns dict {year: -outflow}.
    """
    total = sum(a for _, a in base)
    deduction = min(max(deduction, 0.0), total)
    remaining = deduction
    adjusted = []
    for year, amt in reversed(base):
        red = min(amt, remaining)
        new_amt = amt - red
        adjusted.append((year, -new_amt))  # negative outflow
        remaining -= red
    adjusted.sort(key=lambda x: x[0])
    return dict(adjusted), total

# ------------------------
# UI HEADER
# ------------------------
display_name = USER_NAMES.get(st.session_state["user"], st.session_state["user"])
login_time = dt.datetime.now().strftime("%H:%M")
st.title("💼 Corporate Valuation — DCF & WACC")
st.markdown(f"👤 Logged in as **{display_name}** | Session started at **{login_time}**")

# ------------------------
# SIDEBAR — company + assumptions
# ------------------------
company = st.sidebar.selectbox("Select Company", ["PSS", "MDKB"])

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
use_nwc = st.sidebar.checkbox("Include ΔNWC adjustment (only if FCF missing)", value=True)
nwc_pct = 0.10
if use_nwc:
    nwc_pct = st.sidebar.number_input("ΔNWC % of ΔSales", value=0.10, step=0.005, format="%.4f")

# PSS already had a last-year growth control. Add **MDKB extension growth** for 2029 synthesis.
mdkb_extend_growth = st.sidebar.number_input("MDKB 2029 growth (Sales & FCF)", value=0.02, step=0.005, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("Debt & Financing")
debt_amount = st.sidebar.number_input("Debt (€)", value=0.0, step=1_000_000.0, format="%.2f")
rd = st.sidebar.number_input("Cost of Debt (Rd)", value=0.04, step=0.005, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("Acquisition & IRR Settings")
assumed_price_mdkb = st.sidebar.number_input("Assumed Price for MDKB (€)", value=0.0, step=100_000.0, format="%.0f")

# ------------------------
# BUILD DATAFRAME BY COMPANY
# ------------------------
if company == "PSS":
    df_base = df_pss.copy()
    title_company = "Power Service Solutions GmbH (PSS)"
else:
    # Construct FY25–FY29 for MDKB
    # FY25–FY28 from provided data; FY29 synthesized.
    sales_25_28_k = m_to_k(sales_m_25_28)
    ebit_25_28_k  = m_to_k(ebit_m_25_28)
    net_25_28_k   = m_to_k(net_m_25_28)
    cash_25_28_k  = m_to_k(cash_m_25_28)
    fcf_25_28_k   = m_to_k(fcf_m_25_28)

    # 2028 margins
    s28, ebit28, net28, fcf28, cash28 = sales_25_28_k[-1], ebit_25_28_k[-1], net_25_28_k[-1], fcf_25_28_k[-1], cash_25_28_k[-1]
    ebit_margin_28 = (ebit28 / s28) if s28 != 0 else 0.0
    net_margin_28  = (net28  / s28) if s28 != 0 else 0.0

    # Synthesize 2029 by your rule:
    sales_29_k = int(round(s28 * (1.0 + mdkb_extend_growth)))
    ebit_29_k  = int(round(sales_29_k * ebit_margin_28))       # keep 2028 EBIT margin
    net_29_k   = int(round(sales_29_k * net_margin_28))        # keep 2028 Net margin
    fcf_29_k   = int(round(fcf28 * (1.0 + mdkb_extend_growth)))# grow FCF by same %
    cash_29_k  = int(round(cash28 + fcf_29_k))                  # simple cash roll-forward
    equity_25_29_k = [int(round(equity_m_year[y]*1000)) for y in [2025, 2026, 2027, 2028, 2029]]

    df_base = pd.DataFrame({
        "Sales_kEUR": sales_25_28_k + [sales_29_k],
        "EBIT_kEUR":  ebit_25_28_k  + [ebit_29_k],
        "Net_kEUR":   net_25_28_k   + [net_29_k],
        "Equity_kEUR": equity_25_29_k,
        "Cash_kEUR":  cash_25_28_k  + [cash_29_k],
        "FCF_kEUR":   fcf_25_28_k   + [fcf_29_k],
    }, index=[2025, 2026, 2027, 2028, 2029])
    title_company = "MDKB GmbH"

years = list(df_base.index)

# ---- DISPLAY TABLE: keep Year as text (no €)
st.subheader(f"Excel Extract — Key Lines ({years[0]}–{years[-1]}) — {title_company}")
df_display = df_base.copy()
df_display.insert(0, "Year", [str(y) for y in years])
st.dataframe(
    df_display.style.format({
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
# DCF CALCULATIONS
# ------------------------
E = (df_base["Equity_kEUR"].dropna().iloc[-1]) * 1000  # EUR
D = debt_amount
Re = capm_cost_equity(rf, mrp, beta)
WACC = compute_wacc(E, D, Re, rd, tax)

sales_eur = (df_base["Sales_kEUR"].values * 1000).astype(float)
ebit_eur  = (df_base["EBIT_kEUR"].values  * 1000).astype(float)
net_eur   = (df_base["Net_kEUR"].values   * 1000).astype(float)
fcf_eur   = (df_base["FCF_kEUR"].values   * 1000).astype(float)

# If FCF provided, use it directly
fcfs = fcf_eur.tolist()

cash = float(df_base["Cash_kEUR"].dropna().iloc[-1] * 1000)
pv_fcfs = pv(fcfs, WACC)
tv = fcfs[-1] * (1 + g) / (WACC - g) if WACC > g else np.nan
pv_tv = tv / ((1 + WACC) ** len(fcfs)) if not np.isnan(tv) else 0.0
EV = float(sum(pv_fcfs) + pv_tv)
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
    "Year": [str(y) for y in years],
    "Sales (€)": sales_eur,
    "EBIT (€)": ebit_eur,
    "Net (€)": net_eur,
    "FCF (€)": fcfs,
    "PV(FCF)": pv_fcfs,
})
st.subheader(f"DCF Inputs & Results ({years[0]}–{years[-1]}) — {title_company}")
st.dataframe(
    df_results.style.format({
        "Sales (€)": "€{:,.0f}",
        "EBIT (€)": "€{:,.0f}",
        "Net (€)": "€{:,.0f}",
        "FCF (€)": "€{:,.0f}",
        "PV(FCF)": "€{:,.0f}",
    }),
    use_container_width=True
)

fig = plt.figure(figsize=(9, 4.5))
plt.plot(years, fcfs, "o-", label="FCF (€)")
plt.plot(years, pv_fcfs, "o-", label="PV(FCF)")
plt.axhline(0, color="gray", lw=0.8)
plt.legend()
plt.title(f"Free Cash Flow and Present Value ({years[0]}–{years[-1]}) — {company}")
plt.xlabel("Year")
plt.ylabel("EUR")
st.pyplot(fig)

# ------------------------
# IRR SECTION (after valuation graph)
# ------------------------
st.subheader("💰 Internal Rate of Return (IRR) Analysis")

irr_rows = []
irr_rows_net = []
IRR_fcf = np.nan
IRR_net = np.nan
irr_note_fcf = ""
irr_note_net = ""

if company == "PSS":
    # === PSS IRR LOGIC — EXACTLY AS BEFORE ===
    base_instalments = [
        (2025, 500_000),
        (2026, 2_500_000),
        (2027, 3_500_000),
        (2028, 6_800_000)
    ]
    adjusted_outflows, total_purchase_price = adjust_instalments_absolute_deduction(
        base_instalments, assumed_price_mdkb
    )

    # IRR 1: FCF-based
    irr_cash_flows = []
    for i, y in enumerate(years):
        instal = adjusted_outflows.get(y, 0.0)  # negative or zero
        inflow = fcfs[i] + (tv if (i == len(years)-1 and not np.isnan(tv)) else 0.0)
        net_cf = instal + inflow
        irr_cash_flows.append(net_cf)
        irr_rows.append([str(y), instal, fcfs[i], (tv if i == len(years)-1 else 0.0), net_cf])
    IRR_fcf = irr_bisection(irr_cash_flows)
    if np.isnan(IRR_fcf):
        irr_note_fcf = "No sign change in cash flows; tweak MDKB deduction vs SPA schedule."

    # IRR 2: Net-profit IRR (equity-style)
    effective_price_paid = max(total_purchase_price - assumed_price_mdkb, 0.0)
    irr_cash_flows_net = []
    for i, y in enumerate(years):
        outflow = -effective_price_paid if i == 0 else 0.0
        inflow = net_eur[i] + (tv if (i == len(years)-1 and not np.isnan(tv)) else 0.0)
        net_cf = outflow + inflow
        irr_cash_flows_net.append(net_cf)
        irr_rows_net.append([str(y), outflow, net_eur[i], (tv if i == len(years)-1 else 0.0), net_cf])
    IRR_net = irr_bisection(irr_cash_flows_net)
    if np.isnan(IRR_net):
        irr_note_net = "IRR undefined (e.g., effective price paid = 0). Reduce deduction so price > 0."

else:
    # === MDKB IRR LOGIC — NEW, independent ===
    initial_outflow = -float(assumed_price_mdkb)

    # IRR 1: FCF-based
    irr_cash_flows = []
    for i, y in enumerate(years):
        outflow = initial_outflow if i == 0 else 0.0
        inflow = fcfs[i] + (tv if (i == len(years)-1 and not np.isnan(tv)) else 0.0)
        net_cf = outflow + inflow
        irr_cash_flows.append(net_cf)
        irr_rows.append([str(y), outflow, fcfs[i], (tv if i == len(years)-1 else 0.0), net_cf])
    IRR_fcf = irr_bisection(irr_cash_flows)
    if np.isnan(IRR_fcf):
        irr_note_fcf = "IRR undefined (no sign change). Set a positive price and ensure positive inflows."

    # IRR 2: Net-profit IRR
    irr_cash_flows_net = []
    for i, y in enumerate(years):
        outflow = initial_outflow if i == 0 else 0.0
        inflow = net_eur[i] + (tv if (i == len(years)-1 and not np.isnan(tv)) else 0.0)
        net_cf = outflow + inflow
        irr_cash_flows_net.append(net_cf)
        irr_rows_net.append([str(y), outflow, net_eur[i], (tv if i == len(years)-1 else 0.0), net_cf])
    IRR_net = irr_bisection(irr_cash_flows_net)
    if np.isnan(IRR_net):
        irr_note_net = "IRR undefined (no sign change). Increase price or check net profits."

# --- Metrics + notes
col1, col2 = st.columns(2)
col1.metric("IRR (FCF-based)", f"{IRR_fcf*100:.2f}%" if not np.isnan(IRR_fcf) else "N/A")
col2.metric("IRR (Net Profit-based)", f"{IRR_net*100:.2f}%" if not np.isnan(IRR_net) else "N/A")
if irr_note_fcf:
    st.caption(f"ℹ️ {irr_note_fcf}")
if irr_note_net:
    st.caption(f"ℹ️ {irr_note_net}")

# --- Bar chart
fig_irr = plt.figure(figsize=(5.5, 4))
vals = [IRR_fcf*100 if not np.isnan(IRR_fcf) else np.nan,
        IRR_net*100 if not np.isnan(IRR_net) else np.nan]
plt.bar(["FCF-based", "Net Profit-based"], vals)
plt.ylabel("IRR (%)")
plt.title(f"IRR Comparison — {company}")
for i, val in enumerate(vals):
    if not np.isnan(val):
        plt.text(i, val + 0.5, f"{val:.2f}%", ha="center", fontsize=10)
st.pyplot(fig_irr)

# --- IRR Cash Flow Tables
if company == "PSS":
    st.markdown("#### IRR Cash Flows (FCF-based) — PSS")
    df_irr = pd.DataFrame(
        irr_rows,
        columns=["Year", "Instalment (outflow)", "FCF (inflow)", "Terminal Value (inflow)", "Net CF for IRR"]
    )
else:
    st.markdown("#### IRR Cash Flows (FCF-based) — MDKB")
    df_irr = pd.DataFrame(
        irr_rows,
        columns=["Year", "Initial Outflow (t0)", "FCF (inflow)", "Terminal Value (inflow)", "Net CF for IRR"]
    )

st.dataframe(
    df_irr.style.format({
        df_irr.columns[1]: "€{:,.0f}",
        df_irr.columns[2]: "€{:,.0f}",
        "Terminal Value (inflow)": "€{:,.0f}",
        "Net CF for IRR": "€{:,.0f}",
    }),
    use_container_width=True,
)

st.markdown("#### IRR Cash Flows (Net Profit-based)")
df_irr_net = pd.DataFrame(
    irr_rows_net,
    columns=["Year", "Outflow @t0 (effective price)" if company=="PSS" else "Initial Outflow (t0)",
             "Net Profit (inflow)", "Terminal Value (inflow)", "Net CF for IRR"]
)
st.dataframe(
    df_irr_net.style.format({
        df_irr_net.columns[1]: "€{:,.0f}",
        "Net Profit (inflow)": "€{:,.0f}",
        "Terminal Value (inflow)": "€{:,.0f}",
        "Net CF for IRR": "€{:,.0f}",
    }),
    use_container_width=True,
)

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
        ev_test = sum(pv(fcfs, w)) + (tv_test / ((1 + w) ** len(fcfs)) if not np.isnan(tv_test) else 0.0)
        row.append(ev_test)
    matrix.append(row)
df_sens = pd.DataFrame(matrix, index=[f"{x*100:.1f}%" for x in wacc_range],
                       columns=[f"{y*100:.1f}%" for y in g_range])
st.dataframe(df_sens.style.format("€{:,.0f}"), use_container_width=True)

# ------------------------
# EXPORT OPTIONS
# ------------------------
st.markdown("### 📦 Export Options")
st.info("Choose how to export your results to local folder or browser.")

if st.button("💾 Export locally"):
    out = ts_folder(RESULTS_DIR)
    df_results.to_csv(os.path.join(out, f"{company}_valuation_summary.csv"), index=False)
    df_sens.to_csv(os.path.join(out, f"{company}_sensitivity_matrix.csv"))
    df_irr.to_csv(os.path.join(out, f"{company}_irr_cashflows_fcf.csv"), index=False)
    df_irr_net.to_csv(os.path.join(out, f"{company}_irr_cashflows_netprofit.csv"), index=False)
    with open(os.path.join(out, f"{company}_assumptions.json"), "w") as f:
        json.dump({
            "company": company,
            "rf": rf, "mrp": mrp, "beta": beta, "Re": Re, "Rd": rd, "tax": tax,
            "g": g, "dep_pct": dep_pct, "capex_pct": capex_pct, "use_nwc": use_nwc,
            "nwc_pct": nwc_pct, "debt": float(debt_amount), "EV": EV, "EquityValue": equity_value,
            "WACC": WACC, "fcf": fcfs, "pv_fcfs": pv_fcfs, "tv": tv, "pv_tv": pv_tv,
            "IRR_FCF": IRR_fcf, "IRR_NetProfit": IRR_net,
            "assumed_price_mdkb": float(assumed_price_mdkb),
            "mdkb_extend_growth": float(mdkb_extend_growth),
        }, f, indent=2)
    fig.savefig(os.path.join(out, f"{company}_DCF_chart.png"), dpi=150, bbox_inches="tight")
    fig_irr.savefig(os.path.join(out, f"{company}_IRR_chart.png"), dpi=150, bbox_inches="tight")
    st.success(f"✅ Exported locally to: {out}")

# Browser export
st.markdown("#### ⬇️ Download files to your device")
option = st.multiselect(
    "Select what to download:",
    ["Summary CSV", "Sensitivity CSV", "IRR FCF CSV", "IRR Net CSV", "Excel (Full Report)", "Charts (PNG)"],
    default=["Excel (Full Report)"]
)

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    df_results.to_excel(writer, sheet_name="DCF_Results", index=False)
    df_sens.to_excel(writer, sheet_name="Sensitivity", index=True)
    df_irr.to_excel(writer, sheet_name="IRR_FCF", index=False)
    df_irr_net.to_excel(writer, sheet_name="IRR_NetProfit", index=False)
    pd.DataFrame({
        "Metric": ["Company","EV", "Equity Value", "IRR (FCF)", "IRR (Net Profit)", "WACC", "Re", "Rd",
                   "t0 Outflow / Effective Price", "MDKB Assumption", "MDKB 2029 growth"],
        "Value": [company, EV, equity_value, IRR_fcf, IRR_net, WACC, Re, rd,
                  (max( (500000+2500000+3500000+6800000) - assumed_price_mdkb, 0.0)
                   if company=="PSS" else float(assumed_price_mdkb)),
                  float(assumed_price_mdkb), float(mdkb_extend_growth)]
    }).to_excel(writer, sheet_name="Summary", index=False)
excel_buffer.seek(0)

prefix = company

if "Summary CSV" in option:
    st.download_button(
        label=f"Download {prefix} Summary CSV",
        data=df_results.to_csv(index=False).encode(),
        file_name=f"{prefix}_Valuation_Summary.csv",
        mime="text/csv"
    )

if "Sensitivity CSV" in option:
    st.download_button(
        label=f"Download {prefix} Sensitivity CSV",
        data=df_sens.to_csv().encode(),
        file_name=f"{prefix}_Sensitivity_Matrix.csv",
        mime="text/csv"
    )

if "IRR FCF CSV" in option:
    st.download_button(
        label=f"Download {prefix} IRR Cash Flows (FCF) CSV",
        data=df_irr.to_csv(index=False).encode(),
        file_name=f"{prefix}_IRR_Cashflows_FCF.csv",
        mime="text/csv"
    )

if "IRR Net CSV" in option:
    st.download_button(
        label=f"Download {prefix} IRR Cash Flows (Net Profit) CSV",
        data=df_irr_net.to_csv(index=False).encode(),
        file_name=f"{prefix}_IRR_Cashflows_NetProfit.csv",
        mime="text/csv"
    )

if "Excel (Full Report)" in option:
    st.download_button(
        label=f"Download {prefix} Excel Report",
        data=excel_buffer,
        file_name=f"{prefix}_Valuation_Report_{dt.datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if "Charts (PNG)" in option:
    import tempfile
    tmpfile1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmpfile1.name, dpi=150, bbox_inches="tight")
    tmpfile2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig_irr.savefig(tmpfile2.name, dpi=150, bbox_inches="tight")
    with open(tmpfile1.name, "rb") as f1:
        st.download_button(f"Download {prefix} DCF Chart (PNG)", data=f1, file_name=f"{prefix}_DCF_Chart.png", mime="image/png")
    with open(tmpfile2.name, "rb") as f2:
        st.download_button(f"Download {prefix} IRR Chart (PNG)", data=f2, file_name=f"{prefix}_IRR_Chart.png", mime="image/png")
