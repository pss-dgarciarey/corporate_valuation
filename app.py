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
    "tgv": "Tomas Garcia Villanueva"
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
rf = st.sidebar.number_input("Risk-free rate (Rf)", value=0.027, step=0.001)
mrp = st.sidebar.number_input("Market risk premium (MRP)", value=0.04, step=0.001)
beta = st.sidebar.number_input("Equity beta (β)", value=1.2, step=0.05)
tax = st.sidebar.number_input("Tax rate (T)", value=0.30, step=0.01)
g = st.sidebar.number_input("Terminal growth (g)", value=0.02, step=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Operational Assumptions")
dep_pct = st.sidebar.number_input("Depreciation % of Sales", value=0.01, step=0.001)
capex_pct = st.sidebar.number_input("CapEx % of Sales", value=0.01, step=0.001)
use_nwc = st.sidebar.checkbox("Include ΔNWC adjustment", value=True)
nwc_pct = 0.03
if use_nwc:
    nwc_pct = st.sidebar.number_input("ΔNWC % of ΔSales", value=0.03, step=0.005)
sales_growth = st.sidebar.number_input("Sales growth for 2029", value=0.02, step=0.005)

st.sidebar.markdown("---")
st.sidebar.header("Debt & Financing")
debt_amount = st.sidebar.number_input("Debt (€)", value=0.0, step=1_000_000.0)
rd = st.sidebar.number_input("Cost of Debt (Rd)", value=0.04, step=0.005)

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
    fcfs.append(fcf)

pv_fcfs = pv(fcfs, WACC)
tv = fcfs[-1] * (1 + g) / (WACC - g) if WACC > g else np.nan
pv_tv = tv / ((1 + WACC) ** len(fcfs)) if not np.isnan(tv) else 0
EV = sum(pv_fcfs) + pv_tv
cash = df_base["Cash_kEUR"].iloc[-1] * 1000
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

# ------------------------
# CHART
# ------------------------
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
        ev_test = sum(pv(fcfs, w)) + (tv_test / ((1 + w) ** len(fcfs)))
        row.append(ev_test)
    matrix.append(row)

df_sens = pd.DataFrame(matrix, index=[f"{x*100:.1f}%" for x in wacc_range],
                       columns=[f"{y*100:.1f}%" for y in g_range])
st.dataframe(df_sens.style.format("€{:,.0f}"), use_container_width=True)

# ------------------------
# EXPORT OPTIONS
# ------------------------
st.markdown("### 📦 Export Options")
st.info(
    "Choose how to export your results:\n\n"
    "- **Local Save**: creates a timestamped folder under `results/` (for VSCode/Desktop use)\n"
    "- **Browser Download**: generates downloadable files directly from your browser."
)

# Local export
if st.button("💾 Export locally"):
    out = ts_folder(RESULTS_DIR)
    df_results.to_csv(os.path.join(out, "valuation_summary.csv"), index=False)
    df_sens.to_csv(os.path.join(out, "sensitivity_matrix.csv"))
    with open(os.path.join(out, "assumptions.json"), "w") as f:
        json.dump({
            "rf": rf, "mrp": mrp, "beta": beta,
            "Re": Re, "Rd": rd, "tax": tax, "g": g,
            "dep_pct": dep_pct, "capex_pct": capex_pct,
            "use_nwc": use_nwc, "nwc_pct": nwc_pct,
            "debt": D, "EV": EV, "EquityValue": equity_value,
            "WACC": WACC, "fcf": fcfs, "pv_fcfs": pv_fcfs, "pv_tv": pv_tv
        }, f, indent=2)
    fig.savefig(os.path.join(out, "DCF_chart.png"), dpi=150, bbox_inches="tight")
    st.success(f"✅ Exported locally to: {out}")

# Browser export
st.markdown("#### ⬇️ Download files to your device")
option = st.multiselect(
    "Select what to download:",
    ["Summary CSV", "Sensitivity CSV", "Excel (Full Report)", "Chart (PNG)"],
    default=["Excel (Full Report)"]
)

# Excel report (with assumptions and results)
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    df_summary = pd.DataFrame({
        "Metric": [
            "Enterprise Value (EV)", "Equity Value",
            "Cost of Equity (Re)", "Cost of Debt (Rd)", "WACC",
            "Risk-free rate (Rf)", "Market risk premium (MRP)",
            "Beta (β)", "Tax rate (T)", "Terminal growth (g)",
            "Depreciation % of Sales", "CapEx % of Sales", "ΔNWC % of ΔSales",
            "Debt (D)", "Equity (E)", "Include ΔNWC?", "Cash (final year)"
        ],
        "Value": [
            f"€{EV:,.0f}", f"€{equity_value:,.0f}",
            f"{Re*100:.2f}%", f"{rd*100:.2f}%", f"{WACC*100:.2f}%",
            f"{rf*100:.2f}%", f"{mrp*100:.2f}%", beta,
            f"{tax*100:.2f}%", f"{g*100:.2f}%", f"{dep_pct*100:.2f}%",
            f"{capex_pct*100:.2f}%", f"{nwc_pct*100:.2f}%", f"€{D:,.0f}",
            f"€{E:,.0f}", "Yes" if use_nwc else "No", f"€{cash:,.0f}"
        ]
    })
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    df_results.to_excel(writer, sheet_name="DCF_Results", index=False)
    df_sens.to_excel(writer, sheet_name="Sensitivity", index=True)
excel_buffer.seek(0)

if "Summary CSV" in option:
    st.download_button(
        label="Download Summary CSV",
        data=df_results.to_csv(index=False).encode(),
        file_name="PSS_Valuation_Summary.csv",
        mime="text/csv"
    )

if "Sensitivity CSV" in option:
    st.download_button(
        label="Download Sensitivity CSV",
        data=df_sens.to_csv().encode(),
        file_name="PSS_Sensitivity_Matrix.csv",
        mime="text/csv"
    )

if "Excel (Full Report)" in option:
    st.download_button(
        label="Download Excel Report (with Assumptions)",
        data=excel_buffer,
        file_name=f"PSS_Valuation_Report_{dt.datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if "Chart (PNG)" in option:
    import tempfile
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmpfile.name, dpi=150, bbox_inches="tight")
    with open(tmpfile.name, "rb") as f:
        st.download_button(
            label="Download DCF Chart (PNG)",
            data=f,
            file_name="DCF_Chart.png",
            mime="image/png"
        )
