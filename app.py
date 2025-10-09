# ===============================================================
# Multi-Company DCF + WACC Valuation App (PSS & MDKB) — v7
# ===============================================================

import os, io, json, datetime as dt
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ------------------------
# LOGIN
# ------------------------
load_dotenv()
USERS = {
    os.getenv("USER1"): os.getenv("USER1_PWD"),
    os.getenv("USER2"): os.getenv("USER2_PWD"),
    os.getenv("USER3"): os.getenv("USER3_PWD"),
    os.getenv("USER4"): os.getenv("USER4_PWD"),
    os.getenv("USER5"): os.getenv("USER5_PWD"),
    os.getenv("USER6"): os.getenv("USER6_PWD"),
    os.getenv("USER7"): os.getenv("USER7_PWD"),
}
USER_NAMES = {
    "d.garcia": "Daniel Garcia Rey",
    "t.held": "Thomas Held",
    "b.arrieta": "Borja Arrieta",
    "m.peter": "Michel Peter",
    "c.bahn": "Cristoph Bahn",
    "tgv": "Tomas Garcia Villanueva",
    "l.thai": "Laura Thai"
}

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
            st.success(f"Welcome back {USER_NAMES.get(user, user).split()[0]}!")
        else:
            st.error("Invalid credentials.")
    st.stop()

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(page_title="PSS & MDKB Valuation", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
def ts_folder(root):
    path = os.path.join(root, dt.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    return path

# ------------------------
# BASE DATA — PSS scenarios hardcoded
# ------------------------
years_pss = [2025, 2026, 2027, 2028, 2029]

# B) Current Mid-term Forecast (your existing plan)
df_pss_B = pd.DataFrame({
    "Sales_kEUR":  [55650, 92408, 113161, 120765, 123180],
    "EBIT_kEUR":   [  600,  2106,   5237,   7456,   7641],
    "Net_kEUR":    [  535,  2135,   4845,   5218,   5322],
    "Equity_kEUR": [12596, 14731,  19219,  24750,  25850],
    "Cash_kEUR":   [ 8176, 11205,  20394,  28367,  36000],
    "FCF_kEUR":    [-6884,  2749,   9054,   7716,   5322],
}, index=years_pss)

# A) Initial MPW Proposal (91M Y1) — for now identical to B (this we need to replace later on, waiting for Thomas input)
df_pss_A = df_pss_B.copy()

# C) CAGR 15% to 2029 (55M Y1) — margins ~B; FCF numbers precomputed (drivers: Dep 1%, CapEx 1%, tax 30%, ΔNWC 10% of ΔSales)
df_pss_C = pd.DataFrame({
    "Sales_kEUR":  [55000, 63250, 72738, 83649, 96196],
    "EBIT_kEUR":   [  593,  1441,  3366,  5164,  5967],
    "Net_kEUR":    [  529,  1461,  3114,  3614,  4156],
    "Equity_kEUR": [12596, 13125, 14586, 17700, 21314],
    "Cash_kEUR":   [ 8176,  8591,  8775, 10182, 12706],
    "FCF_kEUR":    [  415,   184,  1407,  2524,  2922],
}, index=years_pss)

# D) CAGR 25% to 2029 (55M Y1)
df_pss_D = pd.DataFrame({
    "Sales_kEUR":  [55000, 68750, 85938, 107422, 134278],
    "EBIT_kEUR":   [  593,  1567,  3977,   6632,   8329],
    "Net_kEUR":    [  529,  1588,  3679,   4641,   5801],
    "Equity_kEUR": [12596, 13125, 14713,  18392,  23033],
    "Cash_kEUR":   [ 8176,  8591,  8313,   9378,  11872],
    "FCF_kEUR":    [  415,  -278,  1065,   2494,   3145],
}, index=years_pss)

# ------------------------
# MDKB inputs (here I'm using the excel file, forgot the name but MDKB data)
# ------------------------
sales_m_25_28  = [13.7, 11.7, 12.0, 12.3]
ebit_m_25_28   = [0.8,  0.9,  1.0,  1.0]
net_m_25_28    = [0.6,  0.7,  0.7,  0.7]
fcf_m_25_28    = [-0.7, 0.3,  0.4,  0.4]
cash_m_25_28   = [2.2,  2.5,  2.9,  3.3]
equity_m_25_29 = [12.1, 12.6, 13.3, 14.0, 14.6]
nwc_m_25_29    = [5.5,  6.4,  6.4,  6.3,  6.2]
def m_to_k(seq): return [int(round(v*1000)) for v in seq]

# ------------------------
# FUNCTIONS (IRR + helpers)
# ------------------------
def capm_cost_equity(rf,mrp,b): return rf + b*mrp
def compute_wacc(E,D,Re,Rd,t): return (E/(E+D))*Re + (D/(E+D))*Rd*(1-t) if (E+D)>0 else Re
def pv(v,r): return [x/((1+r)**(i+1)) for i,x in enumerate(v)]
def npv_from_rate(r,c): return sum(cf/((1+r)**t) for t,cf in enumerate(c))
def irr_bisection(c,low=-0.9999,high=10,tol=1e-8,max_iter=200):
    c=np.array(c,float)
    if not (np.any(c<0) and np.any(c>0)): return float("nan")
    f=lambda r: npv_from_rate(r,c)
    fl,fh=f(low),f(high); n=0
    while fl*fh>0 and high<1e6 and n<50: high*=2; fh=f(high); n+=1
    if fl*fh>0: return float("nan")
    for _ in range(max_iter):
        mid=(low+high)/2; fm=f(mid)
        if abs(fm)<tol: return mid
        if fl*fm<0: high,fh=mid,fm
        else: low,fl=mid,fm
    return mid

def adjust_instalments_absolute_deduction(base,ded):
    total=sum(a for _,a in base)
    ded=min(max(ded,0),total); rem=ded; adj=[]
    for y,a in reversed(base):
        cut=min(a,rem); adj.append((y,-(a-cut))); rem-=cut
    return dict(sorted(adj)), total

# ------------------------
# SIDEBAR (scenario selector right under Company; 4-decimal precision)
# ------------------------
st.sidebar.markdown("### **Company Selection**")
company = st.sidebar.selectbox("**Select Company**", ["PSS", "MDKB"])

# PSS-only scenario selector with your labels
pss_scenario_default = "B) Current Mid-term Forecast"
pss_scenario = pss_scenario_default
if company == "PSS":
    pss_scenario = st.sidebar.selectbox(
        "PSS Scenario",
        [
            "A) Initial MPW Proposal (91M Y1)",
            "B) Current Mid-term Forecast",
            "C) CAGR 15% to 2029 (55M Y1)",
            "D) CAGR 25% to 2029 (55M Y1)",
        ],
        index=1
    )
scenario_code = (pss_scenario or pss_scenario_default)[:1] if company == "PSS" else "X"

st.sidebar.markdown("---")
st.sidebar.markdown("### **Capital & Risk Assumptions**")
rf   = st.sidebar.number_input("Risk-free rate",        value=0.0270, step=0.0001, format="%.4f")
mrp  = st.sidebar.number_input("Market risk premium",   value=0.0400, step=0.0001, format="%.4f")
beta = st.sidebar.number_input("Beta",                  value=1.2000, step=0.0001, format="%.4f")
tax  = st.sidebar.number_input("Tax rate",              value=0.3000, step=0.0001, format="%.4f")
g    = st.sidebar.number_input("Terminal growth (g)",   value=0.0200, step=0.0001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Operational Assumptions**")
dep_pct   = st.sidebar.number_input("Depreciation % of Sales", value=0.0100, step=0.0001, format="%.4f")
capex_pct = st.sidebar.number_input("CapEx % of Sales",        value=0.0100, step=0.0001, format="%.4f")
use_nwc   = st.sidebar.checkbox("Include ΔNWC adjustment", value=True)
default_nwc = -0.4100 if company=="MDKB" else 0.1000
nwc_pct   = st.sidebar.number_input("ΔNWC % of ΔSales", value=float(default_nwc), step=0.0001, format="%.4f")
mdkb_extend_growth = st.sidebar.number_input("MDKB 2029 growth", value=0.0200, step=0.0001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Debt & Financing**")
debt = st.sidebar.number_input("Debt (€)", value=0.0, step=1_000_000.0)
rd   = st.sidebar.number_input("Cost of Debt (Rd)", value=0.0400, step=0.0001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Acquisition & IRR Settings**")
assumed_price_mdkb = st.sidebar.number_input("Assumed Price for MDKB (€)", value=4_500_000.0, step=100_000.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### **FCF Source**")
fcf_source = st.sidebar.radio(
    "Choose FCF model:",
    ["Computed (from EBIT, Dep%, CapEx%, ΔNWC)", "Table (provided FCF, adjusted by drivers)"],
    index=0 if company == "PSS" else 1
)

# ------------------------
# DATA PREP
# ------------------------
if company=="PSS":
    title="Power Service Solutions GmbH (PSS)"
    if   scenario_code == "A": df = df_pss_A.copy()
    elif scenario_code == "B": df = df_pss_B.copy()
    elif scenario_code == "C": df = df_pss_C.copy()
    else:                      df = df_pss_D.copy()
    years = list(df.index)
else:
    # MDKB (unchanged)
    s25_28,eb25_28,n25_28,fc25_28,c25_28=m_to_k(sales_m_25_28),m_to_k(ebit_m_25_28),m_to_k(net_m_25_28),m_to_k(fcf_m_25_28),m_to_k(cash_m_25_28)
    eq25_29,nwc25_29=m_to_k(equity_m_25_29),m_to_k(nwc_m_25_29)
    s28,eb28,n28,fc28,c28=s25_28[-1],eb25_28[-1],n25_28[-1],fc25_28[-1],c25_28[-1]
    ebm,nm=eb28/s28,n28/s28
    s29=int(round(s28*(1+mdkb_extend_growth)))
    eb29=int(round(s29*ebm));n29=int(round(s29*nm))
    fc29=int(round(fc28*(1+mdkb_extend_growth)));c29=int(round(c28+fc29))
    df=pd.DataFrame({
        "Sales_kEUR":s25_28+[s29],
        "EBIT_kEUR":eb25_28+[eb29],
        "Net_kEUR":n25_28+[n29],
        "Equity_kEUR":eq25_29,
        "Cash_kEUR":c25_28+[c29],
        "FCF_kEUR":fc25_28+[fc29],
        "NWC_kEUR":nwc25_29
    },index=[2025,2026,2027,2028,2029])
    title="MDKB GmbH"
    years=list(df.index)

# ------------------------
# CALCULATIONS
# ------------------------
E=df["Equity_kEUR"].iloc[-1]*1000;D=debt
Re=capm_cost_equity(rf,mrp,beta);WACC=compute_wacc(E,D,Re,rd,tax)
sales_eur=(df["Sales_kEUR"].values*1000).astype(float)
ebit_eur=(df["EBIT_kEUR"].values*1000).astype(float)
net_eur=(df["Net_kEUR"].values*1000).astype(float)
cash=df["Cash_kEUR"].iloc[-1]*1000

if "Computed" in fcf_source:
    fcfs=[]
    for i,y in enumerate(years):
        s=sales_eur[i]; prev=sales_eur[i-1] if i>0 else s
        e=ebit_eur[i]
        dep=s*dep_pct; capex=s*capex_pct
        dNWC=(s-prev)*nwc_pct if (use_nwc and i>0) else 0
        fcf=(e*(1-tax))+dep-capex-dNWC
        # Respect explicit plan value only for PSS A/B final year (as in your prior logic)
        if company=="PSS" and y==2029 and scenario_code in ("A","B") and "FCF_kEUR" in df.columns and not pd.isna(df.loc[y,"FCF_kEUR"]):
            val=df.loc[y,"FCF_kEUR"]*1000.0
            if abs(val)>0: fcf=val
        fcfs.append(fcf)
else:
    base_fcfs=(df["FCF_kEUR"].values*1000).astype(float)
    adj=[]
    for i,y in enumerate(years):
        s=sales_eur[i]; prev=sales_eur[i-1] if i>0 else s
        e=ebit_eur[i]
        dep=s*dep_pct; capex=s*capex_pct
        dNWC=(s-prev)*nwc_pct if (use_nwc and i>0) else 0
        driver=(e*(1-tax))+dep-capex-dNWC
        adj_fcf=base_fcfs[i]+0.1*(driver-(e*(1-tax)))
        adj.append(adj_fcf)
    fcfs=adj

pv_fcfs=pv(fcfs,WACC)
tv=fcfs[-1]*(1+g)/(WACC-g) if WACC>g else np.nan
pv_tv=tv/((1+WACC)**len(fcfs)) if not np.isnan(tv) else 0
EV=sum(pv_fcfs)+pv_tv;EqV=EV+cash-D

# ------------------------
# HEADER
# ------------------------
st.title("💼 Corporate Valuation — DCF & WACC")
who = USER_NAMES.get(st.session_state['user'], st.session_state['user'])
scenario_note = f" | Scenario: **{pss_scenario}**" if company=="PSS" else ""
st.caption(f"Logged in as **{who}** — {dt.datetime.now():%H:%M}{scenario_note}")

if "Computed" in fcf_source:
    st.success("The model is currently using **Computed Free Cash Flow (FCFF)**. "
        "FCFF = EBIT × (1 − Tax) + Depreciation − CapEx − ΔNWC.")
else:
    st.info("The model is using **Table Free Cash Flow (Adjusted)** (plan FCF with light driver responsiveness).")

# ------------------------
# RESULTS (single Year column; no duplicate index)
# ------------------------
st.subheader(f"Key Lines — {title}")
disp = df.copy().reset_index().rename(columns={"index":"Year"})
disp["Year"] = disp["Year"].astype(str)
try:
    st.dataframe(
        disp.style.format({c: "{:,.0f}" for c in disp.columns if c.endswith("_kEUR")}),
        use_container_width=True,
        hide_index=True,
    )
except TypeError:
    st.dataframe(disp.set_index("Year").style.format({c: "{:,.0f}" for c in disp.columns if c.endswith("_kEUR")}),
                 use_container_width=True)

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Re",f"{Re*100:.2f}%"); c2.metric("Rd",f"{rd*100:.2f}%")
c3.metric("WACC",f"{WACC*100:.2f}%"); c4.metric("EV",f"€{EV:,.0f}"); c5.metric("Equity",f"€{EqV:,.0f}")
st.caption("⚙️ Note: Changing the FCF model modifies underlying cashflows, hence EV & Equity differ accordingly.")

dfres = pd.DataFrame({
    "Year": [str(y) for y in years],
    "Sales (€)": sales_eur,
    "EBIT (€)":  ebit_eur,
    "Net (€)":   net_eur,
    "FCF (€)":   fcfs,
    "PV(FCF)":   pv_fcfs,
})
try:
    st.dataframe(
        dfres.style.format({c: "€{:,.0f}" for c in dfres.columns if c != "Year"}),
        use_container_width=True,
        hide_index=True,
    )
except TypeError:
    st.dataframe(dfres.set_index("Year").style.format({c: "€{:,.0f}" for c in dfres.columns}),
                 use_container_width=True)

fig=plt.figure(figsize=(9,4.5))
plt.plot(years,fcfs,"o-",label="FCF"); plt.plot(years,pv_fcfs,"o-",label="PV(FCF)")
plt.axhline(0,linewidth=.8); plt.legend(); plt.title(f"Free Cash Flow — {company}")
st.pyplot(fig)

# ------------------------
# IRR (Same method as first script, since Thomas says okay)
# ------------------------
st.subheader("💰 IRR Analysis")
if company=="PSS":
    base=[(2025,500000),(2026,2500000),(2027,3500000),(2028,6800000)]
    adj,total=adjust_instalments_absolute_deduction(base,assumed_price_mdkb)
    irr_cf=[adj.get(y,0)+fcfs[i]+(tv if i==len(years)-1 else 0) for i,y in enumerate(years)]
    IRR_fcf=irr_bisection(irr_cf)
    eff=max(total-assumed_price_mdkb,0)
    irr_cf_net=[(-eff if i==0 else 0)+net_eur[i]+(tv if i==len(years)-1 else 0) for i,y in enumerate(years)]
    IRR_net=irr_bisection(irr_cf_net)
else:
    init=-assumed_price_mdkb
    irr_cf=[(init if i==0 else 0)+fcfs[i]+(tv if i==len(years)-1 else 0) for i in range(len(years))]
    IRR_fcf=irr_bisection(irr_cf)
    irr_cf_net=[(init if i==0 else 0)+net_eur[i]+(tv if i==len(years)-1 else 0) for i in range(len(years))]
    IRR_net=irr_bisection(irr_cf_net)

col1,col2=st.columns(2)
col1.metric("IRR (FCF)",f"{IRR_fcf*100:.2f}%" if not np.isnan(IRR_fcf) else "N/A")
col2.metric("IRR (Net Profit)",f"{IRR_net*100:.2f}%" if not np.isnan(IRR_net) else "N/A")
fig2=plt.figure(figsize=(5.5,4))
plt.bar(["FCF","Net"],[IRR_fcf*100 if not np.isnan(IRR_fcf) else 0,
                       IRR_net*100 if not np.isnan(IRR_net) else 0])
plt.ylabel("%"); plt.title(f"IRR — {company}")
st.pyplot(fig2)

# ------------------------
# SENSITIVITY
# ------------------------
st.markdown("### **📊 Sensitivity: Enterprise Value by WACC & Terminal Growth (g)**")
wr=np.arange(max(0.05,WACC-0.02),WACC+0.025,0.005)
gr=np.arange(g-0.01,g+0.015,0.005)
mt=[[sum(pv(fcfs,w))+(fcfs[-1]*(1+gg)/(w-gg)/((1+w)**len(fcfs)) if w>gg else 0) for gg in gr] for w in wr]
df_sens=pd.DataFrame(mt,index=[f"{w*100:.1f}%" for w in wr],columns=[f"{x*100:.1f}%" for x in gr])
st.dataframe(df_sens.style.format("€{:,.0f}"),use_container_width=True)

# ===============================================================
# EXPORT — Fully Formatted Pretty Excel (Local + Browser)
# ===============================================================
import xlsxwriter

st.markdown("### 📦 Export Options")

excel_buffer = io.BytesIO()
today_str = f"{dt.datetime.now():%Y%m%d}"

with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    workbook = writer.book

    # ---------- FORMATS ----------
    fmt_header  = workbook.add_format({"bold": True, "bg_color": "#E1EAF5", "border": 1, "align": "center"})
    fmt_text    = workbook.add_format({"border": 1, "align": "left"})
    fmt_center  = workbook.add_format({"align": "center", "valign": "vcenter"})
    fmt_year    = workbook.add_format({"num_format": "0", "border": 1, "align": "center"})
    fmt_euro    = workbook.add_format({"num_format": '€#,##0;[Red]-€#,##0', "border": 1, "align": "right"})
    fmt_percent = workbook.add_format({"num_format": "0.00%", "border": 1, "align": "right"})
    fmt_title   = workbook.add_format({"bold": True, "font_size": 14, "align": "center", "valign": "vcenter"})
    fmt_textcell= workbook.add_format({"border": 1, "align": "right"})

    # ===============================================================
    # DCF RESULTS SHEET
    # ===============================================================
    dfres.to_excel(writer, sheet_name="DCF_Results", index=False, startrow=1)
    ws = writer.sheets["DCF_Results"]
    ws.merge_range("A1:F1",
                   f"DCF Results — {title} ({pss_scenario if company=='PSS' else 'Base'})",
                   fmt_title)

    # Header row (row 2)
    for col_num, col in enumerate(dfres.columns):
        ws.write(1, col_num, col, fmt_header)

    # Apply formatting
    ws.set_column("A:A", 8, fmt_year)
    ws.set_column("B:F", 20, fmt_euro)
    ws.freeze_panes(2, 1)

    # Table visual boundary — clear rows beyond row 7
    ws.set_row(7, None, None)  # ensures row7 included as table end
    ws.autofilter(1, 0, 7, 5)

    # Auto-fit all columns dynamically
    for i, col in enumerate(dfres.columns):
        max_len = max(dfres[col].astype(str).map(len).max(), len(col)) + 2
        ws.set_column(i, i, max_len)

    # ===============================================================
    # SENSITIVITY SHEET
    # ===============================================================
    df_sens.to_excel(writer, sheet_name="Sensitivity", index=True, startrow=1)
    ws2 = writer.sheets["Sensitivity"]
    ws2.merge_range("A1:F1", "Sensitivity Table — EV by WACC & Terminal Growth (g)", fmt_title)

    for col_num, col in enumerate(df_sens.reset_index().columns):
        ws2.write(1, col_num, col, fmt_header)
    ws2.set_column(0, len(df_sens.columns), 18, fmt_euro)
    ws2.freeze_panes(2, 1)
    ws2.set_row(12, None, None)   # finish table at row 12

    for i, col in enumerate(df_sens.reset_index().columns):
        max_len = max(df_sens.reset_index()[col].astype(str).map(len).max(), len(col)) + 2
        ws2.set_column(i, i, max_len)

    # ===============================================================
    # SUMMARY SHEET
    # ===============================================================
    summary_data = pd.DataFrame({
        "Metric": ["Company", "Scenario", "FCF Source", "EV", "Equity",
                   "IRR (FCF)", "IRR (Net)", "WACC", "Re", "Rd",
                   "Risk-free rate", "Market risk premium", "Beta", "Tax rate",
                   "Terminal growth (g)", "Debt (€)", "Assumed Price MDKB (€)"],
        "Value": [company, (pss_scenario if company=="PSS" else "MDKB Base"), fcf_source,
                  EV, EqV, IRR_fcf, IRR_net, WACC, Re, rd,
                  rf, mrp, beta, tax, g, debt, assumed_price_mdkb]
    })

    summary_data.to_excel(writer, sheet_name="Summary", index=False, startrow=2)
    ws3 = writer.sheets["Summary"]
    ws3.merge_range("A1:A2", "Summary & Assumptions", fmt_title)

    # Apply column formats: all text right-aligned in B
    for row in range(2, len(summary_data) + 2):
        metric = summary_data.iloc[row - 2, 0]
        val = summary_data.iloc[row - 2, 1]
        if any(k in metric.lower() for k in ["rate", "growth"]) or metric in ["WACC","Re","Rd","IRR (FCF)","IRR (Net)"]:
            ws3.write(row, 1, val, fmt_percent)
        elif "€" in metric or metric in ["EV", "Equity", "Debt (€)", "Assumed Price MDKB (€)"]:
            ws3.write(row, 1, val, fmt_euro)
        else:
            ws3.write(row, 1, str(val), fmt_textcell)

    ws3.set_column("A:A", 30, fmt_text)
    ws3.set_column("B:B", 25)
    ws3.freeze_panes(3, 0)
    ws3.set_row(19, None, None)   # table ends at row 19

    # Auto-fit columns
    for i, col in enumerate(summary_data.columns):
        max_len = max(summary_data[col].astype(str).map(len).max(), len(col)) + 2
        ws3.set_column(i, i, max_len)

    # ===============================================================
    # CHARTS
    # ===============================================================
    ws_chart_dcf = workbook.add_worksheet("Chart_DCF")
    ws_chart_irr = workbook.add_worksheet("Chart_IRR")
    img1 = io.BytesIO(); fig.savefig(img1, format="png", dpi=150); img1.seek(0)
    img2 = io.BytesIO(); fig2.savefig(img2, format="png", dpi=150); img2.seek(0)
    ws_chart_dcf.insert_image("B2", "DCF.png", {"image_data": img1})
    ws_chart_irr.insert_image("B2", "IRR.png", {"image_data": img2})

    workbook.close()

# Reset pointer
excel_buffer.seek(0)

# 2 OPTIONS (same workbook)
c1, c2 = st.columns(2)

# ---- Save locally ----
if c1.button("💾 Save Excel Locally (VSCode User)"):
    out = ts_folder(RESULTS_DIR)
    local_path = os.path.join(out, f"{company}_Valuation_Report_{today_str}.xlsx")
    with open(local_path, "wb") as f:
        f.write(excel_buffer.getbuffer())
    st.success(f"✅ File saved locally:\n{local_path}")

# ---- Browser download ----
    c2.download_button(
        label=f"⬇️ Download Full {company} Excel Report",
        data=excel_buffer,
        file_name=f"{company}_Valuation_Report_{today_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
