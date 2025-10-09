# ===============================================================
# Multi-Company DCF + WACC Valuation App (PSS & MDKB) — v8 (Combined Add)
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

# A) Initial MPW Proposal
df_pss_A = pd.DataFrame({
    "Sales_kEUR":  [91828, 137455, 132071, 134417, 137106],
    "EBIT_kEUR":   [   547,  12908,  12934,  13329,  13329],
    "Net_kEUR":    [   297,  12123,   9133,   8719,   8719],
    "Equity_kEUR": [12596, 14731,  19219,  24750,  25850],
    "Cash_kEUR":   [12600, 28500, 42700, 53800, 62519],
    "FCF_kEUR":    [ 8500, 15900, 14200, 11100,  8719],
}, index=years_pss)

# C) CAGR 15% to 2029
df_pss_C = pd.DataFrame({
    "Sales_kEUR":  [55000, 63250, 72738, 83649, 96196],
    "EBIT_kEUR":   [  593,  1441,  3366,  5164,  5967],
    "Net_kEUR":    [  529,  1461,  3114,  3614,  4156],
    "Equity_kEUR": [12596, 13125, 14586, 17700, 21314],
    "Cash_kEUR":   [ 8176,  8591,  8775, 10182, 12706],
    "FCF_kEUR":    [  415,   184,  1407,  2524,  2922],
}, index=years_pss)

# D) CAGR 25% to 2029
df_pss_D = pd.DataFrame({
    "Sales_kEUR":  [55000, 68750, 85938, 107422, 134278],
    "EBIT_kEUR":   [  593,  1567,  3977,   6632,   8329],
    "Net_kEUR":    [  529,  1588,  3679,   4641,   5801],
    "Equity_kEUR": [12596, 13125, 14713,  18392,  23033],
    "Cash_kEUR":   [ 8176,  8591,  8313,   9378,  11872],
    "FCF_kEUR":    [  415,  -278,  1065,   2494,   3145],
}, index=years_pss)

# ------------------------
# MDKB inputs (mn EUR -> kEUR)
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
# SIDEBAR (adds Combined view; 4-decimal precision preserved)
# ------------------------
st.sidebar.markdown("### **Company Selection**")
company = st.sidebar.selectbox("**Select Company / View**",
                               ["PSS", "MDKB", "PSS + MDKB (Combined)"])

# PSS Scenario selector shown for PSS and Combined
pss_scenario_default = "B) Current Mid-term Forecast"
pss_scenario = pss_scenario_default
if company in ("PSS", "PSS + MDKB (Combined)"):
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
scenario_code = (pss_scenario or pss_scenario_default)[:1]

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
# default for MDKB vs PSS (same logic you had)
default_nwc = -0.4100 if company=="MDKB" else 0.1000
nwc_pct   = st.sidebar.number_input("ΔNWC % of ΔSales", value=float(default_nwc), step=0.0001, format="%.4f")
mdkb_extend_growth = st.sidebar.number_input("MDKB 2029 growth", value=0.0200, step=0.0001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Debt & Financing**")
debt = st.sidebar.number_input("Debt (€) — applied once to Combined Equity", value=0.0, step=1_000_000.0)
rd   = st.sidebar.number_input("Cost of Debt (Rd)", value=0.0400, step=0.0001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Acquisition & IRR Settings**")
assumed_price_mdkb = st.sidebar.number_input("Assumed Price for MDKB (€)", value=4_500_000.0, step=100_000.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### **FCF Source**")
# For Combined we apply the same source to both (keeps it simple & predictable)
fcf_source = st.sidebar.radio(
    "Choose FCF model:",
    ["Computed (from EBIT, Dep%, CapEx%, ΔNWC)", "Table (provided FCF, adjusted by drivers)"],
    index=(0 if company in ("PSS", "PSS + MDKB (Combined)") else 1)
)

# ------------------------
# DATA PREP HELPERS
# ------------------------
def get_pss_df_by_scenario(code:str)->pd.DataFrame:
    if   code == "A": return df_pss_A.copy()
    elif code == "B": return df_pss_B.copy()
    elif code == "C": return df_pss_C.copy()
    else:             return df_pss_D.copy()

def build_mdkb_df(growth_2029:float)->pd.DataFrame:
    s25_28,eb25_28,n25_28,fc25_28,c25_28=m_to_k(sales_m_25_28),m_to_k(ebit_m_25_28),m_to_k(net_m_25_28),m_to_k(fcf_m_25_28),m_to_k(cash_m_25_28)
    eq25_29,nwc25_29=m_to_k(equity_m_25_29),m_to_k(nwc_m_25_29)
    s28,eb28,n28,fc28,c28=s25_28[-1],eb25_28[-1],n25_28[-1],fc25_28[-1],c25_28[-1]
    ebm,nm=eb28/s28, n28/s28
    s29=int(round(s28*(1+growth_2029)))
    eb29=int(round(s29*ebm)); n29=int(round(s29*nm))
    fc29=int(round(fc28*(1+growth_2029))); c29=int(round(c28+fc29))
    df=pd.DataFrame({
        "Sales_kEUR":s25_28+[s29],
        "EBIT_kEUR":eb25_28+[eb29],
        "Net_kEUR": n25_28+[n29],
        "Equity_kEUR":eq25_29,
        "Cash_kEUR": c25_28+[c29],
        "FCF_kEUR":  fc25_28+[fc29],
        "NWC_kEUR":  nwc25_29
    },index=[2025,2026,2027,2028,2029])
    return df

def compute_company_result(df_raw:pd.DataFrame, *,
                           title:str,
                           fcf_source_choice:str,
                           rf, mrp, beta, tax, g,
                           dep_pct, capex_pct, use_nwc, nwc_pct,
                           rd, debt_for_equity_subtract:float,
                           respect_pss_final_fcf:bool,
                           scenario_code:str):
    """
    Returns a dict with:
      df (display df), years, arrays (sales, ebit, net, fcfs, pv_fcfs), EV, EqV, cash, tv, Re, WACC, dfres
      EqV_asset = EV + cash  (no debt subtracted; for Combined sum)
    """
    df = df_raw.copy()
    years = list(df.index)

    # Capital structure
    E = df["Equity_kEUR"].iloc[-1] * 1000.0
    D = debt_for_equity_subtract
    Re = capm_cost_equity(rf,mrp,beta)
    WACC = compute_wacc(E,D,Re,rd,tax)

    # Core series
    sales_eur=(df["Sales_kEUR"].values*1000).astype(float)
    ebit_eur =(df["EBIT_kEUR"].values*1000).astype(float)
    net_eur  =(df["Net_kEUR"].values*1000).astype(float)
    cash     = df["Cash_kEUR"].iloc[-1]*1000.0

    # FCF build
    if "Computed" in fcf_source_choice:
        fcfs=[]
        for i,y in enumerate(years):
            s=sales_eur[i]; prev=sales_eur[i-1] if i>0 else s
            e=ebit_eur[i]
            dep=s*dep_pct; capex=s*capex_pct
            dNWC=(s-prev)*nwc_pct if (use_nwc and i>0) else 0
            fcf=(e*(1-tax))+dep-capex-dNWC

            # Respect explicit plan only for PSS A/B 2029, like your prior logic
            if respect_pss_final_fcf and y==2029 and scenario_code in ("A","B") and "FCF_kEUR" in df.columns and not pd.isna(df.loc[y,"FCF_kEUR"]):
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

    # Valuation
    pv_fcfs = pv(fcfs, WACC)
    tv = fcfs[-1]*(1+g)/(WACC-g) if WACC>g else np.nan
    pv_tv = tv/((1+WACC)**len(fcfs)) if not np.isnan(tv) else 0
    EV = sum(pv_fcfs)+pv_tv
    EqV_asset = EV + cash                   # useful for Combined
    EqV = EqV_asset - D                     # stand-alone equity if we subtract debt here

    # Table for display
    dfres = pd.DataFrame({
        "Year": [str(y) for y in years],
        "Sales (€)": sales_eur,
        "EBIT (€)":  ebit_eur,
        "Net (€)":   net_eur,
        "FCF (€)":   fcfs,
        "PV(FCF)":   pv_fcfs,
    })

    return {
        "title": title, "df": df.reset_index().rename(columns={"index":"Year"}), "years": years,
        "sales": sales_eur, "ebit": ebit_eur, "net": net_eur, "fcfs": fcfs, "pv_fcfs": pv_fcfs,
        "EV": EV, "EqV": EqV, "EqV_asset": EqV_asset, "cash": cash, "tv": tv, "Re": Re, "WACC": WACC, "dfres": dfres
    }

# ------------------------
# PREPARE COMPANY/COMBINED RESULTS
# ------------------------
# Build PSS & MDKB objects (we’ll reuse whether visible or in Combined)
pss_df = get_pss_df_by_scenario(scenario_code)
mdkb_df = build_mdkb_df(mdkb_extend_growth)

# Individual results (for single views): subtract 'debt' when that company is selected,
# but for Combined we will NOT subtract at company level (we subtract once at group level).
pss_res_standalone = compute_company_result(
    pss_df, title="Power Service Solutions GmbH (PSS)", fcf_source_choice=fcf_source,
    rf=rf, mrp=mrp, beta=beta, tax=tax, g=g,
    dep_pct=dep_pct, capex_pct=capex_pct, use_nwc=use_nwc, nwc_pct=nwc_pct,
    rd=rd, debt_for_equity_subtract=(debt if company=="PSS" else 0.0),
    respect_pss_final_fcf=True, scenario_code=scenario_code
)
mdkb_res_standalone = compute_company_result(
    mdkb_df, title="MDKB GmbH", fcf_source_choice=fcf_source,
    rf=rf, mrp=mrp, beta=beta, tax=tax, g=g,
    dep_pct=dep_pct, capex_pct=capex_pct, use_nwc=use_nwc, nwc_pct=nwc_pct,
    rd=rd, debt_for_equity_subtract=(debt if company=="MDKB" else 0.0),
    respect_pss_final_fcf=False, scenario_code="X"
)

# Combined view: add PSS + MDKB
if company == "PSS + MDKB (Combined)":
    # Recompute company results with NO debt subtraction (we subtract once at group level)
    pss_res = compute_company_result(
        pss_df, title="Power Service Solutions GmbH (PSS)", fcf_source_choice=fcf_source,
        rf=rf, mrp=mrp, beta=beta, tax=tax, g=g,
        dep_pct=dep_pct, capex_pct=capex_pct, use_nwc=use_nwc, nwc_pct=nwc_pct,
        rd=rd, debt_for_equity_subtract=0.0,
        respect_pss_final_fcf=True, scenario_code=scenario_code
    )
    mdkb_res = compute_company_result(
        mdkb_df, title="MDKB GmbH", fcf_source_choice=fcf_source,
        rf=rf, mrp=mrp, beta=beta, tax=tax, g=g,
        dep_pct=dep_pct, capex_pct=capex_pct, use_nwc=use_nwc, nwc_pct=nwc_pct,
        rd=rd, debt_for_equity_subtract=0.0,
        respect_pss_final_fcf=False, scenario_code="X"
    )

    # Additive time series (assumes same years 2025–2029)
    years = pss_res["years"]
    sales_eur = pss_res["sales"] + mdkb_res["sales"]
    ebit_eur  = pss_res["ebit"]  + mdkb_res["ebit"]
    net_eur   = pss_res["net"]   + mdkb_res["net"]
    fcfs      = [pss_res["fcfs"][i] + mdkb_res["fcfs"][i] for i in range(len(years))]

    # With common WACC inputs we can (a) sum EVs or (b) recompute from combined FCFs.
    # We do both here and use the sum-of-parts (add EVs) as the displayed EV to stay literal about "adding them".
    EV = pss_res["EV"] + mdkb_res["EV"]
    cash_sum = pss_res["cash"] + mdkb_res["cash"]
    EqV = EV + cash_sum - debt   # subtract group debt once

    # For tables/plots we also build PV(FCF) using a representative WACC.
    # Use PSS WACC (same inputs anyway) — if you set different betas in future, switch to weighted WACC.
    WACC = pss_res["WACC"]
    Re   = pss_res["Re"]
    pv_fcfs = pv(fcfs, WACC)
    tv = fcfs[-1]*(1+g)/(WACC-g) if WACC>g else np.nan
    pv_tv = tv/((1+WACC)**len(fcfs)) if not np.isnan(tv) else 0

    # Display df and dfres (summed)
    # Union columns and sum by common names (Sales/EBIT/Net/Equity/Cash/FCF)
    pss_disp = pss_res["df"].copy()
    mdkb_disp = mdkb_res["df"].copy()
    # Ensure all possible cols exist
    for col in ["Sales_kEUR","EBIT_kEUR","Net_kEUR","Equity_kEUR","Cash_kEUR","FCF_kEUR"]:
        if col not in pss_disp.columns:  pss_disp[col]=0
        if col not in mdkb_disp.columns: mdkb_disp[col]=0
    comb_disp = pss_disp[["Year","Sales_kEUR","EBIT_kEUR","Net_kEUR","Equity_kEUR","Cash_kEUR","FCF_kEUR"]].copy()
    comb_disp.set_index("Year", inplace=True)
    m2 = mdkb_disp[["Year","Sales_kEUR","EBIT_kEUR","Net_kEUR","Equity_kEUR","Cash_kEUR","FCF_kEUR"]].copy().set_index("Year")
    comb_disp = comb_disp.add(m2, fill_value=0).reset_index()

    dfres = pd.DataFrame({
        "Year": [str(y) for y in years],
        "Sales (€)": sales_eur,
        "EBIT (€)":  ebit_eur,
        "Net (€)":   net_eur,
        "FCF (€)":   fcfs,
        "PV(FCF)":   pv_fcfs,
    })

    title = "Combined — PSS + MDKB"
else:
    # Single company branch: lift from the prepared standalones
    if company == "PSS":
        r = pss_res_standalone
    else:
        r = mdkb_res_standalone
    years = r["years"]; sales_eur=r["sales"]; ebit_eur=r["ebit"]; net_eur=r["net"]
    fcfs=r["fcfs"]; pv_fcfs=r["pv_fcfs"]; EV=r["EV"]; EqV=r["EqV"]; WACC=r["WACC"]; Re=r["Re"]
    tv=r["tv"]; title=r["title"]; dfres=r["dfres"]; comb_disp=r["df"]

# ------------------------
# HEADER
# ------------------------
st.title("💼 Corporate Valuation — DCF & WACC")
who = USER_NAMES.get(st.session_state['user'], st.session_state['user'])
scenario_note = f" | PSS Scenario: **{pss_scenario}**" if company in ("PSS","PSS + MDKB (Combined)") else ""
st.caption(f"Logged in as **{who}** — {dt.datetime.now():%H:%M}{scenario_note}")

if "Computed" in fcf_source:
    st.success("The model is currently using **Computed Free Cash Flow (FCFF)**. "
        "FCFF = EBIT × (1 − Tax) + Depreciation − CapEx − ΔNWC.")
else:
    st.info("The model is using **Table Free Cash Flow (Adjusted)** (plan FCF with light driver responsiveness).")

# ------------------------
# RESULTS (pretty table)
# ------------------------
st.subheader(f"Key Lines — {title}")
disp = comb_disp.copy()
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
st.caption("⚙️ Note: Combined view **adds** the two businesses. Debt is subtracted **once** at group level.")

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
plt.axhline(0,linewidth=.8); plt.legend(); plt.title(f"Free Cash Flow — {title}")
st.pyplot(fig)

# ------------------------
# IRR
# ------------------------
st.subheader("💰 IRR Analysis")
if company == "MDKB":
    init=-assumed_price_mdkb
    irr_cf=[(init if i==0 else 0)+fcfs[i]+(0 if i<len(years)-1 else (tv if not np.isnan(tv) else 0)) for i in range(len(years))]
    IRR_fcf=irr_bisection(irr_cf)
    irr_cf_net=[(init if i==0 else 0)+net_eur[i]+(0 if i<len(years)-1 else (tv if not np.isnan(tv) else 0)) for i in range(len(years))]
    IRR_net=irr_bisection(irr_cf_net)

elif company == "PSS":
    base=[(2025,500000),(2026,2500000),(2027,3500000),(2028,6800000)]
    adj,total=adjust_instalments_absolute_deduction(base,assumed_price_mdkb)
    irr_cf=[adj.get(y,0)+fcfs[i]+(0 if i<len(years)-1 else (tv if not np.isnan(tv) else 0)) for i,y in enumerate(years)]
    IRR_fcf=irr_bisection(irr_cf)
    eff=max(total-assumed_price_mdkb,0)
    irr_cf_net=[(-eff if i==0 else 0)+net_eur[i]+(0 if i<len(years)-1 else (tv if not np.isnan(tv) else 0)) for i,y in enumerate(years)]
    IRR_net=irr_bisection(irr_cf_net)

else:  # Combined — apply acquisition outflow once to the group
    base=[(2025,500000),(2026,2500000),(2027,3500000),(2028,6800000)]
    adj,total=adjust_instalments_absolute_deduction(base,assumed_price_mdkb)
    irr_cf=[adj.get(y,0)+fcfs[i]+(0 if i<len(years)-1 else (tv if not np.isnan(tv) else 0)) for i,y in enumerate(years)]
    IRR_fcf=irr_bisection(irr_cf)
    # Net-profit based IRR on combined net
    eff=max(total-assumed_price_mdkb,0)
    irr_cf_net=[(-eff if i==0 else 0)+net_eur[i]+(0 if i<len(years)-1 else (tv if not np.isnan(tv) else 0)) for i,y in enumerate(years)]
    IRR_net=irr_bisection(irr_cf_net)

col1,col2=st.columns(2)
col1.metric("IRR (FCF)",f"{IRR_fcf*100:.2f}%" if not np.isnan(IRR_fcf) else "N/A")
col2.metric("IRR (Net Profit)",f"{IRR_net*100:.2f}%" if not np.isnan(IRR_net) else "N/A")
fig2=plt.figure(figsize=(5.5,4))
plt.bar(["FCF","Net"],[IRR_fcf*100 if not np.isnan(IRR_fcf) else 0,
                       IRR_net*100 if not np.isnan(IRR_net) else 0])
plt.ylabel("%"); plt.title(f"IRR — {title}")
st.pyplot(fig2)

# ------------------------
# SENSITIVITY (on displayed view’s FCFs)
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
    fmt_title   = workbook.add_format({"bold": True, "font_size": 14, "align": "center", "valign": "vcenter"})
    fmt_header  = workbook.add_format({"bold": True, "bg_color": "#E1EAF5", "border": 1, "align": "center"})
    fmt_text    = workbook.add_format({"border": 1, "align": "left"})
    fmt_yeartxt = workbook.add_format({"border": 1, "align": "center"})
    fmt_pct     = workbook.add_format({"num_format": "0.00%", "border": 1, "align": "right"})
    fmt_euro    = workbook.add_format({
        "num_format": '_-€ * #,##0_-;[Red]-€ * #,##0_-;_-€ * "-"??_-;_-@_-',
        "border": 1, "align": "right"
    })
    fmt_text_right = workbook.add_format({"border":1, "align":"right"})
    fmt_num_simple = workbook.add_format({"border":1, "align":"right", "num_format":"0.00"})

    # ===============================================================
    # DCF RESULTS
    # ===============================================================
    dfres.to_excel(writer, sheet_name="DCF_Results", index=False, startrow=1)
    ws = writer.sheets["DCF_Results"]
    dcf_title = f"DCF Results — {title}" + (f" — {pss_scenario}" if company in ("PSS","PSS + MDKB (Combined)") else "")
    ws.merge_range(0, 0, 0, 5, dcf_title, fmt_title)
    ws.set_row(0, 24)
    for c, col in enumerate(dfres.columns):
        ws.write(1, c, col, fmt_header)
    nrows, ncols = dfres.shape
    for r in range(nrows):
        excel_row = 2 + r
        ws.write_string(excel_row, 0, str(dfres.iloc[r, 0]), fmt_yeartxt)
        for c in range(1, ncols):
            val = float(dfres.iloc[r, c])
            ws.write_number(excel_row, c, val, fmt_euro)
    ws.set_column("A:A", 10)
    ws.set_column("B:F", 20)
    ws.freeze_panes(2, 1)
    ws.autofilter(1, 0, 7, 5)

    # ===============================================================
    # SENSITIVITY
    # ===============================================================
    df_sens.to_excel(writer, sheet_name="Sensitivity", index=True, startrow=1)
    ws2 = writer.sheets["Sensitivity"]
    ws2.merge_range(0, 0, 0, 5, "Sensitivity Table — EV by WACC & Terminal Growth (g)", fmt_title)
    ws2.set_row(0, 24)
    sens_cols = list(df_sens.reset_index().columns)
    for c, col in enumerate(sens_cols):
        ws2.write(1, c, col, fmt_header)
    rcount, ccount = df_sens.shape
    for r in range(rcount):
        excel_row = 2 + r
        try:
            wacc_val = float(str(df_sens.index[r]).replace("%",""))/100.0
            ws2.write_number(excel_row, 0, wacc_val, fmt_pct)
        except Exception:
            pass
        for c in range(ccount):
            excel_col = 1 + c
            ws2.write_number(excel_row, excel_col, float(df_sens.iloc[r, c]), fmt_euro)
    ws2.set_column(0, ccount, 18)
    ws2.freeze_panes(2, 1)
    ws2.autofilter(1, 0, min(12, 1 + rcount), ccount)

    # ===============================================================
    # SUMMARY
    # ===============================================================
    summary_rows = [
        ("Company/View", company, "text"),
        ("Scenario (PSS)", (pss_scenario if company in ("PSS","PSS + MDKB (Combined)") else "—"), "text"),
        ("FCF Source", fcf_source, "text"),
        ("EV", EV, "euro"),
        ("Equity", EqV, "euro"),
        ("WACC", WACC, "pct"),
        ("Re", Re, "pct"),
        ("Rd", rd, "pct"),
        ("Risk-free rate", rf, "pct"),
        ("Market risk premium", mrp, "pct"),
        ("Beta", beta, "num"),
        ("Tax rate", tax, "pct"),
        ("Terminal growth (g)", g, "pct"),
        ("Debt (€) — subtracted once", debt, "euro"),
        ("Assumed Price MDKB (€)", assumed_price_mdkb, "euro"),
    ]
    # IRR shown as well
    try:
        summary_rows.insert(5, ("IRR (FCF)", IRR_fcf, "pct"))
        summary_rows.insert(6, ("IRR (Net)", IRR_net, "pct"))
    except NameError:
        pass

    ws3 = workbook.add_worksheet("Summary")
    ws3.merge_range(0, 0, 1, 0, "Summary & Assumptions", fmt_title)
    ws3.write(2, 0, "Metric", fmt_header)
    ws3.write(2, 1, title, fmt_header)
    row = 3
    for label, value, kind in summary_rows:
        ws3.write(row, 0, label, fmt_text)
        is_nan = False
        try:
            if np.isnan(float(value)): is_nan = True
        except (TypeError, ValueError): pass
        if is_nan:
            ws3.write_string(row, 1, "N/A", fmt_text_right)
        elif kind == "euro": ws3.write_number(row, 1, float(value), fmt_euro)
        elif kind == "pct":  ws3.write_number(row, 1, float(value), fmt_pct)
        elif kind == "num":  ws3.write_number(row, 1, float(value), fmt_num_simple)
        else: ws3.write_string(row, 1, str(value), fmt_text_right)
        row += 1
    ws3.set_column("A:A", 30)
    ws3.set_column("B:B", 28)
    ws3.freeze_panes(3, 0)
    ws3.autofilter(2, 0, min(22, row-1), 1)

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
    name = ("Combined" if company=="PSS + MDKB (Combined)" else company)
    local_path = os.path.join(out, f"{name}_Valuation_Report_{today_str}.xlsx")
    with open(local_path, "wb") as f:
        f.write(excel_buffer.getbuffer())
    st.success(f"✅ File saved locally:\n{local_path}")

# ---- Browser download ----
label_name = ("Combined" if company=="PSS + MDKB (Combined)" else company)
c2.download_button(
    label=f"⬇️ Download Full {label_name} Excel Report",
    data=excel_buffer,
    file_name=f"{label_name}_Valuation_Report_{today_str}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
