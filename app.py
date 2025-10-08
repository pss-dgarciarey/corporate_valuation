# ===============================================================
# Multi-Company DCF + WACC Valuation App (PSS & MDKB)
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
}
USER_NAMES = {
    "d.garcia": "Daniel Garcia Rey",
    "t.held": "Thomas Held",
    "b.arrieta": "Borja Arrieta",
    "m.peter": "Michel Peter",
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
# DATASETS
# ------------------------
# --- PSS (kEUR)
years_pss = [2025, 2026, 2027, 2028, 2029]
df_pss = pd.DataFrame({
    "Sales_kEUR": [55650, 92408, 113161, 120765, 123180],
    "EBIT_kEUR":  [600, 2106, 5237, 7456, 7641],
    "Net_kEUR":   [535, 2135, 4845, 5218, 5322],
    "Equity_kEUR":[12596, 14731, 19219, 24750, 25850],
    "Cash_kEUR":  [8176, 11205, 20394, 28367, 36000],
    "FCF_kEUR":   [-6884, 2749, 9054, 7716, 5322],
}, index=years_pss)

# --- MDKB (EUR m → kEUR)
sales_m_25_28 = [13.7, 11.7, 12.0, 12.3]
ebit_m_25_28  = [0.8,  0.9,  1.0,  1.0]
net_m_25_28   = [0.6,  0.7,  0.7,  0.7]
fcf_m_25_28   = [-0.7, 0.3, 0.4, 0.4]
cash_m_25_28  = [2.2,  2.5,  2.9,  3.3]
equity_m_25_29 = [12.1, 12.6, 13.3, 14.0, 14.6]
nwc_m_25_29    = [5.5,  6.4,  6.4,  6.3,  6.2]

def m_to_k(seq): return [int(round(v*1000)) for v in seq]

# ------------------------
# FUNCTIONS
# ------------------------
def capm_cost_equity(rf,mrp,b): return rf+b*mrp
def compute_wacc(E,D,Re,Rd,t): return (E/(E+D))*Re+(D/(E+D))*Rd*(1-t) if (E+D)>0 else Re
def pv(v,r): return [x/((1+r)**(i+1)) for i,x in enumerate(v)]
def npv_from_rate(r,c): return sum(cf/((1+r)**t) for t,cf in enumerate(c))
def irr_bisection(c,low=-0.9999,high=10,tol=1e-8):
    c=np.array(c,float)
    if not(np.any(c<0) and np.any(c>0)):return float("nan")
    f=lambda r:npv_from_rate(r,c)
    fl,fh=f(low),f(high)
    n=0
    while fl*fh>0 and high<1e6 and n<50:high*=2;fh=f(high);n+=1
    if fl*fh>0:return float("nan")
    for _ in range(200):
        mid=(low+high)/2;fm=f(mid)
        if abs(fm)<tol:return mid
        if fl*fm<0:high,fh=mid,fm
        else:low,fl=mid,fm
    return mid
def adjust_instalments_absolute_deduction(base,ded):
    total=sum(a for _,a in base)
    ded=min(max(ded,0),total);rem=ded;adj=[]
    for y,a in reversed(base):
        red=min(a,rem);adj.append((y,-(a-red)));rem-=red
    return dict(sorted(adj)),total

# ------------------------
# SIDEBAR
# ------------------------
# ------------------------
# SIDEBAR (grouped + bold)
# ------------------------
st.sidebar.markdown("### **Company Selection**")
company = st.sidebar.selectbox("**Select Company**", ["PSS", "MDKB"])

st.sidebar.markdown("---")
st.sidebar.markdown("### **Capital & Risk Assumptions**")
rf = st.sidebar.number_input("Risk-free rate (Rf)", value=0.027, step=0.001)
mrp = st.sidebar.number_input("Market risk premium (MRP)", value=0.04, step=0.001)
beta = st.sidebar.number_input("Equity beta (β)", value=1.2, step=0.05)
tax = st.sidebar.number_input("Tax rate (T)", value=0.30, step=0.01)
g = st.sidebar.number_input("Terminal growth (g)", value=0.02, step=0.001)

st.sidebar.markdown("---")
st.sidebar.markdown("### **Operational Assumptions**")
dep_pct = st.sidebar.number_input("Depreciation % of Sales", value=0.01, step=0.001)
capex_pct = st.sidebar.number_input("CapEx % of Sales", value=0.01, step=0.001)
use_nwc = st.sidebar.checkbox("Include ΔNWC adjustment (if no FCF)", value=True)
default_nwc = -0.41 if company == "MDKB" else 0.10
nwc_pct = st.sidebar.number_input("ΔNWC % of ΔSales", value=float(default_nwc), step=0.01)
mdkb_extend_growth = st.sidebar.number_input("MDKB 2029 growth (Sales & FCF)", value=0.02, step=0.005)

st.sidebar.markdown("---")
st.sidebar.markdown("### **Debt & Financing**")
debt = st.sidebar.number_input("Debt (€)", value=0.0, step=1_000_000.0)
rd = st.sidebar.number_input("Cost of Debt (Rd)", value=0.04, step=0.005)

st.sidebar.markdown("---")
st.sidebar.markdown("### **Acquisition & IRR Settings**")
assumed_price_mdkb = st.sidebar.number_input("Assumed Price for MDKB (€)", value=0.0, step=100_000.0)

# ------------------------
# BUILD DATA
# ------------------------
if company=="PSS":
    df=df_pss.copy();title="Power Service Solutions GmbH (PSS)"
else:
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
# DCF
# ------------------------
E=df["Equity_kEUR"].iloc[-1]*1000;D=debt
Re=capm_cost_equity(rf,mrp,beta);WACC=compute_wacc(E,D,Re,rd,tax)
sales,ebit,net,fcf=(df[c].values*1000 for c in["Sales_kEUR","EBIT_kEUR","Net_kEUR","FCF_kEUR"])
cash=df["Cash_kEUR"].iloc[-1]*1000
fcfs=fcf.tolist()
pv_fcfs=pv(fcfs,WACC)
tv=fcfs[-1]*(1+g)/(WACC-g) if WACC>g else np.nan
pv_tv=tv/((1+WACC)**len(fcfs)) if not np.isnan(tv) else 0
EV=sum(pv_fcfs)+pv_tv;EqV=EV+cash-D

# ------------------------
# DISPLAY
# ------------------------
st.subheader(f"Key Lines ({years[0]}–{years[-1]}) — {title}")
disp=df.copy();disp.insert(0,"Year",[str(y) for y in years])
st.dataframe(
    disp.style.format({c:"{:,.0f}" for c in disp.columns if c.endswith("_kEUR")}),
    use_container_width=True
)
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Re",f"{Re*100:.2f}%");c2.metric("Rd",f"{rd*100:.2f}%")
c3.metric("WACC",f"{WACC*100:.2f}%");c4.metric("EV",f"€{EV:,.0f}");c5.metric("Equity",f"€{EqV:,.0f}")

dfres=pd.DataFrame({
    "Year":[str(y) for y in years],
    "Sales (€)":sales,"EBIT (€)":ebit,"Net (€)":net,"FCF (€)":fcfs,"PV(FCF)":pv_fcfs
})
st.dataframe(
    dfres.style.format({c:"€{:,.0f}" for c in dfres.columns if c!="Year"}),
    use_container_width=True
)
fig=plt.figure(figsize=(9,4.5))
plt.plot(years,fcfs,"o-",label="FCF");plt.plot(years,pv_fcfs,"o-",label="PV(FCF)")
plt.axhline(0,color="gray",lw=.8);plt.legend();plt.title(f"Free Cash Flow — {company}");st.pyplot(fig)

# ------------------------
# IRR
# ------------------------
st.subheader("💰 IRR Analysis")
if company=="PSS":
    base=[(2025,500000),(2026,2500000),(2027,3500000),(2028,6800000)]
    adj,total=adjust_instalments_absolute_deduction(base,assumed_price_mdkb)
    irr_cf=[adj.get(y,0)+fcfs[i]+(tv if i==len(years)-1 else 0) for i,y in enumerate(years)]
    IRR_fcf=irr_bisection(irr_cf)
    eff=max(total-assumed_price_mdkb,0)
    irr_cf_net=[(-eff if i==0 else 0)+net[i]+(tv if i==len(years)-1 else 0) for i,y in enumerate(years)]
    IRR_net=irr_bisection(irr_cf_net)
else:
    init=-assumed_price_mdkb
    irr_cf=[(init if i==0 else 0)+fcfs[i]+(tv if i==len(years)-1 else 0) for i in range(len(years))]
    IRR_fcf=irr_bisection(irr_cf)
    irr_cf_net=[(init if i==0 else 0)+net[i]+(tv if i==len(years)-1 else 0) for i in range(len(years))]
    IRR_net=irr_bisection(irr_cf_net)

col1,col2=st.columns(2)
col1.metric("IRR (FCF)",f"{IRR_fcf*100:.2f}%" if not np.isnan(IRR_fcf) else "N/A")
col2.metric("IRR (Net Profit)",f"{IRR_net*100:.2f}%" if not np.isnan(IRR_net) else "N/A")
fig2=plt.figure(figsize=(5.5,4))
plt.bar(["FCF","Net"],[IRR_fcf*100,IRR_net*100]);plt.ylabel("%");plt.title(f"IRR — {company}")
for i,v in enumerate([IRR_fcf*100,IRR_net*100]):
    if not np.isnan(v): plt.text(i,v+0.3,f"{v:.2f}%",ha="center")
st.pyplot(fig2)

# ------------------------
# SENSITIVITY
# ------------------------
st.subheader("📊 Sensitivity EV by WACC & g")
wr=np.arange(max(0.05,WACC-0.02),WACC+0.025,0.005)
gr=np.arange(g-0.01,g+0.015,0.005)
mt=[[sum(pv(fcfs,w))+ (fcfs[-1]*(1+gg)/(w-gg)/((1+w)**len(fcfs)) if w>gg else 0)
     for gg in gr] for w in wr]
st.dataframe(
    pd.DataFrame(mt,index=[f"{w*100:.1f}%" for w in wr],columns=[f"{x*100:.1f}%" for x in gr])
    .style.format("€{:,.0f}"), use_container_width=True)
