# ===============================================================
# Multi-Company DCF + WACC Valuation App (PSS & MDKB) — Dynamic v3
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
# BASE DATA
# ------------------------
years_pss = [2025, 2026, 2027, 2028, 2029]
df_pss = pd.DataFrame({
    "Sales_kEUR": [55650, 92408, 113161, 120765, 123180],
    "EBIT_kEUR":  [600, 2106, 5237, 7456, 7641],
    "Net_kEUR":   [535, 2135, 4845, 5218, 5322],
    "Equity_kEUR":[12596, 14731, 19219, 24750, 25850],
    "Cash_kEUR":  [8176, 11205, 20394, 28367, 36000],
    "FCF_kEUR":   [-6884, 2749, 9054, 7716, 5322],
}, index=years_pss)

sales_m_25_28 = [13.7, 11.7, 12.0, 12.3]
ebit_m_25_28  = [0.8, 0.9, 1.0, 1.0]
net_m_25_28   = [0.6, 0.7, 0.7, 0.7]
fcf_m_25_28   = [-0.7, 0.3, 0.4, 0.4]
cash_m_25_28  = [2.2, 2.5, 2.9, 3.3]
equity_m_25_29 = [12.1, 12.6, 13.3, 14.0, 14.6]
nwc_m_25_29    = [5.5, 6.4, 6.4, 6.3, 6.2]
def m_to_k(seq): return [int(round(v*1000)) for v in seq]

# ------------------------
# FUNCTIONS
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
# SIDEBAR
# ------------------------
st.sidebar.markdown("### **Company Selection**")
company = st.sidebar.selectbox("**Select Company**", ["PSS", "MDKB"])

st.sidebar.markdown("---")
st.sidebar.markdown("### **Capital & Risk Assumptions**")
rf   = st.sidebar.number_input("Risk-free rate",        value=0.027, step=0.001, format="%.3f")
mrp  = st.sidebar.number_input("Market risk premium",   value=0.040, step=0.001, format="%.3f")
beta = st.sidebar.number_input("Beta",                  value=1.200, step=0.001, format="%.3f")
tax  = st.sidebar.number_input("Tax rate",              value=0.300, step=0.001, format="%.3f")
g    = st.sidebar.number_input("Terminal growth (g)",   value=0.020, step=0.001, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Operational Assumptions**")
dep_pct   = st.sidebar.number_input("Depreciation % of Sales", value=0.010, step=0.001, format="%.3f")
capex_pct = st.sidebar.number_input("CapEx % of Sales",        value=0.010, step=0.001, format="%.3f")
use_nwc   = st.sidebar.checkbox("Include ΔNWC adjustment", value=True)
default_nwc = -0.410 if company=="MDKB" else 0.100
nwc_pct   = st.sidebar.number_input("ΔNWC % of ΔSales", value=float(default_nwc), step=0.001, format="%.3f")
mdkb_extend_growth = st.sidebar.number_input("MDKB 2029 growth", value=0.020, step=0.001, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Debt & Financing**")
debt = st.sidebar.number_input("Debt (€)", value=0.0, step=1_000_000.0)
rd   = st.sidebar.number_input("Cost of Debt (Rd)", value=0.040, step=0.001, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.markdown("### **Acquisition & IRR Settings**")
assumed_price_mdkb = st.sidebar.number_input("Assumed Price for MDKB (€)", value=0.0, step=100_000.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### **FCF Source**")
fcf_source = st.sidebar.radio(
    "Choose FCF model:",
    ["Computed (from EBIT, Dep%, CapEx%, ΔNWC)","Table (provided FCF, adjusted by drivers)"],
    index=0 if company=="PSS" else 1
)

# ------------------------
# DATA PREP
# ------------------------
if company=="PSS":
    df=df_pss.copy(); title="Power Service Solutions GmbH (PSS)"
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
        if company=="PSS" and y==2029: fcf=df.loc[y,"FCF_kEUR"]*1000
        fcfs.append(fcf)
else:
    # “Table” mode but responsive to assumptions
    base_fcfs=(df["FCF_kEUR"].values*1000).astype(float)
    adj=[]
    for i,y in enumerate(years):
        s=sales_eur[i]; prev=sales_eur[i-1] if i>0 else s
        e=ebit_eur[i]
        dep=s*dep_pct; capex=s*capex_pct
        dNWC=(s-prev)*nwc_pct if (use_nwc and i>0) else 0
        driver=(e*(1-tax))+dep-capex-dNWC
        # Adjustment proportional to EBIT
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
st.caption(f"Logged in as **{USER_NAMES.get(st.session_state['user'], st.session_state['user'])}** — {dt.datetime.now():%H:%M}")

if "Computed" in fcf_source:
    st.success("The model is currently using **Computed Free Cash Flow (FCFF)**. "
        "This represents the cash flow available to all investors, derived from operating profits before financing effects.\n\n"
        "**Formula:**  \nFCFF = EBIT × (1 − Tax) + Depreciation − CapEx − ΔNWC  \n"
        "Where:  \n• Depreciation = Dep% × Sales  \n• CapEx = CapEx% × Sales  \n• ΔNWC = (Salesₜ − Salesₜ₋₁) × (ΔNWC% of ΔSales)\n\n"
        "This method ensures consistency with the chosen cost and growth drivers but may not match accounting cash flows exactly.")
else:
    st.info("The model is using **Table Free Cash Flow (Adjusted)**. "
            "Original management FCF values are used but adjusted dynamically according to sidebar assumptions. "
            "This allows responsiveness while retaining the plan structure.")

# ------------------------
# RESULTS
# ------------------------
st.subheader(f"Key Lines — {title}")
disp=df.copy();disp.insert(0,"Year",[str(y) for y in years])
st.dataframe(disp.style.format({c:"{:,.0f}" for c in disp.columns if c.endswith("_kEUR")}),use_container_width=True)

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Re",f"{Re*100:.2f}%");c2.metric("Rd",f"{rd*100:.2f}%")
c3.metric("WACC",f"{WACC*100:.2f}%");c4.metric("EV",f"€{EV:,.0f}");c5.metric("Equity",f"€{EqV:,.0f}")
st.caption("⚙️ Note: Changing the FCF model modifies underlying cashflows, hence the EV & Equity values differ accordingly.")

dfres=pd.DataFrame({"Year":[str(y) for y in years],
                    "Sales (€)":sales_eur,"EBIT (€)":ebit_eur,"Net (€)":net_eur,"FCF (€)":fcfs,"PV(FCF)":pv_fcfs})
st.dataframe(dfres.style.format({c:"€{:,.0f}" for c in dfres.columns if c!="Year"}),use_container_width=True)

fig=plt.figure(figsize=(9,4.5))
plt.plot(years,fcfs,"o-",label="FCF");plt.plot(years,pv_fcfs,"o-",label="PV(FCF)")
plt.axhline(0,linewidth=.8);plt.legend();plt.title(f"Free Cash Flow — {company}")
st.pyplot(fig)

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
plt.bar(["FCF","Net"],[IRR_fcf*100,IRR_net*100]);plt.ylabel("%");plt.title(f"IRR — {company}")
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

# ------------------------
# EXPORT
# ------------------------
st.markdown("### 📦 Export Options")
if st.button("💾 Export locally"):
    out=ts_folder(RESULTS_DIR)
    dfres.to_csv(os.path.join(out,"dcf_results.csv"),index=False)
    df_sens.to_csv(os.path.join(out,"sensitivity.csv"))
    with open(os.path.join(out,"assumptions.json"),"w") as f:
        json.dump({"company":company,"FCF_source":fcf_source,"EV":EV,"Equity":EqV,"IRR_FCF":IRR_fcf,"IRR_Net":IRR_net},f,indent=2)
    fig.savefig(os.path.join(out,"DCF_chart.png"),dpi=150,bbox_inches="tight")
    fig2.savefig(os.path.join(out,"IRR_chart.png"),dpi=150,bbox_inches="tight")
    st.success(f"✅ Exported locally to: {out}")

excel_buffer=io.BytesIO()
with pd.ExcelWriter(excel_buffer,engine="xlsxwriter") as writer:
    dfres.to_excel(writer,sheet_name="DCF_Results",index=False)
    df_sens.to_excel(writer,sheet_name="Sensitivity",index=True)
    workbook=writer.book
    ws1=workbook.add_worksheet("Chart_DCF"); ws2=workbook.add_worksheet("Chart_IRR")
    img1=io.BytesIO(); fig.savefig(img1,format="png",dpi=150); img1.seek(0)
    img2=io.BytesIO(); fig2.savefig(img2,format="png",dpi=150); img2.seek(0)
    ws1.insert_image("B2","DCF.png",{"image_data":img1})
    ws2.insert_image("B2","IRR.png",{"image_data":img2})
    pd.DataFrame({"Metric":["Company","FCF Source","EV","Equity","IRR (FCF)","IRR (Net)","WACC","Re","Rd"],
                  "Value":[company,fcf_source,EV,EqV,IRR_fcf,IRR_net,WACC,Re,rd]}).to_excel(writer,sheet_name="Summary",index=False)
excel_buffer.seek(0)
st.download_button(label=f"⬇️ Download {company} Excel Report (with charts)",
                   data=excel_buffer,
                   file_name=f"{company}_Valuation_Report_{dt.datetime.now():%Y%m%d}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
