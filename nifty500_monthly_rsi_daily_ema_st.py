import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

YEARS=10; RP=5; RMA=14; EP=20; SP=10; SM=1.0; BS=10
URL="https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/main/index_history/data/index_membership_history.csv"
OUT="nifty500_monthly_rsi_daily_ema_st.xlsx"

def idx(x):
    x=pd.DatetimeIndex(pd.to_datetime(x,errors="coerce"))
    if x.tz is not None:x=x.tz_localize(None)
    return x.normalize().astype("datetime64[ns]")

def rsi(s,p):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/p,adjust=False,min_periods=p).mean()
    al=l.ewm(alpha=1/p,adjust=False,min_periods=p).mean()
    z=ag/al.replace(0,np.nan); r=100-100/(1+z)
    r=r.where(al!=0,100.0); return r.where(~((ag==0)&(al==0)),50.0)

def st(d,p=10,m=1.0):
    h,l,c=d.High,d.Low,d.Close; pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/p,adjust=False,min_periods=p).mean(); mid=(h+l)/2
    bu=mid+m*a; bl=mid-m*a
    fu=pd.Series(index=d.index,dtype=float); fl=fu.copy(); out=fu.copy()
    dr=pd.Series(index=d.index,dtype=object)
    for i in range(len(d)):
        if pd.isna(a.iloc[i]): continue
        if i==0 or pd.isna(out.iloc[i-1]):
            fu.iloc[i]=bu.iloc[i]; fl.iloc[i]=bl.iloc[i]; out.iloc[i]=fl.iloc[i]; dr.iloc[i]="GREEN"; continue
        pu,pl,pcv=fu.iloc[i-1],fl.iloc[i-1],c.iloc[i-1]
        fu.iloc[i]=bu.iloc[i] if bu.iloc[i]<pu or pcv>pu else pu
        fl.iloc[i]=bl.iloc[i] if bl.iloc[i]>pl or pcv<pl else pl
        if out.iloc[i-1]==pu:
            if c.iloc[i]<=fu.iloc[i]: out.iloc[i]=fu.iloc[i]; dr.iloc[i]="RED"
            else: out.iloc[i]=fl.iloc[i]; dr.iloc[i]="GREEN"
        else:
            if c.iloc[i]>=fl.iloc[i]: out.iloc[i]=fl.iloc[i]; dr.iloc[i]="GREEN"
            else: out.iloc[i]=fu.iloc[i]; dr.iloc[i]="RED"
    return dr

def membership():
    m=pd.read_csv(URL); c={str(x).lower().strip():x for x in m.columns}
    m=m[[c["symbol"],c["valid_from"],c["valid_to"]]].copy(); m.columns=["Symbol","From","To"]
    m.Symbol=m.Symbol.astype(str).str.strip().str.upper()
    m["From"]=pd.to_datetime(m["From"],errors="coerce").dt.normalize().astype("datetime64[ns]")
    m["To"]=pd.to_datetime(m["To"],errors="coerce").dt.normalize().astype("datetime64[ns]")
    return m.dropna(subset=["Symbol","From"])

def member(s,dt,m):
    return not m[(m.Symbol==s)&(m["From"]<=dt)&(m["To"].isna()|(m["To"]>dt))].empty

def download(ss):
    for a in range(3):
        try:
            d=yf.download([f"{s}.NS" for s in ss],period="max",interval="1d",auto_adjust=True,group_by="ticker",threads=True,progress=False)
            if d is not None and not d.empty:return d
        except Exception as e: print(e)
        time.sleep(4*(a+1))
    return None

def extract(b,s):
    try:
        t=f"{s}.NS"
        if isinstance(b.columns,pd.MultiIndex):
            z=b[t].copy() if t in set(b.columns.get_level_values(0)) else b.xs(t,axis=1,level=1).copy()
        else:z=b.copy()
        z=z[["Open","High","Low","Close"]].dropna(); z.index=idx(z.index)
        return z[~z.index.duplicated(keep="last")].sort_index()
    except:return None

def scan(s,d,m,start,end):
    d=d.copy(); d.index=idx(d.index)
    d["EMA20"]=d.Close.ewm(span=EP,adjust=False,min_periods=EP).mean()
    d["ST"]=st(d,SP,SM)
    d["Touch"]=(d.Low<=d.EMA20)&(d.High>=d.EMA20)
    mo=d.resample("ME").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
    if mo.empty:return []
    mo.index=idx(mo.index); mo["RSI"]=rsi(mo.Close,RP); mo["SMA"]=mo.RSI.rolling(RMA,min_periods=RMA).mean()
    mo["Cross"]=(mo.RSI>mo.SMA)&(mo.RSI.shift(1)<=mo.SMA.shift(1)); mo["Active"]=mo.RSI>mo.SMA
    pm=d.index.to_period("M").to_timestamp()-pd.Timedelta(days=1); pm=pm.astype("datetime64[ns]")
    d["PM"]=pm; d["MRSI"]=d.PM.map(mo.RSI.to_dict()); d["MSMA"]=d.PM.map(mo.SMA.to_dict()); d["Cross"]=d.PM.map(mo.Cross.to_dict()); d["Active"]=d.PM.map(mo.Active.to_dict()).fillna(False)
    q=d[(d.index>=start)&(d.index<=end)&d.Active&d.Touch&(d.ST=="GREEN")]
    out=[]
    for dt,r in q.iterrows():
        dt=pd.Timestamp(dt).normalize()
        if not member(s,dt,m):continue
        out.append({"Date":dt.date(),"Stock":s,"Monthly_RSI_5":round(float(r.MRSI),2),"Monthly_RSI_5_SMA_14":round(float(r.MSMA),2),"Monthly_Cross_Above":bool(r.Cross),"Previous_Month_End":pd.Timestamp(r.PM).date(),"Daily_Close":round(float(r.Close),2),"Daily_Low":round(float(r.Low),2),"Daily_High":round(float(r.High),2),"Daily_EMA_20":round(float(r.EMA20),2),"Daily_Supertrend":"GREEN"})
    return out

def main():
    end=pd.Timestamp.today().normalize(); start=end-pd.DateOffset(years=YEARS); m=membership()
    ss=sorted(m.Symbol.unique().tolist()); results=[]; errors=[]
    for n in range(0,len(ss),BS):
        batch=ss[n:n+BS]; print(f"Batch {n//BS+1}/{(len(ss)+BS-1)//BS}")
        b=download(batch)
        if b is None:
            errors += [{"Stock":s,"Error":"Yahoo batch download failed"} for s in batch]; continue
        for s in batch:
            try:
                d=extract(b,s)
                if d is None or d.empty: errors.append({"Stock":s,"Error":"No usable Yahoo data"}); continue
                results += scan(s,d,m,start,end)
            except Exception as e: errors.append({"Stock":s,"Error":str(e)})
        time.sleep(2)
    sig=pd.DataFrame(results)
    cols=["Date","Stock","Monthly_RSI_5","Monthly_RSI_5_SMA_14","Monthly_Cross_Above","Previous_Month_End","Daily_Close","Daily_Low","Daily_High","Daily_EMA_20","Daily_Supertrend"]
    if sig.empty:sig=pd.DataFrame(columns=cols)
    else:sig=sig.sort_values(["Date","Stock"]).reset_index(drop=True)
    err=pd.DataFrame(errors) if errors else pd.DataFrame(columns=["Stock","Error"])
    rules=pd.DataFrame({"Rule":["Period","Universe","Monthly trigger","Monthly active","Daily EMA","Daily touch","Daily Supertrend","Membership","Signal"],"Definition":[f"{start.date()} to {end.date()}","Historical NIFTY 500 on signal date","Monthly RSI(5) crosses above SMA(14) of RSI(5)","Monthly RSI(5) remains above SMA(14)","EMA(20) of daily Close","Daily Low <= EMA20 <= Daily High","Supertrend(10,1) GREEN","Stock must be NIFTY 500 on exact signal date","Monthly active + daily EMA20 touch + ST GREEN"]})
    with pd.ExcelWriter(OUT,engine="openpyxl") as w:
        sig.to_excel(w,sheet_name="Signals",index=False); rules.to_excel(w,sheet_name="Rules",index=False); err.to_excel(w,sheet_name="Data Errors",index=False)
        for ws in w.sheets.values():
            ws.freeze_panes="A2"
            if ws.max_row>1:ws.auto_filter.ref=ws.dimensions
            for cells in ws.columns:
                ws.column_dimensions[cells[0].column_letter].width=min(max(max(len(str(c.value)) if c.value is not None else 0 for c in cells)+2,12),55)
    print(f"TOTAL SIGNAL ROWS: {len(sig)}")
    print(f"DATA ERRORS: {len(err)}")
    print(Path(OUT).resolve())

if __name__=="__main__":main()
