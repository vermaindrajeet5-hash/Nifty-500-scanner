import io,time,requests,numpy as np,pandas as pd,yfinance as yf

START="2020-08-25"
URL="https://www.niftyindices.com/IndexConstituent/ind_nifty200Momentum30_list.csv"
OUT="nifty200_momentum30_historical_signals.xlsx"

def rsi(s,n=5):
 d=s.diff();g=d.clip(lower=0);l=-d.clip(upper=0)
 ag=g.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
 al=l.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
 z=100-100/(1+ag/al.replace(0,np.nan))
 return z.where(~((al==0)&(ag>0)),100).where(~((ag==0)&(al>0)),0)

def supertrend(x,n=10,m=1):
 h,l,c=x.High.astype(float),x.Low.astype(float),x.Close.astype(float)
 pc=c.shift();tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
 atr=tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean();mid=(h+l)/2
 ub=mid+m*atr;lb=mid-m*atr;u=ub.copy();lo=lb.copy();dr=pd.Series(index=x.index,dtype="int64")
 for i in range(len(x)):
  if i==0: dr.iloc[i]=1;continue
  if pd.isna(atr.iloc[i]): dr.iloc[i]=dr.iloc[i-1];continue
  u.iloc[i]=ub.iloc[i] if ub.iloc[i]<u.iloc[i-1] or c.iloc[i-1]>u.iloc[i-1] else u.iloc[i-1]
  lo.iloc[i]=lb.iloc[i] if lb.iloc[i]>lo.iloc[i-1] or c.iloc[i-1]<lo.iloc[i-1] else lo.iloc[i-1]
  dr.iloc[i]=1 if (dr.iloc[i-1]==-1 and c.iloc[i]>u.iloc[i]) or (dr.iloc[i-1]==1 and c.iloc[i]>=lo.iloc[i]) else -1
 return pd.Series(np.where(dr==1,lo,u),index=x.index),dr

def constituents():
 r=requests.get(URL,headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.niftyindices.com/"},timeout=30)
 r.raise_for_status()
 m=pd.read_csv(io.StringIO(r.content.decode("utf-8-sig",errors="replace")))
 m.columns=[str(c).strip() for c in m.columns]
 col=next((c for c in m.columns if c.lower() in ("symbol","ticker","security")),None)
 if not col: raise ValueError(f"Symbol column not found: {list(m.columns)}")
 m["Symbol"]=m[col].astype(str).str.strip().str.upper()
 m=m[m.Symbol.notna() & ~m.Symbol.isin(["NAN",""])].drop_duplicates("Symbol").reset_index(drop=True)
 if len(m)!=30: raise ValueError(f"NSE returned {len(m)} constituents, expected 30")
 return m

def scan(sym):
 t=sym if sym.endswith(".NS") else sym+".NS"
 d=yf.download(t,start=(pd.Timestamp(START)-pd.Timedelta(days=500)).strftime("%Y-%m-%d"),auto_adjust=False,progress=False,threads=False)
 if d.empty:return pd.DataFrame()
 if isinstance(d.columns,pd.MultiIndex):d.columns=[c[0] for c in d.columns]
 d.index=pd.to_datetime(d.index,utc=True).tz_convert(None).normalize()
 d=d.dropna(subset=["Close"]).sort_index()
 d["DRSI"]=rsi(d.Close);d["DRSMA"]=d.DRSI.rolling(14,min_periods=14).mean()
 d["EMA20"]=d.Close.ewm(span=20,adjust=False,min_periods=20).mean()
 d["Touch"]=(d.Low<=d.EMA20)&(d.High>=d.EMA20)
 d["ST"],d["Dir"]=supertrend(d);d["Green"]=d.Dir.eq(1)
 mo=d.Close.resample("ME").last().to_frame("Close")
 mo["RSI"]=rsi(mo.Close);mo["SMA"]=mo.RSI.rolling(14,min_periods=14).mean()
 mo["Cross"]=(mo.RSI>mo.SMA)&(mo.RSI.shift(1)<=mo.SMA.shift(1));mo["Active"]=mo.RSI>mo.SMA
 pos=mo.index.searchsorted(d.index,side="left")-1
 pm=pd.Series(pd.NaT,index=d.index,dtype="datetime64[ns]");ok=pos>=0;pm.loc[ok]=mo.index[pos[ok]]
 d["PM"]=pm;d["MActive"]=d.PM.map(mo.Active.to_dict());d["MCross"]=d.PM.map(mo.Cross.to_dict())
 d["MRSI"]=d.PM.map(mo.RSI.to_dict());d["MSMA"]=d.PM.map(mo.SMA.to_dict())
 q=(d.index>=pd.Timestamp(START))&d.MActive.fillna(False)&d.Touch.fillna(False)&d.Green.fillna(False)&(d.DRSI>d.DRSMA)
 z=d.loc[q]
 if z.empty:return pd.DataFrame()
 return pd.DataFrame({"Date":z.index.date,"Stock":sym,"Monthly_RSI5":z.MRSI.values,"Monthly_RSI5_SMA14":z.MSMA.values,"Monthly_Cross_Above":z.MCross.fillna(False).values,"Previous_Month_End":z.PM.dt.date.values,"Daily_RSI5":z.DRSI.values,"Daily_RSI5_SMA14":z.DRSMA.values,"Close":z.Close.values,"Low":z.Low.values,"High":z.High.values,"EMA20":z.EMA20.values,"Supertrend":z.ST.values,"Supertrend_Green":True})

def main():
 m=constituents();allr=[];errs=[]
 for i,s in enumerate(m.Symbol,1):
  print(f"[{i}/30] {s}",flush=True)
  try:
   z=scan(s)
   if not z.empty:allr.append(z)
  except Exception as e:errs.append({"Stock":s,"Error":repr(e)})
 sig=pd.concat(allr,ignore_index=True).sort_values(["Date","Stock"]) if allr else pd.DataFrame()
 rules=pd.DataFrame({"Rule":["Universe","Period","Monthly","Daily 1","Daily 2","Daily 3","Look-ahead"],"Definition":["NSE Nifty200 Momentum 30 constituents downloaded at run time","25-Aug-2020 onward","Monthly RSI(5) > SMA(14), with cross recorded; active while above","Daily candle touches EMA20","Supertrend(10,1) GREEN","Daily RSI(5) > SMA(14)","Previous completed monthly candle used"]})
 with pd.ExcelWriter(OUT,engine="openpyxl") as w:
  sig.to_excel(w,"Signals",index=False);m.to_excel(w,"Nifty200 Momentum 30",index=False);rules.to_excel(w,"Rules",index=False);pd.DataFrame(errs).to_excel(w,"Data Errors",index=False)
 print(f"Saved {OUT}; signals={len(sig)}",flush=True)
if __name__=="__main__":main()
