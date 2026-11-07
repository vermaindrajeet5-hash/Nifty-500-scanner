#!/usr/bin/env python3
"""Historical NIFTY 500 monthly scanner.

Signal on each completed month:
- Member of NIFTY 500 on that month's last trading date
- Supertrend(20, 1) green: monthly close > Supertrend line
- Wilder ADX(10) this month > previous month's ADX(10)
- Wilder RSI(5) > its own 14-month SMA

Uses Yahoo Finance daily OHLC resampled to monthly candles and a historical
membership CSV. Review the Scan Notes sheet before relying on results.
"""
import os, io, time, logging, smtplib
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
import numpy as np
import pandas as pd
import requests
import yfinance as yf

YEARS_BACK=int(os.getenv("YEARS_BACK","10"))
OUT=Path(os.getenv("OUTPUT_FILE","nifty500_monthly_supertrend_adx_rsi.xlsx"))
MEMBERSHIP_URL=os.getenv("MEMBERSHIP_URL","https://raw.githubusercontent.com/aditya-jha/nse-historical-membership/main/index_history/data/index_membership_history.csv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log=logging.getLogger("scanner")

def membership_history():
    r=requests.get(MEMBERSHIP_URL,timeout=60); r.raise_for_status()
    raw=pd.read_csv(io.BytesIO(r.content))
    raw.columns=[str(c).strip().lower().replace(" ","_") for c in raw.columns]
    def col(names):
        return next((c for c in names if c in raw.columns),None)
    sym=col(["symbol","ticker","security"])
    start=col(["valid_from","from_date","start_date","date_added","from"])
    end=col(["valid_to","to_date","end_date","date_removed","to"])
    idx=col(["index_name","index"])
    if not all([sym,start,end]): raise ValueError(f"Unrecognized membership CSV columns: {list(raw.columns)}")
    m=pd.DataFrame({"Symbol":raw[sym].astype(str).str.strip().str.upper(),
                    "From":pd.to_datetime(raw[start],errors="coerce",dayfirst=True),
                    "To":pd.to_datetime(raw[end],errors="coerce",dayfirst=True)})
    if idx: m=m[raw[idx].astype(str).str.contains(r"NIFTY\s*500",case=False,na=False).values]
    m=m.dropna(subset=["Symbol","From"])
    m["To"]=m["To"].fillna(pd.Timestamp.max)
    return m

def rma(s,n):
    out=pd.Series(np.nan,index=s.index,dtype=float)
    v=s.dropna()
    if len(v)<n: return out
    pos=s.index.get_loc(v.index[n-1]); out.iloc[pos]=v.iloc[:n].mean()
    for i in range(pos+1,len(s)):
        if pd.notna(out.iloc[i-1]) and pd.notna(s.iloc[i]):
            out.iloc[i]=(out.iloc[i-1]*(n-1)+s.iloc[i])/n
    return out

def indicators(d):
    x=d.copy(); h,l,c=x.High,x.Low,x.Close; pc=c.shift()
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    up=h.diff(); dn=-l.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.),index=x.index)
    mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=x.index)
    atr10=rma(tr,10)
    pdi=100*rma(pdm,10)/atr10; mdi=100*rma(mdm,10)/atr10
    dx=100*(pdi-mdi).abs()/(pdi+mdi)
    x["ADX10"]=rma(dx,10); x["PrevADX10"]=x.ADX10.shift()
    delta=c.diff(); ag=rma(delta.clip(lower=0),5); al=rma(-delta.clip(upper=0),5)
    rs=ag/al.replace(0,np.nan); rsi=100-100/(1+rs)
    rsi=rsi.mask((al==0)&(ag>0),100).mask((al==0)&(ag==0),50)
    x["RSI5"]=rsi; x["RSI5_SMA14"]=rsi.rolling(14,min_periods=14).mean()
    atr20=rma(tr,20); mid=(h+l)/2
    bu=mid+atr20; bl=mid-atr20
    fu=pd.Series(np.nan,index=x.index); fl=fu.copy(); st=fu.copy()
    for i in range(len(x)):
        if pd.isna(atr20.iloc[i]): continue
        if i==0 or pd.isna(fu.iloc[i-1]):
            fu.iloc[i]=bu.iloc[i]; fl.iloc[i]=bl.iloc[i]
            st.iloc[i]=bu.iloc[i] if c.iloc[i]<=bu.iloc[i] else bl.iloc[i]
        else:
            pu,pl=fu.iloc[i-1],fl.iloc[i-1]
            fu.iloc[i]=bu.iloc[i] if bu.iloc[i]<pu or c.iloc[i-1]>pu else pu
            fl.iloc[i]=bl.iloc[i] if bl.iloc[i]>pl or c.iloc[i-1]<pl else pl
            st.iloc[i]=(fu.iloc[i] if c.iloc[i]<=fu.iloc[i] else fl.iloc[i]) if st.iloc[i-1]==pu else (fl.iloc[i] if c.iloc[i]>=fl.iloc[i] else fu.iloc[i])
    x["ST20_1"]=st; x["ST_Green"]=c>st
    return x

def download(symbol,start,end):
    d=yf.download(symbol+".NS",start=start,end=end,interval="1d",auto_adjust=False,progress=False,threads=False)
    if d.empty: return d
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d[["High","Low","Close"]].apply(pd.to_numeric,errors="coerce").dropna()
    d.index=pd.to_datetime(d.index).tz_localize(None)
    return d


def is_last_nse_trading_day(today=None):
    """Return True only on the final NSE trading session of the month."""
    import exchange_calendars as xc
    day = pd.Timestamp(today or pd.Timestamp.now(tz="Asia/Kolkata").date()).normalize()
    cal = xc.get_calendar("XNSE")
    if not cal.is_session(day):
        return False
    return cal.next_session(day).month != day.month


def scan(history=True):
    m=membership_history()
    now=pd.Timestamp.now(tz="Asia/Kolkata")
    if history:
        start=(now-pd.DateOffset(years=YEARS_BACK)).date()
    else:
        start=now.to_period("M").start_time.date()
    ds=(pd.Timestamp(start)-pd.DateOffset(years=4)).strftime("%Y-%m-%d")
    de=(now+pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    hits=[]; skipped=[]
    for no,symbol in enumerate(sorted(m.Symbol.unique()),1):
        try:
            d=download(symbol,ds,de)
            if d.empty: skipped.append((symbol,"No Yahoo Finance data")); continue
            month=d.resample("ME").agg({"High":"max","Low":"min","Close":"last"}).dropna()
            month=indicators(month)
            last_dates=d.Close.resample("ME").apply(lambda z:z.dropna().index[-1] if not z.dropna().empty else pd.NaT)
            month["LastTradingDate"]=last_dates
            month=month[month.index.date>=start]
            for month_end,row in month.iterrows():
                day=pd.Timestamp(row.LastTradingDate)
                member=m[(m.Symbol==symbol)&(m.From<=day)&(m.To>day)]
                if member.empty: continue
                vals=[row.ADX10,row.PrevADX10,row.RSI5,row.RSI5_SMA14,row.ST20_1]
                if not np.isfinite(np.asarray(vals,dtype=float)).all(): continue
                if row.ST_Green and row.ADX10>row.PrevADX10 and row.RSI5>row.RSI5_SMA14:
                    hits.append({"Month":month_end.strftime("%Y-%m"),"Signal Date":day.date().isoformat(),"Symbol":symbol,
                      "Monthly Close":round(row.Close,2),"Supertrend (20,1)":round(row.ST20_1,2),
                      "ADX (10)":round(row.ADX10,2),"Previous Month ADX (10)":round(row.PrevADX10,2),
                      "RSI (5)":round(row.RSI5,2),"RSI (5) SMA (14)":round(row.RSI5_SMA14,2)})
            if no%25==0: log.info("Processed %d/%d",no,m.Symbol.nunique())
            time.sleep(.04)
        except Exception as e:
            skipped.append((symbol,str(e)[:200])); log.exception("Failed %s",symbol)
    notes=pd.DataFrame([
      ["Universe","Historical NIFTY 500 membership from configured membership CSV; membership tested on each month's final available trading date."],
      ["Monthly candle","Daily Yahoo Finance OHLC resampled to calendar months; signal is evaluated on completed monthly rows."],
      ["Supertrend","Green is defined as monthly close above Supertrend(20,1)."],
      ["ADX","Wilder ADX(10); current completed month must be strictly greater than prior completed month."],
      ["RSI","Wilder RSI(5) must be strictly greater than its own 14-month simple moving average."],
      ["Limitations","Historical membership CSV schema/coverage and Yahoo data are external dependencies. Indicator implementations/data may differ from TradingView or other charting platforms; validate samples before trading."]
    ],columns=["Item","Details"])
    with pd.ExcelWriter(OUT,engine="openpyxl") as w:
        pd.DataFrame(hits).sort_values(["Signal Date","Symbol"],ascending=[False,True]).to_excel(w,sheet_name="Qualified Stocks",index=False) if hits else pd.DataFrame().to_excel(w,sheet_name="Qualified Stocks",index=False)
        notes.to_excel(w,sheet_name="Scan Notes",index=False)
        pd.DataFrame(skipped,columns=["Symbol","Reason"]).to_excel(w,sheet_name="Skipped Symbols",index=False)
    log.info("Saved %s (%d qualifying rows)",OUT,len(hits))
    return OUT

def email_report(path):
    user=os.getenv("GMAIL_USERNAME"); password=os.getenv("GMAIL_APP_PASSWORD"); to=os.getenv("GMAIL_TO")
    if not all([user,password,to]): log.info("Email secrets missing; workbook saved without email."); return
    msg=EmailMessage(); msg["Subject"]="NIFTY 500 monthly Supertrend ADX RSI historical scan"
    msg["From"]=user; msg["To"]=to
    msg.set_content("Attached is the NIFTY 500 monthly historical scan workbook.")
    msg.add_attachment(path.read_bytes(),maintype="application",subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",filename=path.name)
    with smtplib.SMTP_SSL("smtp.gmail.com",465,timeout=30) as s:
        s.login(user,password); s.send_message(msg)

if __name__ == "__main__":
    scheduled = os.getenv("GITHUB_EVENT_NAME") == "schedule"
    if scheduled:
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        if not is_last_nse_trading_day(today):
            log.info("Not the final NSE trading session of the month; exiting without a scan.")
            raise SystemExit(0)
        log.info("Final NSE trading session confirmed; scanning the current month.")
        report = scan(history=False)
    else:
        log.info("Manual run: scanning the full historical period.")
        report = scan(history=True)
    email_report(report)
