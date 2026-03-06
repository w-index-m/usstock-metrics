"""
app.py  ―  US/JP Stock Intelligence Hub
米国株エージェント × NASDAQ100ダッシュボード × jstock-metrics 全機能 統合版
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
import re
import time
import concurrent.futures
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import Counter
from io import BytesIO

# ── オプション: Gemini ──────────────────────────────────────────
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
# ページ設定
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    layout="wide",
    page_title="📊 Stock Intelligence Hub",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── グローバル ── */
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
/* コンパクトメトリック */
div[data-testid="metric-container"] {
    background: #1e2130;
    border-radius: 8px;
    padding: 8px 12px;
    border: 1px solid #ffffff18;
}
div[data-testid="metric-container"] label { font-size: 0.72rem !important; color: #aaa !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
/* エージェントログ */
.alog {
    background: #0d1117;
    border-left: 3px solid #00d4ff;
    border-radius: 6px;
    padding: 6px 12px;
    margin: 4px 0;
    font-family: monospace;
    font-size: 0.78rem;
    color: #ccc;
}
.alog-ok   { border-color: #00ff88; }
.alog-warn { border-color: #ffaa00; }
.alog-err  { border-color: #ff4444; }
/* ニュースカード */
.news-card {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 5px 0;
    border-left: 3px solid #00d4ff;
    font-size: 0.85rem;
}
/* セクションヘッダー */
.sec-head {
    background: linear-gradient(90deg,#1a1a2e,#0d1117);
    border-left: 4px solid #00d4ff;
    padding: 6px 14px;
    border-radius: 4px;
    margin-bottom: 10px;
    font-weight: bold;
    color: #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# AI 初期化
# ════════════════════════════════════════════════════════════════
_gemini_model = None
if _GEMINI_AVAILABLE:
    try:
        _key = st.secrets.get("GEMINI_API_KEY", "")
        if _key:
            genai.configure(api_key=_key)
            _gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    except Exception:
        pass

def ai_call(prompt: str, max_tokens: int = 600) -> tuple[str, str]:
    """Gemini呼び出し（失敗時はフォールバックメッセージ）"""
    if _gemini_model:
        try:
            resp = _gemini_model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                text = resp.candidates[0].content.parts[0].text
            if text:
                return text, "Gemini"
        except Exception as e:
            return f"Gemini エラー: {e}", "Error"
    return "（AIキー未設定 — st.secretsに GEMINI_API_KEY を追加してください）", "N/A"

# ════════════════════════════════════════════════════════════════
# ユーティリティ
# ════════════════════════════════════════════════════════════════
def fmt_large(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def fmt_num(v, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v:.{dec}f}"

def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v*100:.1f}%"

def color_val(v):
    if v is None: return "#aaa"
    return "#00ff88" if float(v) >= 0 else "#ff4444"

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ════════════════════════════════════════════════════════════════
# NASDAQ100 銘柄マスタ（代表100銘柄）
# ════════════════════════════════════════════════════════════════
NDX100 = {
    "AAPL":  ("Apple",            "Technology"),
    "MSFT":  ("Microsoft",        "Technology"),
    "NVDA":  ("NVIDIA",           "Technology"),
    "AMZN":  ("Amazon",           "Consumer Cyclical"),
    "META":  ("Meta",             "Technology"),
    "GOOGL": ("Alphabet A",       "Communication"),
    "GOOG":  ("Alphabet C",       "Communication"),
    "TSLA":  ("Tesla",            "Consumer Cyclical"),
    "AVGO":  ("Broadcom",         "Technology"),
    "COST":  ("Costco",           "Consumer Defensive"),
    "NFLX":  ("Netflix",          "Communication"),
    "ASML":  ("ASML",             "Technology"),
    "AMD":   ("AMD",              "Technology"),
    "QCOM":  ("Qualcomm",         "Technology"),
    "TMUS":  ("T-Mobile",         "Communication"),
    "INTC":  ("Intel",            "Technology"),
    "TXN":   ("Texas Instruments","Technology"),
    "AMAT":  ("Applied Materials","Technology"),
    "INTU":  ("Intuit",           "Technology"),
    "BKNG":  ("Booking Holdings", "Consumer Cyclical"),
    "ISRG":  ("Intuitive Surgical","Healthcare"),
    "REGN":  ("Regeneron",        "Healthcare"),
    "VRTX":  ("Vertex Pharma",    "Healthcare"),
    "MU":    ("Micron",           "Technology"),
    "LRCX":  ("Lam Research",     "Technology"),
    "KLAC":  ("KLA",              "Technology"),
    "CDNS":  ("Cadence Design",   "Technology"),
    "SNPS":  ("Synopsys",         "Technology"),
    "PANW":  ("Palo Alto",        "Technology"),
    "CRWD":  ("CrowdStrike",      "Technology"),
    "ADSK":  ("Autodesk",         "Technology"),
    "MRVL":  ("Marvell Tech",     "Technology"),
    "FTNT":  ("Fortinet",         "Technology"),
    "CSX":   ("CSX",              "Industrials"),
    "PCAR":  ("PACCAR",           "Industrials"),
    "ODFL":  ("Old Dominion",     "Industrials"),
    "FAST":  ("Fastenal",         "Industrials"),
    "CTAS":  ("Cintas",           "Industrials"),
    "DXCM":  ("DexCom",           "Healthcare"),
    "IDXX":  ("IDEXX Labs",       "Healthcare"),
    "BIIB":  ("Biogen",           "Healthcare"),
    "GILD":  ("Gilead",           "Healthcare"),
    "ILMN":  ("Illumina",         "Healthcare"),
    "MRNA":  ("Moderna",          "Healthcare"),
    "PYPL":  ("PayPal",           "Financial"),
    "ADBE":  ("Adobe",            "Technology"),
    "CRM":   ("Salesforce",       "Technology"),
    "NOW":   ("ServiceNow",       "Technology"),
    "WDAY":  ("Workday",          "Technology"),
    "TEAM":  ("Atlassian",        "Technology"),
    "ZS":    ("Zscaler",          "Technology"),
    "DDOG":  ("Datadog",          "Technology"),
    "MDB":   ("MongoDB",          "Technology"),
    "OKTA":  ("Okta",             "Technology"),
    "ZM":    ("Zoom",             "Technology"),
    "DOCU":  ("DocuSign",         "Technology"),
    "ROKU":  ("Roku",             "Communication"),
    "TTWO":  ("Take-Two",         "Communication"),
    "EA":    ("Electronic Arts",  "Communication"),
    "WBD":   ("Warner Bros",      "Communication"),
    "CMCSA": ("Comcast",          "Communication"),
    "AMGN":  ("Amgen",            "Healthcare"),
    "AZN":   ("AstraZeneca",      "Healthcare"),
    "MELI":  ("MercadoLibre",     "Consumer Cyclical"),
    "PDD":   ("PDD Holdings",     "Consumer Cyclical"),
    "JD":    ("JD.com",           "Consumer Cyclical"),
    "BIDU":  ("Baidu",            "Communication"),
    "NTES":  ("NetEase",          "Communication"),
    "SBUX":  ("Starbucks",        "Consumer Cyclical"),
    "MNST":  ("Monster Bev",      "Consumer Defensive"),
    "KDP":   ("Keurig Dr Pepper", "Consumer Defensive"),
    "PEP":   ("PepsiCo",          "Consumer Defensive"),
    "MAR":   ("Marriott",         "Consumer Cyclical"),
    "ABNB":  ("Airbnb",           "Consumer Cyclical"),
    "EXPE":  ("Expedia",          "Consumer Cyclical"),
    "ORLY":  ("O'Reilly Auto",    "Consumer Cyclical"),
    "CSGP":  ("CoStar",           "Real Estate"),
    "EXC":   ("Exelon",           "Utilities"),
    "XEL":   ("Xcel Energy",      "Utilities"),
    "AEP":   ("AEP",              "Utilities"),
    "FANG":  ("Diamondback",      "Energy"),
    "ON":    ("ON Semiconductor", "Technology"),
    "MPWR":  ("Monolithic Power", "Technology"),
    "ENPH":  ("Enphase",          "Technology"),
    "SEDG":  ("SolarEdge",        "Technology"),
    "CEG":   ("Constellation",    "Utilities"),
    "GEHC":  ("GE HealthCare",    "Healthcare"),
    "MDLZ":  ("Mondelez",         "Consumer Defensive"),
    "HON":   ("Honeywell",        "Industrials"),
    "CPRT":  ("Copart",           "Industrials"),
    "VRSK":  ("Verisk",           "Industrials"),
    "ANSS":  ("Ansys",            "Technology"),
    "NXPI":  ("NXP Semi",         "Technology"),
    "MCHP":  ("Microchip Tech",   "Technology"),
    "SMCI":  ("Super Micro",      "Technology"),
    "ARM":   ("ARM Holdings",     "Technology"),
    "APP":   ("AppLovin",         "Technology"),
    "PLTR":  ("Palantir",         "Technology"),
    "HOOD":  ("Robinhood",        "Financial"),
}

# ════════════════════════════════════════════════════════════════
# 日本株マスタ（jstock-metrics v2 から）
# ════════════════════════════════════════════════════════════════
JP_TICKERS = {
    '7203.T':('トヨタ','自動車'),'7267.T':('ホンダ','自動車'),'7201.T':('日産自','自動車'),
    '7261.T':('マツダ','自動車'),'7269.T':('スズキ','自動車'),'7270.T':('ＳＵＢＡＲＵ','自動車'),
    '6501.T':('日立','電気機器'),'6502.T':('東芝','電気機器'),'6503.T':('三菱電','電気機器'),
    '6504.T':('富士電機','電気機器'),'6506.T':('安川電','電気機器'),'6645.T':('オムロン','電気機器'),
    '6702.T':('富士通','電気機器'),'6723.T':('ルネサス','電気機器'),'6752.T':('パナソニック','電気機器'),
    '6758.T':('ソニーＧ','電気機器'),'6762.T':('ＴＤＫ','電気機器'),'6857.T':('アドバンテスト','電気機器'),
    '6861.T':('キーエンス','電気機器'),'6920.T':('レーザーテック','電気機器'),'6954.T':('ファナック','電気機器'),
    '6981.T':('村田製','電気機器'),'7735.T':('スクリン','電気機器'),'7751.T':('キヤノン','電気機器'),
    '8035.T':('東エレク','電気機器'),
    '9984.T':('ＳＢＧ','通信'),'9432.T':('ＮＴＴ','通信'),'9433.T':('ＫＤＤＩ','通信'),
    '9434.T':('ＳＢ','通信'),'9613.T':('ＮＴＴデータ','通信'),
    '8306.T':('三菱ＵＦＪ','銀行'),'8316.T':('三井住友ＦＧ','銀行'),'8411.T':('みずほＦＧ','銀行'),
    '8308.T':('りそなＨＤ','銀行'),'8309.T':('三井住友トラ','銀行'),
    '8601.T':('大和','証券'),'8604.T':('野村','証券'),
    '8766.T':('東京海上','保険'),'8630.T':('ＳＯＭＰＯ','保険'),'8725.T':('ＭＳ＆ＡＤ','保険'),
    '8750.T':('第一生命ＨＤ','保険'),
    '8801.T':('三井不','不動産'),'8802.T':('菱地所','不動産'),'8804.T':('東建物','不動産'),
    '8830.T':('住友不','不動産'),
    '8001.T':('伊藤忠','商社'),'8002.T':('丸紅','商社'),'8031.T':('三井物','商社'),
    '8053.T':('住友商','商社'),'8058.T':('三菱商','商社'),
    '9020.T':('ＪＲ東日本','鉄道・バス'),'9021.T':('ＪＲ西日本','鉄道・バス'),
    '9022.T':('ＪＲ東海','鉄道・バス'),
    '9101.T':('郵船','海運'),'9104.T':('商船三井','海運'),'9107.T':('川崎汽','海運'),
    '9201.T':('ＪＡＬ','空運'),'9202.T':('ＡＮＡＨＤ','空運'),
    '2802.T':('味の素','食品'),'2503.T':('キリンＨＤ','食品'),'2502.T':('アサヒ','食品'),
    '2914.T':('ＪＴ','食品'),
    '4502.T':('武田薬','医薬品'),'4503.T':('アステラス薬','医薬品'),'4519.T':('中外製薬','医薬品'),
    '4568.T':('第一三共','医薬品'),
    '3382.T':('セブン＆アイ','小売業'),'9983.T':('ファストリ','小売業'),'9843.T':('ニトリＨＤ','小売業'),
    '3092.T':('ＺＯＺＯ','小売業'),
    '1605.T':('ＩＮＰＥＸ','鉱業'),
    '9501.T':('東電ＨＤ','電力'),'9502.T':('中部電','電力'),'9503.T':('関西電','電力'),
    '7011.T':('三菱重','機械'),'7013.T':('ＩＨＩ','機械'),'6326.T':('クボタ','機械'),
    '4452.T':('花王','化学'),'4063.T':('信越化','化学'),'4183.T':('三井化学','化学'),
    '5401.T':('日本製鉄','鉄鋼'),'5411.T':('ＪＦＥ','鉄鋼'),
    '7974.T':('任天堂','サービス'),'9735.T':('セコム','サービス'),'9766.T':('コナミＧ','サービス'),
    '2413.T':('エムスリー','サービス'),
    '4689.T':('ＬＹＨＤｊp','情報・通信'),'4755.T':('楽天Ｇ','情報・通信'),
}

# ════════════════════════════════════════════════════════════════
# データ取得関数
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def get_ohlcv(ticker: str, days: int = 400) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


@st.cache_data(ttl=600)
def get_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900)
def fetch_rss(url: str, source: str, max_items: int = 10, ticker_filter: str = "") -> list[dict]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item"):
            title   = item.findtext("title", "").strip()
            link    = item.findtext("link", "").strip()
            pubdate = item.findtext("pubDate", "")[:16].strip()
            desc    = re.sub(r"<[^>]+>", "", item.findtext("description", ""))[:120].strip()
            if not title:
                continue
            if ticker_filter and ticker_filter.upper() not in title.upper():
                continue
            items.append({"source": source, "title": title,
                          "link": link, "date": pubdate, "summary": desc})
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def fetch_stock_news(ticker: str, company: str = "", max_each: int = 8) -> list[dict]:
    """個別銘柄ニュースをYahoo Finance + Reuters + Google RSSから並列取得"""
    tasks = [
        (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
         "Yahoo Finance", max_each, ""),
        ("https://feeds.reuters.com/reuters/businessNews",
         "Reuters", max_each, ticker),
        (f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
         "Google News", max_each, ""),
    ]
    all_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(fetch_rss, url, src, n, f) for url, src, n, f in tasks]
        for fut in concurrent.futures.as_completed(futures):
            try:
                all_items.extend(fut.result())
            except Exception:
                pass
    # 重複除去
    seen, unique = set(), []
    for it in all_items:
        if it["title"] not in seen:
            seen.add(it["title"])
            unique.append(it)
    return unique


def fetch_market_news(max_each: int = 10) -> list[dict]:
    tasks = [
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters Business", max_each, ""),
        ("https://feeds.reuters.com/reuters/technologyNews", "Reuters Tech", max_each, ""),
        ("https://news.google.com/rss/search?q=nasdaq+market&hl=en-US&gl=US&ceid=US:en",
         "Google News", max_each, ""),
    ]
    all_items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(fetch_rss, url, src, n, f) for url, src, n, f in tasks]
        for fut in concurrent.futures.as_completed(futures):
            try:
                all_items.extend(fut.result())
            except Exception:
                pass
    seen, unique = set(), []
    for it in all_items:
        if it["title"] not in seen:
            seen.add(it["title"])
            unique.append(it)
    return unique[:30]


# ════════════════════════════════════════════════════════════════
# NASDAQ100 ダッシュボード用 計算
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800)
def ndx_snapshot(tickers: dict) -> pd.DataFrame:
    """全銘柄の直近スナップショット（リターン・出来高・RSI）"""
    rows = []
    def _fetch(sym, name, sector):
        try:
            df = get_ohlcv(sym, 60)
            if df.empty or len(df) < 5:
                return None
            c = df["Close"].dropna()
            cur = float(c.iloc[-1])
            d1  = float((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100) if len(c)>=2 else 0
            d5  = float((c.iloc[-1]-c.iloc[-5])/c.iloc[-5]*100) if len(c)>=5 else 0
            d20 = float((c.iloc[-1]-c.iloc[0])/c.iloc[0]*100) if len(c)>=20 else 0
            vol_r = float(df["Volume"].iloc[-5:].mean() / (df["Volume"].mean()+1))
            # RSI
            delta = c.diff(); g = delta.clip(lower=0).rolling(14).mean()
            l = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = float(100 - 100/(1+g.iloc[-1]/(l.iloc[-1]+1e-9)))
            return {"Ticker":sym,"Name":name,"Sector":sector,
                    "Price":round(cur,2),"1D%":round(d1,2),
                    "5D%":round(d5,2),"1M%":round(d20,2),
                    "VolRatio":round(vol_r,2),"RSI":round(rsi,1)}
        except Exception:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(_fetch, sym, name, sec): sym
                for sym,(name,sec) in tickers.items()}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)
    return pd.DataFrame(rows).sort_values("1D%", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800)
def sector_returns(tickers: dict, days: int = 20) -> pd.DataFrame:
    snap = ndx_snapshot(tickers)
    if snap.empty:
        return pd.DataFrame()
    col = "1M%" if days >= 20 else "5D%"
    grp = snap.groupby("Sector")[col].mean().reset_index()
    grp.columns = ["Sector", "AvgReturn"]
    return grp.sort_values("AvgReturn", ascending=False)


@st.cache_data(ttl=1800)
def jp_sector_performance(ticker_map: dict, period_days: int = 20) -> pd.DataFrame:
    end_d = datetime.today()
    start_d = end_d - timedelta(days=period_days + 10)
    sec_ret: dict = {}
    for ticker, (name, sector) in ticker_map.items():
        try:
            df = yf.download(ticker, start=start_d, end=end_d, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 2: continue
            c = df["Close"].dropna()
            ret = float((c.iloc[-1]-c.iloc[0])/c.iloc[0]*100)
            sec_ret.setdefault(sector, []).append(ret)
        except Exception:
            continue
    rows = [{"業種":s,"平均リターン(%)":round(np.mean(v),2),"銘柄数":len(v),
             "上昇率(%)":round(sum(1 for x in v if x>0)/len(v)*100,1)}
            for s,v in sec_ret.items()]
    return pd.DataFrame(rows).sort_values("平均リターン(%)", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800)
def jp_volume_surge(ticker_map: dict, surge_ratio: float = 2.0) -> pd.DataFrame:
    rows = []
    end_d = datetime.today(); start_d = end_d - timedelta(days=30)
    for ticker,(name,sector) in ticker_map.items():
        try:
            df = yf.download(ticker, start=start_d, end=end_d, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df)<10: continue
            v = df["Volume"].dropna()
            r_avg = v.iloc[-5:].mean(); b_avg = v.iloc[-20:-5].mean()
            if b_avg == 0: continue
            ratio = r_avg / b_avg
            if ratio >= surge_ratio:
                p_chg = float((df["Close"].iloc[-1]-df["Close"].iloc[-5])/df["Close"].iloc[-5]*100)
                rows.append({"企業名":name,"業種":sector,"出来高倍率":round(ratio,2),
                             "株価変化率(5日%)":round(p_chg,2),"現在値":round(float(df["Close"].iloc[-1]),1)})
        except Exception:
            continue
    df_r = pd.DataFrame(rows)
    if not df_r.empty:
        df_r = df_r.sort_values("出来高倍率", ascending=False).reset_index(drop=True)
    return df_r


@st.cache_data(ttl=1800)
def jp_52week(ticker_map: dict) -> pd.DataFrame:
    rows = []
    end_d = datetime.today(); start_d = end_d - timedelta(days=365)
    for ticker,(name,sector) in ticker_map.items():
        try:
            df = yf.download(ticker, start=start_d, end=end_d, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df)<50: continue
            h52 = float(df["High"].max()); l52 = float(df["Low"].min())
            cur = float(df["Close"].iloc[-1])
            rows.append({"企業名":name,"業種":sector,"現在値":round(cur,1),
                         "52週高値":round(h52,1),"52週安値":round(l52,1),
                         "高値乖離(%)":round((cur-h52)/h52*100,2),
                         "安値乖離(%)":round((cur-l52)/l52*100,2),
                         "新高値":"✅" if float(df["High"].iloc[-1])>=h52*0.995 else "",
                         "新安値":"⚠️" if float(df["Low"].iloc[-1])<=l52*1.005 else ""})
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def jp_momentum(ticker_map: dict) -> pd.DataFrame:
    rows = []
    end_d = datetime.today(); start_d = end_d - timedelta(days=30)
    for ticker,(name,sector) in ticker_map.items():
        try:
            df = yf.download(ticker, start=start_d, end=end_d, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df)<10: continue
            pc = float((df["Close"].iloc[-1]-df["Close"].iloc[0])/df["Close"].iloc[0]*100)
            vc = float((df["Volume"].iloc[-5:].mean()-df["Volume"].mean())/(df["Volume"].mean()+1)*100)
            score = pc * np.log1p(max(vc,0)/100+1)
            rows.append({"企業名":name,"業種":sector,"モメンタムスコア":round(score,3),
                         "株価騰落率(%)":round(pc,2),"出来高変化率(%)":round(vc,2),
                         "現在値":round(float(df["Close"].iloc[-1]),1)})
        except Exception:
            continue
    df_r = pd.DataFrame(rows)
    if not df_r.empty:
        df_r = df_r.sort_values("モメンタムスコア", ascending=False).reset_index(drop=True)
    return df_r


@st.cache_data(ttl=1800)
def jp_ma_deviation(ticker_map: dict) -> pd.DataFrame:
    rows = []
    end_d = datetime.today(); start_d = end_d - timedelta(days=260)
    for ticker,(name,sector) in ticker_map.items():
        try:
            df = yf.download(ticker, start=start_d, end=end_d, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df)<75: continue
            c = df["Close"].dropna(); cur = float(c.iloc[-1])
            ma25 = float(c.rolling(25).mean().iloc[-1])
            ma75 = float(c.rolling(75).mean().iloc[-1]) if len(c)>=75 else None
            rows.append({"企業名":name,"業種":sector,"現在値":round(cur,1),
                         "25日MA乖離(%)":round((cur-ma25)/ma25*100,2),
                         "75日MA乖離(%)":round((cur-ma75)/ma75*100,2) if ma75 else None})
        except Exception:
            continue
    df_r = pd.DataFrame(rows)
    if not df_r.empty:
        df_r = df_r.sort_values("25日MA乖離(%)", ascending=False).reset_index(drop=True)
    return df_r


@st.cache_data(ttl=1800)
def jp_cross_signals(ticker_map: dict, lookback: int = 10) -> pd.DataFrame:
    rows = []
    end_d = datetime.today(); start_d = end_d - timedelta(days=120)
    for ticker,(name,sector) in ticker_map.items():
        try:
            df = yf.download(ticker, start=start_d, end=end_d, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df)<75: continue
            c = df["Close"].dropna()
            d = c.rolling(25).mean() - c.rolling(75).mean()
            for i in range(max(1,len(d)-lookback), len(d)):
                if pd.isna(d.iloc[i]) or pd.isna(d.iloc[i-1]): continue
                if d.iloc[i-1]<0 and d.iloc[i]>=0:
                    rows.append({"企業名":name,"業種":sector,"シグナル":"🟡 ゴールデンクロス",
                                 "発生日":str(d.index[i])[:10],"現在値":round(float(c.iloc[-1]),1)}); break
                elif d.iloc[i-1]>0 and d.iloc[i]<=0:
                    rows.append({"企業名":name,"業種":sector,"シグナル":"💀 デッドクロス",
                                 "発生日":str(d.index[i])[:10],"現在値":round(float(c.iloc[-1]),1)}); break
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def jp_correlation_divergence(ticker_map: dict, days: int = 60, window: int = 20) -> pd.DataFrame:
    end_d = datetime.today(); start_d = end_d - timedelta(days=days+10)
    bm = yf.download("^N225", start=start_d, end=end_d, progress=False)
    if isinstance(bm.columns, pd.MultiIndex):
        bm.columns = bm.columns.droplevel(1)
    mkt = bm["Close"].pct_change().dropna()
    rows = []
    for ticker,(name,sector) in ticker_map.items():
        try:
            df = yf.download(ticker, start=start_d, end=end_d, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df)<window+5: continue
            ret = df["Close"].pct_change().dropna()
            com = ret.index.intersection(mkt.index)
            if len(com)<window+5: continue
            r = ret.loc[com]; m = mkt.loc[com]
            cl = float(r.corr(m)); cr = float(r.iloc[-window:].corr(m.iloc[-window:]))
            pc = float((df["Close"].iloc[-1]-df["Close"].iloc[-5])/df["Close"].iloc[-5]*100)
            rows.append({"企業名":name,"業種":sector,"長期相関":round(cl,3),
                         "直近相関":round(cr,3),"相関乖離度":round(cl-cr,3),
                         "直近5日株価変化(%)":round(pc,2)})
        except Exception:
            continue
    df_r = pd.DataFrame(rows)
    if not df_r.empty:
        df_r = df_r.sort_values("相関乖離度", ascending=False).reset_index(drop=True)
    return df_r


# ════════════════════════════════════════════════════════════════
# チャート系
# ════════════════════════════════════════════════════════════════

def candlestick_chart(df: pd.DataFrame, ticker: str, height: int = 400) -> go.Figure:
    df = df.tail(120).copy()
    df.index = pd.to_datetime(df.index)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=ticker,
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ))
    for p, c in [(25,"#00d4ff"),(50,"#ffaa00"),(200,"#ff88ff")]:
        if len(df) >= p:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(p).mean(),
                                     name=f"MA{p}", line=dict(color=c,width=1.5), opacity=0.8))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e0e0e0"), height=height,
        xaxis=dict(gridcolor="#333", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#333"),
        legend=dict(bgcolor="#1a1a2e", orientation="h", y=1.02),
        margin=dict(l=10,r=10,t=30,b=10),
        title=f"{ticker} 株価チャート",
    )
    return fig


def sector_heatmap_fig(df_snap: pd.DataFrame) -> go.Figure:
    """NASDAQ100セクター別ヒートマップ"""
    pivot = df_snap.groupby("Sector").agg(
        Ret1D=("1D%","mean"), Ret5D=("5D%","mean"), Ret1M=("1M%","mean"), Count=("Ticker","count")
    ).reset_index()
    fig = go.Figure(data=go.Heatmap(
        z=[pivot["Ret1D"].tolist(), pivot["Ret5D"].tolist(), pivot["Ret1M"].tolist()],
        x=pivot["Sector"].tolist(),
        y=["1日","5日","1ヶ月"],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:+.1f}%" for v in pivot["Ret1D"]],
              [f"{v:+.1f}%" for v in pivot["Ret5D"]],
              [f"{v:+.1f}%" for v in pivot["Ret1M"]]],
        texttemplate="%{text}", textfont=dict(size=10),
    ))
    fig.update_layout(
        title="NASDAQ100 セクター別リターン ヒートマップ",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e0e0e0"),
        height=300, margin=dict(l=10,r=10,t=40,b=10),
        xaxis=dict(tickangle=-30),
    )
    return fig


# ════════════════════════════════════════════════════════════════
# AI エージェント（個別銘柄）
# ════════════════════════════════════════════════════════════════

def run_agent(ticker: str, log_ph, status_ph) -> dict:
    logs = []
    def log(cls, icon, msg, detail=""):
        logs.append((cls, icon, msg, detail))
        html = "".join(
            f'<div class="alog alog-{c}"><b>{i}</b> {m}'
            f'{"<br><span style=\'color:#888;font-size:0.75rem\'>" + d + "</span>" if d else ""}'
            f'</div>' for c,i,m,d in logs)
        log_ph.markdown(html, unsafe_allow_html=True)

    results = {}

    # Step1: 価格
    status_ph.info("📡 Step 1/4 価格データ取得中...")
    log("","📡","tool_get_price_data",f"ticker={ticker}")
    df = get_ohlcv(ticker, 400)
    if df.empty:
        log("err","❌","価格データ取得失敗")
        return {"error": f"{ticker} の価格データを取得できませんでした"}
    c = df["Close"].dropna()
    cur = float(c.iloc[-1])
    d1  = round(float((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100),2) if len(c)>=2 else 0
    d5  = round(float((c.iloc[-1]-c.iloc[-5])/c.iloc[-5]*100),2) if len(c)>=5 else 0
    d20 = round(float((c.iloc[-1]-c.iloc[-20])/c.iloc[-20]*100),2) if len(c)>=20 else 0
    d252= round(float((c.iloc[-1]-c.iloc[0])/c.iloc[0]*100),2)
    h52 = round(float(df["High"].max()),2); l52 = round(float(df["Low"].min()),2)
    rets = c.pct_change().dropna()
    sharpe = round(float(rets.mean()*252/(rets.std()*np.sqrt(252)+1e-9)),3)
    delta = c.diff(); g = delta.clip(lower=0).rolling(14).mean()
    lo = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = round(float(100-100/(1+g.iloc[-1]/(lo.iloc[-1]+1e-9))),1)
    ma25 = round(float(c.rolling(25).mean().iloc[-1]),2) if len(c)>=25 else None
    ma50 = round(float(c.rolling(50).mean().iloc[-1]),2) if len(c)>=50 else None
    ma200= round(float(c.rolling(200).mean().iloc[-1]),2) if len(c)>=200 else None
    price_data = {"current":cur,"d1":d1,"d5":d5,"d20":d20,"d252":d252,
                  "h52":h52,"l52":l52,"sharpe":sharpe,"rsi":rsi,
                  "ma25":ma25,"ma50":ma50,"ma200":ma200,"df":df}
    results["price"] = price_data
    log("ok","✅","価格取得完了",f"${cur:,.2f} | 1日:{d1:+.1f}% | RSI:{rsi}")

    # Step2: ファンダメンタルズ
    status_ph.info("📊 Step 2/4 決算・財務データ確認中...")
    log("","📊","tool_get_fundamentals",f"ticker={ticker}")
    info = get_info(ticker)
    fund = {
        "name": info.get("longName") or info.get("shortName", ticker),
        "sector": info.get("sector","N/A"), "industry": info.get("industry","N/A"),
        "mktcap": info.get("marketCap"), "pe": info.get("trailingPE"),
        "fpe": info.get("forwardPE"), "pb": info.get("priceToBook"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "rev": info.get("totalRevenue"), "gm": info.get("grossMargins"),
        "om": info.get("operatingMargins"), "npm": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"), "rev_g": info.get("revenueGrowth"),
        "earn_g": info.get("earningsGrowth"), "de": info.get("debtToEquity"),
        "fcf": info.get("freeCashflow"), "div": info.get("dividendYield"),
        "beta": info.get("beta"), "target": info.get("targetMeanPrice"),
        "rec": info.get("recommendationKey",""),
        "summary": (info.get("longBusinessSummary","")[:400]),
    }
    results["fund"] = fund
    log("ok","✅","財務取得完了",
        f"{fund['name']} | {fund['sector']} | PER:{fmt_num(fund['pe'])}x | 純利益率:{fmt_pct(fund['npm'])}")

    # Step3: ニュース
    status_ph.info("📰 Step 3/4 ニュース収集中...")
    log("","📰","tool_get_news",f"Yahoo Finance + Reuters + Google News")
    news_items = fetch_stock_news(ticker, fund["name"])
    results["news"] = news_items
    log("ok","✅","ニュース収集完了",f"{len(news_items)}件取得")

    # Step4: セクター比較
    status_ph.info("🌐 Step 4/4 セクター・テーマ分析中...")
    log("","🌐","tool_sector_analysis",f"sector={fund['sector']}")
    PEERS = {
        "Technology":["AAPL","MSFT","NVDA","AMD","INTC"],
        "Communication":["GOOGL","META","NFLX","CMCSA"],
        "Consumer Cyclical":["AMZN","TSLA","MELI","BKNG"],
        "Healthcare":["AMGN","GILD","REGN","VRTX"],
        "Financial":["PYPL","HOOD"],
        "Industrials":["HON","CSX","CPRT"],
        "Consumer Defensive":["COST","PEP","MDLZ"],
    }
    peers = [p for p in PEERS.get(fund["sector"],[]) if p != ticker][:3]
    peer_ret = {}
    for p in peers:
        try:
            pdf = get_ohlcv(p, 95)
            if not pdf.empty:
                pc = pdf["Close"].dropna()
                peer_ret[p] = round(float((pc.iloc[-1]-pc.iloc[0])/pc.iloc[0]*100),2)
        except Exception:
            pass
    try:
        spy = get_ohlcv("SPY", 95)
        sc = spy["Close"].dropna()
        spy_ret = round(float((sc.iloc[-1]-sc.iloc[0])/sc.iloc[0]*100),2)
    except Exception:
        spy_ret = None
    results["theme"] = {"peers": peer_ret, "spy": spy_ret, "sector": fund["sector"]}
    log("ok","✅","テーマ分析完了",
        "ピア: "+", ".join(f"{k}:{v:+.1f}%" for k,v in peer_ret.items()))

    # Step5: AIレポート
    status_ph.info("🤖 AIが統合レポート生成中...")
    log("","🤖","AIレポート生成","Gemini 2.5 Pro")
    news_txt = "\n".join(f"- [{n['source']}] {n['title']}" for n in news_items[:8])
    peer_txt = "\n".join(f"  {k}: {v:+.2f}%" for k,v in peer_ret.items())
    prompt = f"""あなたはウォール街のトップアナリストです。
{fund['name']}({ticker})の機関投資家向け総合分析レポートを作成してください。

【価格】現在値:${price_data['current']:,.2f} / 1日:{d1:+.1f}% / 1週:{d5:+.1f}% / 1ヶ月:{d20:+.1f}% / 1年:{d252:+.1f}%
RSI:{rsi} / シャープ:{sharpe} / MA25:${ma25} / MA50:${ma50} / MA200:${ma200}
52週高値:${h52}({round((cur-h52)/h52*100,1):+.1f}%) / 安値:${l52}

【財務】時価総額:{fmt_large(fund['mktcap'])} / PER:{fmt_num(fund['pe'])}x / フォワードPER:{fmt_num(fund['fpe'])}x
粗利:{fmt_pct(fund['gm'])} / 営業益:{fmt_pct(fund['om'])} / 純益:{fmt_pct(fund['npm'])}
ROE:{fmt_pct(fund['roe'])} / 売上成長:{fmt_pct(fund['rev_g'])} / 利益成長:{fmt_pct(fund['earn_g'])}
D/E:{fmt_num(fund['de'])} / FCF:{fmt_large(fund['fcf'])} / ベータ:{fmt_num(fund['beta'])}
アナリスト推奨:{fund['rec']} / 目標株価:${fmt_num(fund['target'])}

【ニュース（直近）】
{news_txt if news_txt else 'なし'}

【セクター比較（3ヶ月）】
{peer_txt if peer_txt else 'データなし'}
SPY: {f"{spy_ret:+.2f}%" if spy_ret is not None else "N/A"}

以下の構成で分析してください（日本語）:
### 📌 エグゼクティブサマリー
### 💰 テクニカル分析
### 📊 ファンダメンタルズ評価
### 📰 ニュース・センチメント（強気/弱気/中立を明示）
### 🌐 セクターポジション
### ⚠️ 主要リスク（3点）
### 🎯 投資判断（5段階）と目標株価レンジ（3〜6ヶ月）
※投資助言ではなく参考情報です。"""
    ai_report, ai_name = ai_call(prompt)
    results["ai_report"] = ai_report
    results["ai_name"] = ai_name
    log("ok","🎉","レポート生成完了！")
    status_ph.success("✅ 分析完了！")
    return results


# ════════════════════════════════════════════════════════════════
# HTML レポート生成
# ════════════════════════════════════════════════════════════════

def make_html_report(data: dict, ticker: str) -> str:
    p = data.get("price", {}); f = data.get("fund", {})
    news = data.get("news", []); theme = data.get("theme", {})
    ai  = data.get("ai_report", ""); now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cur = p.get("current", 0); d1 = p.get("d1", 0)
    dc = "#00ff88" if d1 >= 0 else "#ff4444"

    news_html = "".join(f"""
    <div style="border-left:3px solid #00d4ff;padding:8px 12px;margin:5px 0;background:#1a1a2e;border-radius:4px">
      <div style="font-size:0.72rem;color:#00d4ff">{n['source']} | {n.get('date','')}</div>
      <div style="margin-top:3px"><a href="{n.get('link','#')}" style="color:#e0e0e0;text-decoration:none" target="_blank">{n['title']}</a></div>
    </div>""" for n in news[:8])

    peer_html = "".join(
        f'<span style="background:#1a1a2e;padding:5px 10px;border-radius:20px;margin:3px;display:inline-block;border:1px solid {"#00ff88" if v>=0 else "#ff4444"}">{k}: <b style="color:{"#00ff88" if v>=0 else "#ff4444"}">{v:+.1f}%</b></span>'
        for k, v in (theme.get("peers") or {}).items()
    )
    ai_html = re.sub(r"### (.+)", r"<h3 style='color:#00d4ff;margin-top:16px'>\1</h3>",
               re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#ffaa00'>\1</b>",
               ai.replace("\n","<br>")))

    return f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<title>{f.get('name',ticker)} 分析レポート</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif;line-height:1.6}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px}}
.hdr{{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:12px;padding:20px;margin-bottom:16px;border:1px solid #00d4ff33}}
.grid{{display:grid;gap:12px;margin-bottom:16px}}
.g3{{grid-template-columns:repeat(3,1fr)}} .g2{{grid-template-columns:repeat(2,1fr)}}
.card{{background:#1a1a2e;border-radius:8px;padding:14px;border:1px solid #ffffff11}}
.card h3{{color:#00d4ff;font-size:0.82rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:0.83rem}}
td{{padding:6px 8px;border-bottom:1px solid #ffffff0f}}
.sec{{background:linear-gradient(135deg,#0f3460,#16213e);border-radius:10px;padding:16px;margin-bottom:14px;border:1px solid #00d4ff22}}
.sec h2{{color:#00d4ff;margin-bottom:12px}}
.foot{{text-align:center;color:#555;font-size:0.73rem;margin-top:24px;padding:12px}}
</style></head><body><div class="wrap">
<div class="hdr">
  <h1 style="color:#00d4ff;font-size:1.6rem">{f.get('name',ticker)} <span style="color:#888;font-size:1rem">({ticker})</span></h1>
  <div style="color:#888;font-size:0.82rem;margin-top:4px">{f.get('sector','N/A')} | {f.get('industry','N/A')} | 生成: {now}</div>
</div>
<div class="grid g3">
  <div class="card" style="text-align:center">
    <div style="font-size:2rem;font-weight:bold;color:#00d4ff">${cur:,.2f}</div>
    <div style="color:{dc};font-weight:bold">{d1:+.2f}% (1日)</div>
  </div>
  <div class="card"><table>
    <tr><td>1週間</td><td style="color:{'#00ff88' if p.get('d5',0)>=0 else '#ff4444'};font-weight:bold">{p.get('d5',0):+.2f}%</td></tr>
    <tr><td>1ヶ月</td><td style="color:{'#00ff88' if p.get('d20',0)>=0 else '#ff4444'};font-weight:bold">{p.get('d20',0):+.2f}%</td></tr>
    <tr><td>1年</td><td style="color:{'#00ff88' if p.get('d252',0)>=0 else '#ff4444'};font-weight:bold">{p.get('d252',0):+.2f}%</td></tr>
  </table></div>
  <div class="card"><table>
    <tr><td>RSI(14)</td><td style="color:{'#ff4444' if p.get('rsi',50)>70 else '#00ff88' if p.get('rsi',50)<30 else '#ffaa00'};font-weight:bold">{p.get('rsi',0)}</td></tr>
    <tr><td>シャープ</td><td>{p.get('sharpe',0)}</td></tr>
    <tr><td>52週高値比</td><td>{round((cur-p.get('h52',cur))/p.get('h52',cur+1)*100,1):+.1f}%</td></tr>
  </table></div>
</div>
<div class="grid g2">
  <div class="card"><h3>バリュエーション</h3><table>
    <tr><td>時価総額</td><td>{fmt_large(f.get('mktcap'))}</td></tr>
    <tr><td>PER</td><td>{fmt_num(f.get('pe'))}x</td></tr>
    <tr><td>PBR</td><td>{fmt_num(f.get('pb'))}x</td></tr>
    <tr><td>PSR</td><td>{fmt_num(f.get('ps'))}x</td></tr>
  </table></div>
  <div class="card"><h3>収益性</h3><table>
    <tr><td>粗利益率</td><td>{fmt_pct(f.get('gm'))}</td></tr>
    <tr><td>営業利益率</td><td>{fmt_pct(f.get('om'))}</td></tr>
    <tr><td>純利益率</td><td>{fmt_pct(f.get('npm'))}</td></tr>
    <tr><td>ROE</td><td>{fmt_pct(f.get('roe'))}</td></tr>
  </table></div>
</div>
<div class="sec"><h2>🌐 セクター比較（3ヶ月）</h2>
  SPY: <b style="color:{'#00ff88' if (theme.get('spy') or 0)>=0 else '#ff4444'}">{f"{theme.get('spy'):+.1f}%" if theme.get('spy') is not None else 'N/A'}</b>&nbsp;&nbsp;
  {peer_html}
</div>
<div class="sec"><h2>📰 最新ニュース</h2>{news_html if news_html else '<p style="color:#888">なし</p>'}</div>
<div class="sec"><h2>🤖 AI総合分析</h2><div style="font-size:0.88rem">{ai_html}</div></div>
<div class="foot">⚠️ 投資助言ではありません。データソース: Yahoo Finance, Reuters | AI: Gemini 2.5 Pro | {now}</div>
</div></body></html>"""


# ════════════════════════════════════════════════════════════════
# メイン UI
# ════════════════════════════════════════════════════════════════

st.title("📊 Stock Intelligence Hub")
st.caption("NASDAQ100 市場俯瞰 × AI個別銘柄エージェント × 日本株マルチ分析")

# ── タブ ─────────────────────────────────────────────────────
TAB_LABELS = [
    "🌐 NASDAQ100",
    "🤖 AI個別分析",
    "🔄 JP セクター",
    "🔥 JP 需給",
    "📈 JP 価格パターン",
    "💡 JP モメンタム",
    "📰 個別ニュース",
    "🗺️ 市場ニュース",
]
tabs = st.tabs(TAB_LABELS)

# ══════════════════════════════════════════════════════════════
# Tab 0: NASDAQ100 ダッシュボード
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-head">🌐 NASDAQ100 マーケットダッシュボード</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2,2,3])
    with col_ctrl1:
        ndx_period = st.selectbox("期間", ["1日","5日","1ヶ月"], index=0, key="ndx_period")
    with col_ctrl2:
        top_n_ndx = st.slider("ランキング表示数", 5, 20, 10, key="top_n_ndx")
    with col_ctrl3:
        run_ndx = st.button("▶ NASDAQ100 分析実行", type="primary", key="run_ndx")

    if run_ndx:
        with st.spinner("NASDAQ100 全銘柄データ取得中（初回は30秒ほどかかります）..."):
            snap = ndx_snapshot(NDX100)

        if snap.empty:
            st.error("データ取得失敗")
        else:
            ret_col = {"1日":"1D%","5日":"5D%","1ヶ月":"1M%"}[ndx_period]

            # KPI行
            rising  = (snap[ret_col] > 0).sum()
            falling = (snap[ret_col] < 0).sum()
            avg_ret = snap[ret_col].mean()
            top1    = snap.iloc[0]
            bot1    = snap.iloc[-1]

            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("📈 上昇銘柄", f"{rising}")
            k2.metric("📉 下落銘柄", f"{falling}")
            k3.metric("📊 平均リターン", f"{avg_ret:+.2f}%")
            k4.metric("🏆 最強", f"{top1['Ticker']}", f"{top1[ret_col]:+.1f}%")
            k5.metric("💀 最弱", f"{bot1['Ticker']}", f"{bot1[ret_col]:+.1f}%")

            st.divider()

            # セクターヒートマップ
            fig_heat = sector_heatmap_fig(snap)
            st.plotly_chart(fig_heat, use_container_width=True)

            st.divider()

            # 上昇・下落ランキング
            col_up, col_dn = st.columns(2)
            with col_up:
                st.markdown(f"#### 🟢 上昇ランキング TOP{top_n_ndx}")
                top_df = snap.nlargest(top_n_ndx, ret_col)[["Ticker","Name","Sector","Price",ret_col,"RSI","VolRatio"]]
                st.dataframe(
                    top_df.style.format({ret_col:"{:+.2f}%","Price":"${:,.2f}","RSI":"{:.1f}","VolRatio":"{:.2f}x"}),
                    use_container_width=True, hide_index=True
                )
            with col_dn:
                st.markdown(f"#### 🔴 下落ランキング BOTTOM{top_n_ndx}")
                bot_df = snap.nsmallest(top_n_ndx, ret_col)[["Ticker","Name","Sector","Price",ret_col,"RSI","VolRatio"]]
                st.dataframe(
                    bot_df.style.format({ret_col:"{:+.2f}%","Price":"${:,.2f}","RSI":"{:.1f}","VolRatio":"{:.2f}x"}),
                    use_container_width=True, hide_index=True
                )

            st.divider()

            # 出来高急増（VolRatio >= 2）
            surge = snap[snap["VolRatio"] >= 2.0].sort_values("VolRatio", ascending=False)
            if not surge.empty:
                st.markdown("#### 🔥 出来高急増銘柄（直近5日平均 ÷ 全期間平均 ≥ 2.0x）")
                st.dataframe(
                    surge[["Ticker","Name","Sector","Price",ret_col,"VolRatio","RSI"]].style.format(
                        {ret_col:"{:+.2f}%","Price":"${:,.2f}","VolRatio":"{:.2f}x","RSI":"{:.1f}"}),
                    use_container_width=True, hide_index=True
                )

            st.divider()

            # RSIスクリーニング
            col_overbought, col_oversold = st.columns(2)
            with col_overbought:
                ob = snap[snap["RSI"] >= 70].sort_values("RSI", ascending=False)
                st.markdown(f"#### 🔴 過買い銘柄（RSI≥70） — {len(ob)}件")
                if not ob.empty:
                    st.dataframe(ob[["Ticker","Name","Price","RSI",ret_col]].style.format(
                        {"Price":"${:,.2f}","RSI":"{:.1f}",ret_col:"{:+.2f}%"}),
                        use_container_width=True, hide_index=True)
            with col_oversold:
                os_ = snap[snap["RSI"] <= 30].sort_values("RSI")
                st.markdown(f"#### 🟢 過売り銘柄（RSI≤30） — {len(os_)}件")
                if not os_.empty:
                    st.dataframe(os_[["Ticker","Name","Price","RSI",ret_col]].style.format(
                        {"Price":"${:,.2f}","RSI":"{:.1f}",ret_col:"{:+.2f}%"}),
                        use_container_width=True, hide_index=True)

            st.divider()

            # AI市場コメント
            top5_str = snap.nlargest(5, ret_col)[["Ticker","Name","Sector",ret_col,"RSI"]].to_string(index=False)
            bot5_str = snap.nsmallest(5, ret_col)[["Ticker","Name","Sector",ret_col,"RSI"]].to_string(index=False)
            surge_str = surge.head(5)[["Ticker","Name",ret_col,"VolRatio"]].to_string(index=False) if not surge.empty else "なし"
            prompt_ndx = f"""あなたはNASDAQ専門のストラテジストです。
以下は本日のNASDAQ100の状況です（期間:{ndx_period}）。

【上昇TOP5】
{top5_str}

【下落BOTTOM5】
{bot5_str}

【出来高急増（上位5）】
{surge_str}

上昇{rising}銘柄 / 下落{falling}銘柄 / 平均リターン:{avg_ret:+.2f}%

以下を400文字以内で分析してください（日本語）:
1. 現在の市場センチメント（強気/弱気/中立）
2. 牽引セクターと売られているセクター
3. 注目すべき出来高・価格動向
4. 投資家へのアドバイス"""
            with st.spinner("AI市場分析中..."):
                comment, ai_name = ai_call(prompt_ndx)
            st.markdown("#### 🤖 AI市場総合レポート")
            st.info(f"{comment}\n\n_AI: {ai_name}_")

    else:
        st.info("「▶ NASDAQ100 分析実行」ボタンを押すと分析が始まります。\n\n"
                "- 🗺️ セクター別リターン ヒートマップ\n"
                "- 🏆 上昇/下落 ランキング\n"
                "- 🔥 出来高急増スクリーナー\n"
                "- 📊 RSI 過買い/過売りスクリーナー\n"
                "- 🤖 AI市場総合レポート")

# ══════════════════════════════════════════════════════════════
# Tab 1: AI 個別銘柄エージェント
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec-head">🤖 AI個別銘柄エージェント（米国株）</div>', unsafe_allow_html=True)

    # サンプル銘柄ボタン
    samples = ["AAPL","NVDA","TSLA","GOOGL","AMZN","MSFT","META","AMD","PLTR","ARM"]
    cols_s = st.columns(len(samples))
    for i, sym in enumerate(samples):
        if cols_s[i].button(sym, key=f"s_{sym}"):
            st.session_state["agent_ticker"] = sym

    col_in, col_btn = st.columns([5,1])
    with col_in:
        ticker_in = st.text_input("ティッカー入力（例: AAPL）",
                                  value=st.session_state.get("agent_ticker",""),
                                  key="ticker_in", label_visibility="collapsed",
                                  placeholder="例: AAPL, NVDA, TSLA ...")
    with col_btn:
        run_agent_btn = st.button("🚀 分析", type="primary", key="run_agent_btn")

    if run_agent_btn and ticker_in.strip():
        tkr = ticker_in.strip().upper()
        st.divider()

        col_log, col_main = st.columns([1, 2])
        with col_log:
            st.markdown("**🔧 思考プロセス**")
            log_ph = st.empty()
            status_ph = st.empty()
        with col_main:
            st.markdown("**📊 リアルタイム結果**")
            res_ph = st.empty()
            res_ph.info("🤔 エージェント分析中...")

        data = run_agent(tkr, log_ph, status_ph)
        res_ph.empty()

        if "error" in data:
            st.error(data["error"])
        else:
            p = data["price"]; f = data["fund"]
            st.divider()
            st.markdown(f"## {f.get('name',tkr)} ({tkr})")

            # コンパクトKPI
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("現在値", f"${p['current']:,.2f}", f"{p['d1']:+.2f}%")
            c2.metric("PER", f"{fmt_num(f.get('pe'))}x")
            c3.metric("RSI", f"{p['rsi']}", "過買い" if p['rsi']>70 else "過売り" if p['rsi']<30 else "中立")
            c4.metric("シャープ", f"{p['sharpe']}")
            c5.metric("時価総額", fmt_large(f.get("mktcap")))
            c6.metric("推奨", (f.get("rec") or "N/A").upper())

            # タブ
            t_chart, t_fund, t_news, t_theme, t_ai = st.tabs(
                ["📈 チャート","📊 財務","📰 ニュース","🌐 テーマ","🤖 AIレポート"])

            with t_chart:
                st.plotly_chart(candlestick_chart(p["df"], tkr, 380), use_container_width=True)
                # 出来高（コンパクト）
                vol_df = p["df"].tail(60).copy()
                vol_df.index = pd.to_datetime(vol_df.index)
                vc = ["#00ff88" if c>=o else "#ff4444"
                      for c,o in zip(vol_df["Close"],vol_df["Open"])]
                fig_v = go.Figure(go.Bar(x=vol_df.index, y=vol_df["Volume"], marker_color=vc))
                fig_v.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
                                    font=dict(color="#e0e0e0"),height=180,
                                    margin=dict(l=10,r=10,t=10,b=10),
                                    xaxis=dict(gridcolor="#333"),yaxis=dict(gridcolor="#333"))
                st.plotly_chart(fig_v, use_container_width=True)

            with t_fund:
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    st.markdown("**バリュエーション**")
                    st.table(pd.DataFrame({
                        "指標":["時価総額","PER","フォワードPER","PBR","PSR"],
                        "値":[fmt_large(f.get("mktcap")),f"{fmt_num(f.get('pe'))}x",
                              f"{fmt_num(f.get('fpe'))}x",f"{fmt_num(f.get('pb'))}x",
                              f"{fmt_num(f.get('ps'))}x"]
                    }))
                with col_f2:
                    st.markdown("**収益性**")
                    st.table(pd.DataFrame({
                        "指標":["粗利益率","営業利益率","純利益率","ROE","売上成長"],
                        "値":[fmt_pct(f.get("gm")),fmt_pct(f.get("om")),
                              fmt_pct(f.get("npm")),fmt_pct(f.get("roe")),fmt_pct(f.get("rev_g"))]
                    }))
                with col_f3:
                    st.markdown("**健全性**")
                    st.table(pd.DataFrame({
                        "指標":["FCF","D/E","ベータ","配当","目標株価"],
                        "値":[fmt_large(f.get("fcf")),fmt_num(f.get("de")),
                              fmt_num(f.get("beta")),fmt_pct(f.get("div")),
                              f"${fmt_num(f.get('target'))}"]
                    }))
                if f.get("summary"):
                    with st.expander("📝 事業概要"):
                        st.write(f["summary"])

            with t_news:
                news_items = data.get("news", [])
                if not news_items:
                    st.info("ニュースなし")
                else:
                    src_cnt = Counter(n["source"] for n in news_items)
                    sc = st.columns(len(src_cnt))
                    for i,(src,cnt) in enumerate(src_cnt.items()):
                        sc[i].metric(src, f"{cnt}件")
                    st.divider()
                    for n in news_items:
                        src_c = {"Yahoo Finance":"🟦","Reuters":"🟫","Google News":"🔵"}.get(n["source"],"⚪")
                        with st.expander(f"{src_c} {n['title'][:65]}{'...' if len(n['title'])>65 else ''}"):
                            col_a, col_b = st.columns([3,1])
                            with col_a:
                                st.markdown(f"**{n['title']}**")
                                if n.get("summary"): st.caption(n["summary"])
                            with col_b:
                                if n.get("date"): st.caption(f"📅 {n['date']}")
                                if n.get("link"): st.markdown(f"[🔗 開く]({n['link']})")
                    # AIニュース要約
                    if st.button("🤖 ニュースをAI要約", key="ai_news_btn"):
                        hdl = "\n".join(f"- [{n['source']}] {n['title']}" for n in news_items[:10])
                        prompt_news = f"""以下は{f.get('name',tkr)}の最新ニュースです。
{hdl}
投資家向けに200文字以内で:
1. センチメント: 強気/弱気/中立
2. 注目ポイント
3. 株価への影響"""
                        with st.spinner("AI要約中..."):
                            nc, nn = ai_call(prompt_news)
                        st.info(f"{nc}\n\n_AI: {nn}_")

            with t_theme:
                td = data.get("theme", {})
                peers = td.get("peers", {})
                spy_r = td.get("spy")
                if peers or spy_r:
                    all_t = [tkr] + list(peers.keys()) + (["SPY"] if spy_r else [])
                    all_r = [p.get("d20",0)] + list(peers.values()) + ([spy_r] if spy_r else [])
                    clrs  = ["#00d4ff" if t==tkr else ("#00ff88" if r>=0 else "#ff4444")
                             for t,r in zip(all_t,all_r)]
                    fig_p = go.Figure(go.Bar(x=all_t, y=all_r, marker_color=clrs,
                                            text=[f"{r:+.1f}%" for r in all_r], textposition="outside"))
                    fig_p.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
                                        font=dict(color="#e0e0e0"),height=320,
                                        yaxis=dict(gridcolor="#333",title="リターン (%)"),
                                        margin=dict(l=10,r=10,t=30,b=10),
                                        title="ピア比較（1ヶ月）")
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.info("比較データなし")

            with t_ai:
                st.markdown(data.get("ai_report","レポートなし"))

            # HTMLダウンロード
            st.divider()
            html_dl = make_html_report(data, tkr)
            st.download_button("📥 HTMLレポートをダウンロード", data=html_dl.encode("utf-8"),
                               file_name=f"{tkr}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                               mime="text/html", type="primary", use_container_width=True)

    elif not run_agent_btn:
        st.markdown("""
<div style="text-align:center;padding:40px 20px;color:#888">
<div style="font-size:3rem">🤖</div>
<div style="margin-top:12px;color:#ccc;font-size:1rem">ティッカーを入力して「🚀 分析」を押してください</div>
<div style="margin-top:8px;font-size:0.85rem">
AIエージェントが ① 価格 ② 財務 ③ ニュース ④ セクター を自律分析してレポートを生成します
</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Tab 2: JP セクターローテーション
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-head">🔄 日本株 セクターローテーション</div>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns([2,3])
    with col_r1:
        rot_period = st.selectbox("分析期間", [5,10,20,60],
                                  format_func=lambda x:{5:"1週間",10:"2週間",20:"1ヶ月",60:"3ヶ月"}[x],
                                  key="rot_period")
    with col_r2:
        run_rot = st.button("▶ セクター分析実行", type="primary", key="run_rot")

    if run_rot:
        with st.spinner("日本株セクターデータ取得中..."):
            df_sec = jp_sector_performance(JP_TICKERS, rot_period)
        if not df_sec.empty:
            top = df_sec.iloc[0]; bot = df_sec.iloc[-1]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("📈 最強セクター", top["業種"], f"{top['平均リターン(%)']:+.2f}%")
            c2.metric("📉 最弱セクター", bot["業種"], f"{bot['平均リターン(%)']:+.2f}%")
            c3.metric("🟢 上昇", f"{(df_sec['平均リターン(%)']>0).sum()}業種")
            c4.metric("🔴 下落", f"{(df_sec['平均リターン(%)']<0).sum()}業種")

            # 棒グラフ
            df_s = df_sec.sort_values("平均リターン(%)", ascending=True)
            clrs = ["#d32f2f" if v<0 else "#388e3c" for v in df_s["平均リターン(%)"]]
            fig_sec = go.Figure(go.Bar(
                y=df_s["業種"], x=df_s["平均リターン(%)"],
                orientation="h", marker_color=clrs,
                text=[f"{v:+.2f}%" for v in df_s["平均リターン(%)"]],
                textposition="outside",
            ))
            fig_sec.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",
                                  font=dict(color="#e0e0e0"),height=max(350,len(df_s)*28),
                                  xaxis=dict(gridcolor="#333",title="平均リターン(%)"),
                                  margin=dict(l=10,r=80,t=30,b=10),
                                  title="セクター別平均リターン")
            st.plotly_chart(fig_sec, use_container_width=True)

            st.dataframe(df_sec.style.format({"平均リターン(%)":"{:+.2f}","上昇率(%)":"{:.1f}"}),
                         use_container_width=True, hide_index=True)

            top5 = df_sec.head(5)[["業種","平均リターン(%)","上昇率(%)"]].to_string(index=False)
            bot5 = df_sec.tail(5)[["業種","平均リターン(%)","上昇率(%)"]].to_string(index=False)
            period_label = {5:"1週間",10:"2週間",20:"1ヶ月",60:"3ヶ月"}[rot_period]
            with st.spinner("AI分析中..."):
                comment, ai_name = ai_call(
                    f"日本株ストラテジストとして{period_label}のセクターローテーションを400文字以内で分析:\n"
                    f"【買われ上位5】\n{top5}\n\n【売られ下位5】\n{bot5}\n\n"
                    "1.ローテーションの特徴 2.背景 3.投資アドバイス")
            st.info(f"🤖 **AIセクター分析（{ai_name}）**\n\n{comment}")
    else:
        st.info("「▶ セクター分析実行」を押してください")

# ══════════════════════════════════════════════════════════════
# Tab 3: JP 需給スクリーナー
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec-head">🔥 日本株 需給スクリーナー</div>', unsafe_allow_html=True)
    col_v1, col_v2 = st.columns([2,3])
    with col_v1:
        surge_thr = st.slider("出来高急増閾値（倍）", 1.5, 5.0, 2.0, 0.5, key="surge_thr")
    with col_v2:
        run_vol = st.button("▶ 需給分析実行", type="primary", key="run_vol")

    if run_vol:
        with st.spinner("出来高データ取得中..."):
            df_surge = jp_volume_surge(JP_TICKERS, surge_thr)
        if df_surge.empty:
            st.info(f"出来高が{surge_thr}倍以上の銘柄は現在検出されませんでした")
        else:
            st.success(f"🔺 {len(df_surge)}銘柄検出")
            st.dataframe(df_surge.style.format({"出来高倍率":"{:.2f}x","株価変化率(5日%)":"{:+.2f}"}),
                         use_container_width=True, hide_index=True)
            top5 = df_surge.head(5)[["企業名","業種","出来高倍率","株価変化率(5日%)"]].to_string(index=False)
            with st.spinner("AI分析中..."):
                comment, ai_name = ai_call(
                    f"以下は出来高急増の日本株上位5社。投資家向けに200文字以内でコメント:\n{top5}\n"
                    "1.機関投資家の動きか 2.業種テーマ 3.注意点")
            st.info(f"🤖 **AI需給解説（{ai_name}）**\n\n{comment}")
    else:
        st.info("「▶ 需給分析実行」を押してください")

# ══════════════════════════════════════════════════════════════
# Tab 4: JP 価格パターン
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-head">📈 日本株 価格パターン分析</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([2,3])
    with col_p1:
        cross_lb = st.slider("クロス検出 直近何日", 3, 20, 10, key="cross_lb")
    with col_p2:
        run_price = st.button("▶ 価格パターン分析実行", type="primary", key="run_price")

    if run_price:
        # 52週高安値
        st.markdown("#### 🏔️ 52週高値・安値")
        with st.spinner("52週データ取得中..."):
            df_52 = jp_52week(JP_TICKERS)
        if not df_52.empty:
            nh = df_52[df_52["新高値"]!=""]; nl = df_52[df_52["新安値"]!=""]
            hl_idx = len(nh)/max(len(nh)+len(nl),1)*100
            c1,c2,c3 = st.columns(3)
            c1.metric("🔺 新高値銘柄", f"{len(nh)}件")
            c2.metric("🔻 新安値銘柄", f"{len(nl)}件")
            c3.metric("📊 ハイローIndex", f"{hl_idx:.1f}%",
                      "強気" if hl_idx>=70 else "弱気" if hl_idx<=30 else "中立")
            col_52a, col_52b = st.columns(2)
            with col_52a:
                st.markdown("**新高値**")
                if not nh.empty:
                    st.dataframe(nh[["企業名","業種","現在値","52週高値","高値乖離(%)"]],
                                 use_container_width=True, hide_index=True)
            with col_52b:
                st.markdown("**新安値**")
                if not nl.empty:
                    st.dataframe(nl[["企業名","業種","現在値","52週安値","安値乖離(%)"]],
                                 use_container_width=True, hide_index=True)

        st.divider()

        # MA乖離率
        st.markdown("#### 📐 移動平均乖離率")
        with st.spinner("MA計算中..."):
            df_ma = jp_ma_deviation(JP_TICKERS)
        if not df_ma.empty:
            cm1, cm2 = st.columns(2)
            with cm1:
                st.markdown("**🔴 上方乖離 上位10（買われすぎ）**")
                st.dataframe(df_ma.head(10)[["企業名","業種","現在値","25日MA乖離(%)","75日MA乖離(%)"]].style.format(
                    {"25日MA乖離(%)":"{:+.2f}","75日MA乖離(%)":"{:+.2f}"}),
                    use_container_width=True, hide_index=True)
            with cm2:
                st.markdown("**🟢 下方乖離 下位10（売られすぎ）**")
                st.dataframe(df_ma.tail(10)[["企業名","業種","現在値","25日MA乖離(%)","75日MA乖離(%)"]].style.format(
                    {"25日MA乖離(%)":"{:+.2f}","75日MA乖離(%)":"{:+.2f}"}),
                    use_container_width=True, hide_index=True)

        st.divider()

        # GC/DC
        st.markdown(f"#### 🔔 ゴールデン/デッドクロス（直近{cross_lb}日）")
        with st.spinner("クロスシグナル検出中..."):
            df_cross = jp_cross_signals(JP_TICKERS, cross_lb)
        if df_cross.empty:
            st.info("クロスシグナルは検出されませんでした")
        else:
            gc = df_cross[df_cross["シグナル"].str.contains("ゴールデン")]
            dc = df_cross[df_cross["シグナル"].str.contains("デッド")]
            cx1, cx2 = st.columns(2)
            with cx1:
                st.markdown(f"**🟡 GC — {len(gc)}件**")
                if not gc.empty:
                    st.dataframe(gc[["企業名","業種","発生日","現在値"]], use_container_width=True, hide_index=True)
            with cx2:
                st.markdown(f"**💀 DC — {len(dc)}件**")
                if not dc.empty:
                    st.dataframe(dc[["企業名","業種","発生日","現在値"]], use_container_width=True, hide_index=True)
            cross_str = df_cross.head(8)[["企業名","業種","シグナル","発生日"]].to_string(index=False)
            with st.spinner("AI分析中..."):
                comment, ai_name = ai_call(
                    f"直近のクロスシグナル発生銘柄:\n{cross_str}\n200文字以内で注目ポイントをコメント")
            st.info(f"🤖 **AI解説（{ai_name}）**\n\n{comment}")
    else:
        st.info("「▶ 価格パターン分析実行」を押してください")

# ══════════════════════════════════════════════════════════════
# Tab 5: JP モメンタム・相関分析
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec-head">💡 日本株 モメンタム・相関分析</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns([2,3])
    with col_m1:
        corr_win = st.slider("相関崩れウィンドウ（日）", 10, 30, 20, key="corr_win")
    with col_m2:
        run_mom = st.button("▶ モメンタム・相関分析実行", type="primary", key="run_mom")

    if run_mom:
        # モメンタムスコア
        st.markdown("#### 🚀 週次モメンタムスコア")
        with st.spinner("モメンタム計算中..."):
            df_mom = jp_momentum(JP_TICKERS)
        if not df_mom.empty:
            mm1, mm2 = st.columns(2)
            with mm1:
                st.markdown("**📈 上位20（買いモメンタム）**")
                st.dataframe(df_mom.head(20).style.format(
                    {"モメンタムスコア":"{:.3f}","株価騰落率(%)":"{:+.2f}","出来高変化率(%)":"{:+.2f}"}),
                    use_container_width=True, hide_index=True)
            with mm2:
                st.markdown("**📉 下位20（売りモメンタム）**")
                st.dataframe(df_mom.tail(20).sort_values("モメンタムスコア").style.format(
                    {"モメンタムスコア":"{:.3f}","株価騰落率(%)":"{:+.2f}","出来高変化率(%)":"{:+.2f}"}),
                    use_container_width=True, hide_index=True)
            top10 = df_mom.head(10)[["企業名","業種","モメンタムスコア","株価騰落率(%)"]].to_string(index=False)
            bot10 = df_mom.tail(10)[["企業名","業種","モメンタムスコア","株価騰落率(%)"]].to_string(index=False)
            with st.spinner("AI週次レポート生成中..."):
                comment, ai_name = ai_call(
                    f"日本株ストラテジストとして週次モメンタムレポートを400文字以内で:\n"
                    f"【高モメ上位10】\n{top10}\n【低モメ下位10】\n{bot10}\n"
                    "1.今週のモメンタム相場の特徴 2.注目銘柄 3.逆張りの観点")
            st.info(f"🤖 **AI週次レポート（{ai_name}）**\n\n{comment}")

        st.divider()

        # 相関崩れ
        st.markdown("#### 🔍 日経平均との相関崩れ検知")
        with st.spinner("相関分析中..."):
            df_corr = jp_correlation_divergence(JP_TICKERS, window=corr_win)
        if not df_corr.empty:
            cr1, cr2 = st.columns(2)
            with cr1:
                st.markdown("**🟡 独自上昇（相関崩れ+上昇）**")
                rising = df_corr[df_corr["直近5日株価変化(%)"]>0].head(15)
                st.dataframe(rising.style.format(
                    {"長期相関":"{:.3f}","直近相関":"{:.3f}","相関乖離度":"{:.3f}","直近5日株価変化(%)":"{:+.2f}"}),
                    use_container_width=True, hide_index=True)
            with cr2:
                st.markdown("**🔴 独自下落（相関崩れ+下落）**")
                falling = df_corr[df_corr["直近5日株価変化(%)"]<0].head(15)
                st.dataframe(falling.style.format(
                    {"長期相関":"{:.3f}","直近相関":"{:.3f}","相関乖離度":"{:.3f}","直近5日株価変化(%)":"{:+.2f}"}),
                    use_container_width=True, hide_index=True)
            top_div = df_corr.head(5)[["企業名","業種","相関乖離度","直近5日株価変化(%)"]].to_string(index=False)
            with st.spinner("AI分析中..."):
                comment, ai_name = ai_call(
                    f"日経平均との相関崩れ上位5社:\n{top_div}\n200文字以内:\n1.個別材料の種類 2.投資機会またはリスク")
            st.info(f"🤖 **AI相関分析（{ai_name}）**\n\n{comment}")
    else:
        st.info("「▶ モメンタム・相関分析実行」を押してください")

# ══════════════════════════════════════════════════════════════
# Tab 6: 個別銘柄ニュース（強化版）
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec-head">📰 個別銘柄ニュース（Yahoo Finance / Reuters / Google News）</div>',
                unsafe_allow_html=True)

    col_n1, col_n2, col_n3 = st.columns([2,2,3])
    with col_n1:
        market_sel = st.radio("市場", ["米国株", "日本株"], horizontal=True, key="mkt_sel")
    with col_n2:
        if market_sel == "米国株":
            news_ticker = st.text_input("ティッカー", value="NVDA", key="news_ticker_us")
        else:
            jp_options = {f"{v[0]}（{k}）": k for k,v in JP_TICKERS.items()}
            sel_lbl = st.selectbox("銘柄", list(jp_options.keys()), key="news_ticker_jp")
            news_ticker = jp_options[sel_lbl].replace(".T","")
    with col_n3:
        run_news_btn = st.button("▶ ニュース取得", type="primary", key="run_news_btn")
        do_ai_news   = st.checkbox("🤖 AIセンチメント分析", value=True, key="do_ai_news")

    if run_news_btn:
        tkr_news = news_ticker.strip().upper()
        company_news = ""
        if market_sel == "米国株":
            company_news = NDX100.get(tkr_news, (tkr_news,""))[0]
        else:
            company_news = JP_TICKERS.get(tkr_news+".T", (tkr_news,""))[0]

        with st.spinner(f"{tkr_news} のニュースを収集中..."):
            if market_sel == "米国株":
                items = fetch_stock_news(tkr_news, company_news, max_each=10)
            else:
                # 日本株：Yahoo Japan RSS + 株探
                def _jp_yahoo(code):
                    return fetch_rss(f"https://finance.yahoo.co.jp/rss/stocks/{code}",
                                     "Yahoo!Finance JP", 10, "")
                def _kabutan(code):
                    try:
                        r = requests.get(f"https://kabutan.jp/stock/news?code={code}",
                                         headers=_HEADERS, timeout=12)
                        if r.status_code != 200: return []
                        titles = re.findall(r'<a href="(/news/[^"]+)"[^>]*>([^<]{5,120})</a>', r.text)
                        times  = re.findall(r'<time[^>]*>([^<]+)</time>', r.text)
                        out = []
                        for i,(path,title) in enumerate(titles[:10]):
                            t = title.strip()
                            if len(t)<5 or "株探" in t: continue
                            out.append({"source":"株探","title":t,
                                        "link":f"https://kabutan.jp{path}",
                                        "date":times[i].strip() if i<len(times) else "","summary":""})
                        return out
                    except Exception: return []
                code = tkr_news.replace(".T","")
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    f1 = ex.submit(_jp_yahoo, code)
                    f2 = ex.submit(_kabutan, code)
                    items = f1.result() + f2.result()

        if not items:
            st.warning("ニュースを取得できませんでした")
        else:
            src_cnt = Counter(n["source"] for n in items)
            sc = st.columns(min(len(src_cnt),4))
            SRC_ICON = {"Yahoo Finance":"🟦","Reuters":"🟫","Google News":"🔵",
                        "Yahoo!Finance JP":"🟦","株探":"🟩"}
            for i,(src,cnt) in enumerate(src_cnt.items()):
                sc[i].metric(f"{SRC_ICON.get(src,'⚪')} {src}", f"{cnt}件")
            st.divider()
            for n in items:
                icon = SRC_ICON.get(n["source"],"⚪")
                with st.expander(f"{icon} {n['title'][:70]}{'...' if len(n['title'])>70 else ''}"):
                    col_a, col_b = st.columns([4,1])
                    with col_a:
                        st.markdown(f"**{n['title']}**")
                        if n.get("summary"): st.caption(n["summary"])
                    with col_b:
                        if n.get("date"): st.caption(f"📅 {n['date']}")
                        if n.get("link"): st.markdown(f"[🔗 開く]({n['link']})")

            if do_ai_news:
                st.divider()
                hdl = "\n".join(f"- [{n['source']}] {n['title']}" for n in items[:12])
                name_for_prompt = company_news or tkr_news
                with st.spinner("AIセンチメント分析中..."):
                    comment, ai_name = ai_call(
                        f"以下は{name_for_prompt}({tkr_news})の最新ニュースです。\n{hdl}\n\n"
                        "投資家向けに300文字以内:\n1.センチメント判定: 強気/弱気/中立\n"
                        "2.注目イベント\n3.株価への影響の可能性")
                st.markdown("#### 🤖 AIニュース分析")
                st.info(f"{comment}\n\n_AI: {ai_name}_")

# ══════════════════════════════════════════════════════════════
# Tab 7: 市場全体ニュース
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="sec-head">🗺️ 市場全体ニュース（Reuters / Google News）</div>',
                unsafe_allow_html=True)
    run_mkt_news = st.button("▶ 市場ニュースを取得", type="primary", key="run_mkt_news")

    if run_mkt_news:
        with st.spinner("市場ニュース取得中..."):
            mkt_news = fetch_market_news(10)

        if not mkt_news:
            st.info("ニュースを取得できませんでした")
        else:
            src_cnt = Counter(n["source"] for n in mkt_news)
            sc = st.columns(min(len(src_cnt),4))
            SRC_ICON2 = {"Reuters Business":"🟫","Reuters Tech":"🟤","Google News":"🔵"}
            for i,(src,cnt) in enumerate(src_cnt.items()):
                sc[i].metric(f"{SRC_ICON2.get(src,'⚪')} {src}", f"{cnt}件")
            st.divider()
            for n in mkt_news:
                icon = SRC_ICON2.get(n["source"],"⚪")
                with st.expander(f"{icon} {n['title'][:70]}{'...' if len(n['title'])>70 else ''}"):
                    col_a, col_b = st.columns([4,1])
                    with col_a:
                        st.markdown(f"**{n['title']}**")
                        if n.get("summary"): st.caption(n["summary"])
                    with col_b:
                        if n.get("date"): st.caption(f"📅 {n['date']}")
                        if n.get("link"): st.markdown(f"[🔗 開く]({n['link']})")

            if st.checkbox("🤖 市場全体AIサマリーを表示", value=True, key="mkt_ai"):
                hdl = "\n".join(f"[{n['source']}] {n['title']}" for n in mkt_news[:15])
                with st.spinner("AI要約中..."):
                    comment, ai_name = ai_call(
                        f"本日の米国市場関連ニュースです。\n{hdl}\n\n投資家向けに300文字以内:\n"
                        "1.市場全体のセンチメント 2.注目テーマ・セクター 3.今後の注意点")
                st.markdown("#### 🤖 市場全体AI要約")
                st.info(f"{comment}\n\n_AI: {ai_name}_")
