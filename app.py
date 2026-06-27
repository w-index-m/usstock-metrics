# -*- coding: utf-8 -*-
"""
NASDAQ-100 プロ仕様 銘柄分析ダッシュボード v2.5
─────────────────────────────────────────────────────
修正履歴（v2.5）
  - パフォーマンス分析のデータソース フォールバック追加
    Tiingo → Stooq → Yahoo Finance の順で自動切り替え
  - get_market_data / analyze_ticker を共通ラッパーで統合
  - 各プロバイダーで取得失敗時にトーストで通知
─────────────────────────────────────────────────────
必要 Secrets:
  TIINGO_API_KEY / GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY
  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / NOTIFY_EMAIL
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import concurrent.futures
import os, io, json, time, smtplib, re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ── フォント ──────────────────────────────────────────────────
for font in ["fonts/NotoSansJP-VariableFont_wght.ttf"]:
    if os.path.exists(font):
        fm.fontManager.addfont(font)
        prop = fm.FontProperties(fname=font)
        plt.rcParams["font.family"] = prop.get_name()
        break
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["IPAexGothic", "TakaoPGothic", "DejaVu Sans"]

# ── ページ設定 ────────────────────────────────────────────────
st.set_page_config(page_title="NASDAQ-100 プロ分析ツール", layout="wide", page_icon="🚀")

# ── Secrets ───────────────────────────────────────────────────
TIINGO_API_KEY     = st.secrets.get("TIINGO_API_KEY", "")
GEMINI_API_KEY     = st.secrets.get("GEMINI_API_KEY", "")
GROQ_API_KEY       = st.secrets.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
SMTP_HOST          = st.secrets.get("SMTP_HOST", "")
SMTP_PORT          = int(st.secrets.get("SMTP_PORT", 587))
SMTP_USER          = st.secrets.get("SMTP_USER", "")
SMTP_PASS          = st.secrets.get("SMTP_PASS", "")
NOTIFY_EMAIL       = st.secrets.get("NOTIFY_EMAIL", "")

if not TIINGO_API_KEY:
    st.warning("⚠️ TIINGO_API_KEY が未設定です。Stooq / Yahoo Finance にフォールバックします。")

if GEMINI_API_KEY and GENAI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

# ── AI モデル設定 ─────────────────────────────────────────────
GEMINI_MODELS     = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
GROQ_MODELS       = ["llama-3.3-70b-versatile", "llama3-8b-8192"]
OPENROUTER_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "microsoft/phi-4-reasoning-plus:free",
]

# ── AI フォールバック呼び出し ──────────────────────────────────

def _call_gemini(prompt: str) -> str | None:
    if not GEMINI_API_KEY or not GENAI_AVAILABLE:
        return None
    for model_name in GEMINI_MODELS:
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name)
                return f"[Gemini:{model_name}]\n{model.generate_content(prompt).text}"
            except Exception as e:
                err = str(e)
                if "429" in err:
                    wait = 15 * (2 ** attempt)
                    st.toast(f"⏳ Gemini 429 → {wait}秒待機", icon="⏳")
                    time.sleep(wait)
                    if attempt == 2:
                        break
                elif "404" in err or "not found" in err.lower():
                    break
                else:
                    return None
    return None


def _call_groq(prompt: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    for model_name in GROQ_MODELS:
        for attempt in range(3):
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={"model": model_name, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 1024, "temperature": 0.3},
                    timeout=20,
                )
                if r.status_code == 200:
                    return f"[Groq:{model_name}]\n{r.json()['choices'][0]['message']['content']}"
                elif r.status_code == 429:
                    wait = 15 * (2 ** attempt)
                    st.toast(f"⏳ Groq 429 → {wait}秒待機", icon="⏳")
                    time.sleep(wait)
                    if attempt == 2:
                        break
                else:
                    break
            except Exception:
                return None
    return None


def _call_openrouter(prompt: str) -> str | None:
    if not OPENROUTER_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nasdaq-dashboard.streamlit.app",
        "X-Title": "NASDAQ-100 Dashboard",
    }
    for model_name in OPENROUTER_MODELS:
        for attempt in range(3):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={"model": model_name, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 1024, "temperature": 0.3},
                    timeout=20,
                )
                if r.status_code == 200:
                    return f"[OpenRouter:{model_name}]\n{r.json()['choices'][0]['message']['content']}"
                elif r.status_code == 429:
                    wait = 15 * (2 ** attempt)
                    st.toast(f"⏳ OpenRouter 429 → {wait}秒待機", icon="⏳")
                    time.sleep(wait)
                    if attempt == 2:
                        break
                else:
                    break
            except Exception:
                return None
    return None


def call_ai(prompt: str) -> str:
    """Gemini → Groq → OpenRouter の順でフォールバック"""
    result = _call_gemini(prompt)
    if result:
        return result
    st.toast("🔁 Gemini → Groq へ切り替え", icon="🔁")
    result = _call_groq(prompt)
    if result:
        return result
    st.toast("🔁 Groq → OpenRouter へ切り替え", icon="🔁")
    result = _call_openrouter(prompt)
    if result:
        return result
    return "⚠️ 全 AI プロバイダーで失敗しました。APIキー・プラン残量を確認してください。"


# ── AI キャッシュ付きラッパー ──────────────────────────────────

def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text).strip()


@st.cache_data(ttl=3600, show_spinner=False)
def ai_translate_titles(titles_tuple: tuple) -> dict:
    """英語見出し一覧を一括翻訳して {英語: 日本語} を返す"""
    if not titles_tuple:
        return {}
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles_tuple))
    raw = call_ai(
        "以下の英語ニュース見出しを日本語に翻訳してください。\n"
        "必ず「番号. 翻訳文」の形式のみで返してください（説明・コメント不要）。\n\n"
        + numbered
    )
    result = {}
    for line in raw.splitlines():
        m = re.match(r'^(\d+)[.)]\s*(.+)', line.strip())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(titles_tuple):
                result[titles_tuple[idx]] = m.group(2).strip()
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def ai_translate_description(text: str) -> str:
    """RSS 記事概要を日本語に翻訳（200文字以内）"""
    if not text:
        return ""
    return call_ai(
        f"以下の英語テキストを自然な日本語に翻訳してください（200文字以内）:\n\n{text[:800]}"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def ai_translate_and_summarize(text: str, context: str = "") -> str:
    return call_ai(f"""
以下の英文テキストを日本語に翻訳し、投資家向けに300文字以内で要約してください。
{f'背景情報: {context}' if context else ''}
英文:\n{text}
出力形式:\n【翻訳要約】\n（ここに日本語で記載）
""")


@st.cache_data(ttl=3600, show_spinner=False)
def ai_sentiment(headlines_tuple: tuple, ticker: str) -> str:
    if not headlines_tuple:
        return "（データなし）"
    return call_ai(f"""
以下は米国株「{ticker}」の最新ニュース見出しです。
センチメント（強気/弱気/中立）を判定し、理由を日本語200文字以内で説明してください。
ニュース見出し:\n{chr(10).join(f"- {h}" for h in headlines_tuple)}
出力形式:\nセンチメント: [強気 / 弱気 / 中立]\n理由: （日本語で説明）
""")


@st.cache_data(ttl=3600, show_spinner=False)
def ai_earnings_analysis(ticker: str, xbrl_str: str, eps_str: str) -> str:
    return call_ai(f"""
米国株「{ticker}」の決算データを分析し、投資家向け日本語レポートを400文字以内で作成してください。
## XBRL財務データ\n{xbrl_str}
## EPS履歴\n{eps_str}
観点: 1.売上・利益トレンド 2.EPSサプライズ傾向 3.投資判断上の注目点
""")


@st.cache_data(ttl=3600, show_spinner=False)
def ai_company_summary(ticker: str) -> str:
    return call_ai(f"""
米国株「{ticker}」について、事業内容・強み・主な収益源を投資家向けに日本語300文字以内で要約してください。
""")


# ── NASDAQ-100 ティッカー ─────────────────────────────────────
nasdaq100_tickers = sorted(set([
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","AVGO","COST","PEP",
    "ADBE","NFLX","AMD","INTC","CSCO","QCOM","TXN","AMAT","HON","INTU",
    "SBUX","BKNG","ADP","REGN","VRTX","LRCX","MU","PANW","KLAC","CDNS",
    "ADI","NXPI","FTNT","WDAY","SNPS","MELI","CRWD","CTAS","VRT","CEG",
    "MRVL","ORLY","MNST","ASML","CSX","ROST","PAYX","AEP","KDP","ODFL",
    "DXCM","TEAM","FAST","BIIB","CHTR","EA","EXC","FANG","GEHC","GFS",
    "IDXX","KHC","LCID","LULU","MCHP","MDLZ","MRNA","PCAR","PDD","PYPL",
    "RIVN","SIRI","TTD","WBD","XEL","ZS","ANSS","AZN","BKR","CDW",
    "CPRT","DLTR","EBAY","ENPH","ILMN","JD","MAR","MSTR","OKTA",
    "ON","SPLK","TTWO","UAL","WBA","ZM","APP","ARM","SMCI","PLTR",
    "ANET","APH","GLW","COHR","AAOI","LITE","IQE ",
]))

EDGAR_HEADERS = {
    "User-Agent": "NasdaqDashboard/2.5 research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# =============================================================
# ── 株価データ取得 フォールバック付き ────────────────────────
# 優先順位: Tiingo → Stooq → Yahoo Finance
# =============================================================

def _prices_from_tiingo(ticker: str, start, end, api_key: str) -> pd.Series | None:
    """Tiingo から日次 adjClose を取得して日次リターンで返す"""
    if not api_key:
        return None
    try:
        r = requests.get(
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
            params={"startDate": str(start), "endDate": str(end),
                    "resampleFreq": "daily", "token": api_key},
            timeout=(3, 8),  # connect 3秒 / read 8秒（旧: 15秒フラット）
        )
        if r.status_code != 200:
            return None
        df = pd.DataFrame(r.json())
        if df.empty or "adjClose" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df["adjClose"].pct_change().dropna()
    except Exception:
        return None


def _prices_from_stooq(ticker: str, start, end) -> pd.Series | None:
    """
    Stooq CSV API から日次終値を取得してリターンで返す。
    Stooq のシンボルは '.US' サフィックスを付ける（例: AAPL.US）。
    QQQ は QQQ.US として取得可能。
    """
    try:
        stooq_symbol = f"{ticker}.US"
        url = (
            f"https://stooq.com/q/d/l/?s={stooq_symbol}"
            f"&d1={str(start).replace('-','')}&d2={str(end).replace('-','')}&i=d"
        )
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df["Close"].pct_change().dropna()
    except Exception:
        return None


def _prices_from_yahoo(ticker: str, start, end) -> pd.Series | None:
    """
    Yahoo Finance v8 API から日次 adjclose を取得してリターンで返す。
    レート制限に注意（並列呼び出し時は適宜スリープを）。
    """
    try:
        start_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
        end_ts   = int(datetime.combine(end,   datetime.min.time()).timestamp())
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1d&period1={start_ts}&period2={end_ts}"
        )
        r = requests.get(
            url, timeout=(5, 10),
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        if r.status_code != 200:
            return None
        result = r.json().get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        adjclose   = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        if not timestamps or not adjclose:
            return None
        dates = pd.to_datetime(timestamps, unit="s").normalize()
        series = pd.Series(adjclose, index=dates, dtype=float).dropna()
        return series.pct_change().dropna()
    except Exception:
        return None


def _fetch_returns_with_fallback(
    ticker: str, start, end, api_key: str, label: str = ""
) -> tuple[pd.Series | None, str]:
    """
    Tiingo → Stooq → Yahoo Finance の順でリターン系列を取得。
    成功したプロバイダー名も返す。
    label は進捗表示用（例: "QQQ"）。
    """
    tag = label or ticker

    # 1️⃣ Tiingo
    if api_key:
        result = _prices_from_tiingo(ticker, start, end, api_key)
        if result is not None and len(result) > 10:
            return result, "Tiingo"

    # 2️⃣ Stooq
    result = _prices_from_stooq(ticker, start, end)
    if result is not None and len(result) > 10:
        return result, "Stooq"

    # 3️⃣ Yahoo Finance
    result = _prices_from_yahoo(ticker, start, end)
    if result is not None and len(result) > 10:
        return result, "Yahoo Finance"

    return None, "（全プロバイダー失敗）"


# ── 市場ベンチマーク（QQQ）────────────────────────────────────

def get_market_data(api_key, start, end):
    """QQQ のリターン系列を取得。失敗時は None。"""
    result, provider = _fetch_returns_with_fallback("QQQ", start, end, api_key, label="QQQ")
    if result is None:
        st.error("❌ QQQ（市場ベンチマーク）の取得に全プロバイダーで失敗しました")
        return None
    if provider != "Tiingo":
        st.toast(f"📡 QQQ データ: {provider} を使用", icon="📡")
    return result


# ── 個別銘柄分析 ──────────────────────────────────────────────

def analyze_ticker(ticker, market_returns, api_key, start, end):
    """
    個別銘柄のリターン系列を取得し、統計指標を計算して返す。
    データ取得は Tiingo → Stooq → Yahoo Finance の順でフォールバック。
    """
    returns, provider = _fetch_returns_with_fallback(ticker, start, end, api_key, label=ticker)
    if returns is None:
        return None

    common = returns.index.intersection(market_returns.index)
    if len(common) < 60:
        return None

    x = returns.loc[common].values
    y = market_returns.loc[common].values
    annual_return = np.mean(x) * 252
    annual_risk   = np.std(x) * np.sqrt(252)
    beta     = np.cov(x, y)[0, 1] / np.var(y)
    alpha    = annual_return - (0.01 + beta * (np.mean(y) * 252 - 0.01))
    sharpe   = (annual_return - 0.01) / annual_risk
    residual = np.std(x - beta * y) * np.sqrt(252)

    return {
        "銘柄": ticker,
        "データソース": provider,          # どのプロバイダーで取得できたかを記録
        "年間リターン": annual_return,
        "年間リスク": annual_risk,
        "シャープレシオ": sharpe,
        "ベータ": beta,
        "アルファ": alpha,
        "レジデュアルリスク": residual,
    }


# ── CIK 取得 ──────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_cik(ticker: str) -> str | None:
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "NasdaqDashboard/2.5 research@example.com"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        for entry in r.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None

# ── EDGAR フィリング ───────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_edgar_filings(ticker: str, form_type: str = "10-K", count: int = 4) -> list[dict]:
    try:
        cik = get_cik(ticker)
        if not cik:
            return []
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                         headers=EDGAR_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        data       = r.json()
        filings    = data.get("filings", {}).get("recent", {})
        forms      = filings.get("form", [])
        dates      = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        results    = []
        for i, f in enumerate(forms):
            if f == form_type:
                acc = accessions[i].replace("-", "")
                results.append({"form": f, "date": dates[i], "accession": accessions[i],
                                 "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"})
                if len(results) >= count:
                    break
        return results
    except Exception:
        return []

# ── XBRL 財務数値 ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_xbrl_financials(ticker: str) -> pd.DataFrame:
    try:
        cik = get_cik(ticker)
        if not cik:
            return pd.DataFrame()
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                         headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        facts = r.json().get("facts", {})

        def extract_series(concept_path: str, label: str) -> pd.DataFrame:
            ns, concept = concept_path.split(":")
            units   = facts.get(ns, {}).get(concept, {}).get("units", {})
            entries = []
            for uk in ["USD", "USD/shares", "shares"]:
                entries = units.get(uk, [])
                if entries:
                    break
            q_entries = sorted(
                [e for e in entries if e.get("form") in ("10-Q", "10-K") and e.get("fp")],
                key=lambda x: x.get("end", ""), reverse=True,
            )
            rows, seen = [], set()
            for e in q_entries:
                key = e.get("end")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"期間末": key, label: e.get("val")})
                if len(rows) >= 5:
                    break
            return pd.DataFrame(rows).set_index("期間末") if rows else pd.DataFrame()

        rev_df = extract_series("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "売上高")
        if rev_df.empty:
            rev_df = extract_series("us-gaap:Revenues", "売上高")
        if rev_df.empty:
            rev_df = extract_series("us-gaap:SalesRevenueNet", "売上高")
        ni_df  = extract_series("us-gaap:NetIncomeLoss", "純利益")
        eps_df = extract_series("us-gaap:EarningsPerShareBasic", "EPS（基本）")

        dfs = [d for d in [rev_df, ni_df, eps_df] if not d.empty]
        if not dfs:
            return pd.DataFrame()
        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.join(d, how="outer")
        return merged.sort_index(ascending=False).head(5).reset_index().assign(銘柄=ticker)
    except Exception:
        return pd.DataFrame()

# ── Yahoo Finance 決算データ ──────────────────────────────────

@st.cache_data(ttl=1800)
def get_earnings_data(ticker: str) -> dict:
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        })
        r = session.get(
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            "?modules=calendarEvents,earningsHistory,defaultKeyStatistics",
            timeout=(3, 5),
            stream=False,
        )
        session.close()

        if r.status_code != 200:
            return {}

        data = r.json().get("quoteSummary", {}).get("result", [{}])[0]
        earnings_dates = data.get("calendarEvents", {}).get("earnings", {}).get("earningsDate", [])
        next_date      = earnings_dates[0].get("fmt", "N/A") if earnings_dates else "N/A"

        history     = data.get("earningsHistory", {}).get("history", [])
        eps_history = []
        for h in history[-4:]:
            eps_history.append({
                "期間":       h.get("quarter", {}).get("fmt", ""),
                "EPS実績":    h.get("epsActual", {}).get("raw"),
                "EPS予想":    h.get("epsEstimate", {}).get("raw"),
                "サプライズ%": h.get("surprisePercent", {}).get("raw"),
            })

        stats = data.get("defaultKeyStatistics", {})
        return {
            "次回決算日": next_date,
            "EPS履歴":    eps_history,
            "PER(予想)":  stats.get("forwardPE", {}).get("raw"),
            "PBR":        stats.get("priceToBook", {}).get("raw"),
        }
    except Exception:
        return {}

# ── Yahoo Finance ニュース ────────────────────────────────────

@st.cache_data(ttl=900)
def get_news_headlines(ticker: str, max_items: int = 10) -> list[str]:
    try:
        r = requests.get(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            timeout=(3, 5),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return []
        root   = ET.fromstring(r.content)
        titles = []
        for item in root.findall(".//item"):
            el = item.find("title")
            if el is not None and el.text:
                titles.append(el.text.strip())
            if len(titles) >= max_items:
                break
        return titles
    except Exception:
        return []

# ── 市場ニュース（複数RSS並列取得）──────────────────────────────

# (name, url, category)  category: "market" | "tech"
_NEWS_SOURCES = [
    ("Yahoo Finance",   "https://finance.yahoo.com/rss/topstories",                        "market"),
    ("MarketWatch",     "https://feeds.marketwatch.com/marketwatch/topstories/",           "market"),
    ("MarketWatch MP",  "https://feeds.marketwatch.com/marketwatch/marketpulse/",          "market"),
    ("Investing.com",   "https://www.investing.com/rss/news_25.rss",                       "market"),
    ("Reuters Biz",     "https://feeds.reuters.com/reuters/businessNews",                  "market"),
    ("TechCrunch",      "https://techcrunch.com/feed/",                                    "tech"),
    ("The Verge",       "https://www.theverge.com/rss/index.xml",                          "tech"),
    ("CNBC Tech",       "https://www.cnbc.com/id/19854910/device/rss/rss.html",            "tech"),
    ("Reuters Tech",    "https://feeds.reuters.com/reuters/technologyNews",                "tech"),
]

@st.cache_data(ttl=600, show_spinner=False)
def get_market_news(max_per_source: int = 12) -> list[dict]:
    """全ソースを並列取得。各アイテムに category('market'|'tech') を付与"""

    def _fetch(name: str, url: str, category: str) -> list[dict]:
        try:
            r = requests.get(
                url, timeout=(4, 8),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            if r.status_code != 200:
                return []
            root  = ET.fromstring(r.content)
            items = []
            for item in root.findall(".//item"):
                title_el = item.find("title")
                link_el  = item.find("link")
                pub_el   = item.find("pubDate")
                if not (title_el is not None and title_el.text):
                    continue
                desc_el  = item.find("description")
                desc_raw = (desc_el.text or "") if desc_el is not None else ""
                items.append({
                    "カテゴリ": category,
                    "ソース":   name,
                    "見出し":   title_el.text.strip(),
                    "リンク":   (link_el.text or "").strip() if link_el is not None else "",
                    "日時":     (pub_el.text or "")[:22] if pub_el is not None else "",
                    "概要":     _strip_html(desc_raw)[:500],
                })
                if len(items) >= max_per_source:
                    break
            return items
        except Exception:
            return []

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_NEWS_SOURCES)) as ex:
        futures = {ex.submit(_fetch, name, url, cat): name for name, url, cat in _NEWS_SOURCES}
        for f in concurrent.futures.as_completed(futures):
            results.extend(f.result())
    return results


# ── リアルタイム株価クォート ──────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_quote_data(tickers: tuple) -> pd.DataFrame:
    """Yahoo Finance v7 から現在値・前日比を取得（TTL 60秒）"""
    try:
        symbols = ",".join(tickers)
        url = (
            f"https://query2.finance.yahoo.com/v7/finance/quote"
            f"?symbols={symbols}"
            f"&fields=regularMarketPrice,regularMarketChange,regularMarketChangePercent,shortName"
        )
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(5, 10))
        if r.status_code != 200:
            return pd.DataFrame()
        results = r.json().get("quoteResponse", {}).get("result", [])
        rows = []
        for q in results:
            rows.append({
                "シンボル": q.get("symbol", ""),
                "名称": (q.get("shortName") or "")[:25],
                "現在値": q.get("regularMarketPrice"),
                "前日比": q.get("regularMarketChange"),
                "前日比%": q.get("regularMarketChangePercent"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_ohlcv_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """Yahoo Finance v8 から OHLCV データを取得"""
    try:
        end_ts   = int(datetime.today().timestamp())
        start_ts = int((datetime.today() - timedelta(days=days)).timestamp())
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1d&period1={start_ts}&period2={end_ts}"
        )
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(5, 10))
        if r.status_code != 200:
            return pd.DataFrame()
        result = r.json().get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()
        timestamps = result[0].get("timestamp", [])
        quotes     = result[0].get("indicators", {}).get("quote", [{}])[0]
        dates      = pd.to_datetime(timestamps, unit="s").normalize()
        df = pd.DataFrame({
            "Open":   quotes.get("open",   []),
            "High":   quotes.get("high",   []),
            "Low":    quotes.get("low",    []),
            "Close":  quotes.get("close",  []),
            "Volume": quotes.get("volume", []),
        }, index=dates)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


# ── テクニカル指標 ────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def build_tech_chart(df: pd.DataFrame, ticker: str):
    """Plotly でテクニカル分析チャート（ローソク足+MA / 出来高 / RSI / MACD）"""
    close  = df["Close"]
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi    = calc_rsi(close)
    macd_l, sig_l, hist = calc_macd(close)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.175, 0.175],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker}  株価 + 移動平均", "出来高", "RSI (14)", "MACD (12,26,9)",
        ],
    )

    # Row 1: ローソク足 + SMA
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=close,
        name="OHLC",
        increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
        showlegend=False,
    ), row=1, col=1)
    for sma, color, name in [
        (sma20,  "#FFA726", "SMA 20"),
        (sma50,  "#42A5F5", "SMA 50"),
        (sma200, "#EC407A", "SMA 200"),
    ]:
        if sma.notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=df.index, y=sma, name=name,
                line=dict(color=color, width=1.3),
            ), row=1, col=1)

    # Row 2: 出来高
    vol_colors = [
        "#26A69A" if c >= o else "#EF5350"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="出来高",
        marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    # Row 3: RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI",
        line=dict(color="#AB47BC", width=1.5),
    ), row=3, col=1)
    for level, color in [
        (70, "rgba(239,83,80,0.5)"),
        (50, "rgba(255,255,255,0.2)"),
        (30, "rgba(38,166,154,0.5)"),
    ]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=3, col=1)

    # Row 4: MACD
    hist_colors = ["#26A69A" if v >= 0 else "#EF5350" for v in hist.fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, name="ヒストグラム",
        marker_color=hist_colors, showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_l, name="MACD",
        line=dict(color="#42A5F5", width=1.3),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=sig_l, name="シグナル",
        line=dict(color="#FFA726", width=1.3),
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)", row=4, col=1)

    fig.update_layout(
        height=820, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    return fig


# ── メール通知 ────────────────────────────────────────────────

def send_email_notification(subject, body, attachment_bytes=None, filename="report.txt"):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, NOTIFY_EMAIL]):
        return False, "SMTP設定が不完全です"
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER; msg["To"] = NOTIFY_EMAIL; msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        if attachment_bytes:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            msg.attach(part)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(); server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, NOTIFY_EMAIL, msg.as_string())
        return True, "メール送信成功"
    except Exception as e:
        return False, f"メール送信失敗: {e}"

# ── テキストレポート ──────────────────────────────────────────

def build_text_report(ticker, earnings_info, xbrl_df, headlines, sentiment, ai_analysis):
    lines = [
        "=" * 60,
        f"  {ticker} 決算レポート（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "=" * 60,
        f"\n【次回決算日】 {earnings_info.get('次回決算日', 'N/A')}",
        f"【PER(予想)】  {earnings_info.get('PER(予想)', 'N/A')}",
        f"【PBR】       {earnings_info.get('PBR', 'N/A')}",
        "\n【EPS履歴・サプライズ】",
    ]
    for e in earnings_info.get("EPS履歴", []):
        lines.append(f"  {e['期間']} | 実績:{e['EPS実績']} | 予想:{e['EPS予想']} | サプライズ:{e.get('サプライズ%','N/A')}%")
    if not xbrl_df.empty:
        lines += ["\n【XBRL財務データ】", xbrl_df.to_string(index=False)]
    lines += ["\n【最新ニュース】"] + [f"  - {h}" for h in headlines]
    lines += [f"\n【センチメント】\n{sentiment}", f"\n【AI分析】\n{ai_analysis}", "=" * 60]
    return "\n".join(lines)

# ── グラフ ────────────────────────────────────────────────────

def plot_xbrl_quarterly(xbrl_df, ticker):
    if xbrl_df.empty:
        return None
    cols = [c for c in ["売上高", "純利益"] if c in xbrl_df.columns]
    if not cols:
        return None
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        vals = xbrl_df[col].dropna()
        if vals.empty:
            continue
        periods = xbrl_df.loc[vals.index, "期間末"].tolist() if "期間末" in xbrl_df.columns else list(range(len(vals)))
        ax.bar(range(len(vals)), vals.values / 1e9, color="#4472C4")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(periods, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{ticker} {col}（十億USD）")
        ax.set_ylabel("十億USD")
    plt.tight_layout()
    return fig


def plot_eps_surprise(eps_history, ticker):
    if not eps_history:
        return None
    df = pd.DataFrame(eps_history).dropna(subset=["EPS実績", "EPS予想"])
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(df))
    ax.bar(x, df["EPS実績"], label="EPS実績", color="#4472C4", alpha=0.9)
    ax.bar(x, df["EPS予想"], label="EPS予想", color="#ED7D31", alpha=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["期間"].tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{ticker} EPS 実績 vs 予想")
    ax.legend()
    plt.tight_layout()
    return fig

# =============================================================
# モジュールレベル定数
# =============================================================
MARKET_INDICES = [
    ("^GSPC", "S&P 500",      "ES=F"),
    ("^NDX",  "NASDAQ 100",   "NQ=F"),
    ("^DJI",  "ダウ平均",     "YM=F"),
    ("^RUT",  "Russell 2000", "RTY=F"),
]
MARKET_MACRO = [
    ("^VIX",     "VIX（恐怖指数）"),
    ("^TNX",     "米10年債利回り(%)"),
    ("GC=F",     "金 ($/oz)"),
    ("CL=F",     "WTI原油 ($/bbl)"),
    ("DX-Y.NYB", "ドル指数 (DXY)"),
]
DOW30_TICKERS = sorted([
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
])

# =============================================================
# センチメントスコア計算
# =============================================================

@st.cache_data(ttl=300, show_spinner=False)
def calc_market_sentiment(index_ticker: str, futures_ticker: str) -> dict:
    """テクニカル + 先物/CFD ファクターから市場センチメントスコアを算出
    スコア範囲: ±12（SMA×3 ±1, RSI ±2, MACD ±1, 5d ±2, VIX ±2, CFD ±2）
    """
    ohlcv = get_ohlcv_data(index_ticker, 400)
    if ohlcv.empty or len(ohlcv) < 50:
        return {}

    close   = ohlcv["Close"]
    current = close.iloc[-1]
    factors = []

    # SMA 比較
    for window, fname in [(20, "SMA20"), (50, "SMA50"), (200, "SMA200")]:
        if len(close) < window:
            continue
        sma_val = close.rolling(window).mean().iloc[-1]
        sc  = 1 if current > sma_val else -1
        pct = (current / sma_val - 1) * 100
        factors.append({
            "ファクター": f"終値 vs {fname}",
            "スコア":    sc,
            "詳細":      f"{current:,.1f}  {'▲' if sc>0 else '▼'}  {fname} {sma_val:,.1f}  ({pct:+.1f}%)",
        })

    # RSI
    rsi_val = calc_rsi(close).iloc[-1]
    sc_rsi  = 2 if rsi_val < 22 else 1 if rsi_val < 30 else \
              -2 if rsi_val > 78 else -1 if rsi_val > 70 else 0
    factors.append({
        "ファクター": "RSI(14)",
        "スコア":    sc_rsi,
        "詳細":      f"RSI = {rsi_val:.1f}  ({'買われすぎ' if rsi_val>70 else '売られすぎ' if rsi_val<30 else '中立圏'})",
    })

    # MACD
    macd_l, sig_l, hist = calc_macd(close)
    sc_macd    = 1 if hist.iloc[-1] > 0 else -1
    cross_note = " ← ゴールデンクロス🟢" if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0 else \
                 " ← デッドクロス🔴"     if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0 else ""
    factors.append({
        "ファクター": "MACD(12,26,9)",
        "スコア":    sc_macd,
        "詳細":      f"MACD {macd_l.iloc[-1]:.2f} / Signal {sig_l.iloc[-1]:.2f}{cross_note}",
    })

    # 5日モメンタム
    if len(close) >= 6:
        ret5  = (current / close.iloc[-6] - 1) * 100
        sc_5d = 2 if ret5 > 2 else 1 if ret5 > 0.5 else \
                -2 if ret5 < -2 else -1 if ret5 < -0.5 else 0
        factors.append({
            "ファクター": "5日モメンタム",
            "スコア":    sc_5d,
            "詳細":      f"5営業日リターン: {ret5:+.2f}%",
        })

    # VIX
    vix_df = get_quote_data(("^VIX",))
    if not vix_df.empty and pd.notna(vix_df.iloc[0]["現在値"]):
        vix_val   = vix_df.iloc[0]["現在値"]
        sc_vix    = 2 if vix_val < 14 else 1 if vix_val < 18 else \
                    -2 if vix_val > 30 else -1 if vix_val > 23 else 0
        vix_label = "極度の楽観" if vix_val < 14 else "低ボラ" if vix_val < 18 else \
                    "警戒域"     if vix_val < 23 else "高ボラ" if vix_val < 30 else "恐怖"
        factors.append({
            "ファクター": "VIX（恐怖指数）",
            "スコア":    sc_vix,
            "詳細":      f"VIX = {vix_val:.1f}  ({vix_label})",
        })

    # 先物/CFD（24h取引 — 土日・時間外も反映）
    fut_df = get_quote_data((futures_ticker,))
    if not fut_df.empty and pd.notna(fut_df.iloc[0]["現在値"]):
        fut_price = fut_df.iloc[0]["現在値"]
        fut_chgp  = fut_df.iloc[0]["前日比%"]
        if pd.notna(fut_chgp):
            sc_cfd = 2 if fut_chgp > 1.5 else 1 if fut_chgp > 0.3 else \
                     -2 if fut_chgp < -1.5 else -1 if fut_chgp < -0.3 else 0
            factors.append({
                "ファクター": f"先物/CFD ({futures_ticker})",
                "スコア":    sc_cfd,
                "詳細":      f"{fut_price:,.1f}  前日比: {fut_chgp:+.2f}%  ※土日・時間外も反映",
            })

    total = sum(f["スコア"] for f in factors)
    if   total >= 7:  verdict = "🟢 強気"
    elif total >= 3:  verdict = "🟡 やや強気"
    elif total >= -2: verdict = "⚪ 中立"
    elif total >= -6: verdict = "🟠 やや弱気"
    else:             verdict = "🔴 弱気"

    return {"total": total, "verdict": verdict, "factors": factors}


# =============================================================
# モメンタムランキング用
# =============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Wikipedia から S&P500 構成銘柄を取得"""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []


def _fetch_closes_30d(ticker: str) -> pd.Series | None:
    """Yahoo Finance から30日分の終値を高速取得（モメンタム専用）"""
    try:
        end_ts   = int(datetime.today().timestamp())
        start_ts = int((datetime.today() - timedelta(days=40)).timestamp())
        r = requests.get(
            f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1d&period1={start_ts}&period2={end_ts}",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=(3, 5),
        )
        if r.status_code != 200:
            return None
        result = r.json().get("chart", {}).get("result", [])
        if not result:
            return None
        closes = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        dates  = pd.to_datetime(result[0].get("timestamp", []), unit="s").normalize()
        return pd.Series(closes, index=dates, dtype=float).dropna()
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def calc_momentum_scores(tickers: tuple, workers: int = 15) -> pd.DataFrame:
    """当日/5日/20日リターンの加重スコア（0.5/0.3/0.2）でランキング"""

    def _score(ticker: str) -> dict | None:
        c = _fetch_closes_30d(ticker)
        if c is None or len(c) < 6:
            return None
        r1  = (c.iloc[-1] / c.iloc[-2]  - 1) * 100 if len(c) >= 2  else 0.0
        r5  = (c.iloc[-1] / c.iloc[-6]  - 1) * 100 if len(c) >= 6  else 0.0
        r20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) >= 21 else 0.0
        return {"銘柄": ticker, "当日(%)": round(r1,2), "5日(%)": round(r5,2),
                "20日(%)": round(r20,2), "スコア": round(r1*0.5 + r5*0.3 + r20*0.2, 3)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in concurrent.futures.as_completed({ex.submit(_score, t): t for t in tickers}):
            v = r.result()
            if v:
                results.append(v)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_values("スコア", ascending=False).reset_index(drop=True)
    df.insert(0, "順位", range(1, len(df) + 1))
    return df


# =============================================================
# マーケットスクリーン用スパークラインキャッシュ
# =============================================================

@st.cache_data(ttl=120, show_spinner=False)
def get_sparkline_data(tickers: tuple) -> dict:
    """30日分終値を並列取得して dict[ticker->Series] で返す"""
    result: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(_fetch_closes_30d, t): t for t in tickers}
        for f in concurrent.futures.as_completed(futures):
            t = futures[f]; s = f.result()
            if s is not None and len(s) > 1:
                result[t] = s.tail(30)
    return result


@st.cache_data(ttl=60, show_spinner=False)
def get_intraday_data(tickers: tuple) -> dict:
    """当日5分足の価格系列と前日終値を並列取得して dict[ticker->{"prices":Series,"prev_close":float}] で返す"""
    def _fetch(ticker: str):
        try:
            url = (
                f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
                f"?interval=5m&range=1d"
            )
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(5, 10))
            if r.status_code != 200:
                return ticker, None
            data = r.json().get("chart", {}).get("result", [])
            if not data:
                return ticker, None
            meta       = data[0].get("meta", {})
            prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")
            timestamps = data[0].get("timestamp", [])
            closes     = data[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            if not timestamps or not closes or not prev_close:
                return ticker, None
            times  = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York")
            prices = pd.Series(closes, index=times, dtype=float).dropna()
            return ticker, {"prices": prices, "prev_close": float(prev_close)}
        except Exception:
            return ticker, None

    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(_fetch, t): t for t in tickers}
        for f in concurrent.futures.as_completed(futures):
            t, data = f.result()
            if data is not None and len(data["prices"]) > 0:
                result[t] = data
    return result


# =============================================================
# セクター比較
# =============================================================
OPTICAL_TICKERS = ["CIEN", "COHR", "LITE", "VIAV", "AAOI"]
SEMI_TICKERS    = ["NVDA", "AMD", "AVGO", "INTC", "QCOM"]
SEMI_ETF        = ["SMH", "SOXX"]

@st.cache_data(ttl=1800, show_spinner=False)
def get_normalized_prices(tickers: tuple, days: int = 365) -> pd.DataFrame:
    """複数ティッカーの終値を取得し、始値=100に正規化したDataFrameを返す"""
    def _fetch(t: str):
        ohlcv = get_ohlcv_data(t, days + 10)
        return t, ohlcv["Close"] if not ohlcv.empty else None

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for t, s in [f.result() for f in concurrent.futures.as_completed(
            {ex.submit(_fetch, t): t for t in tickers}
        )]:
            if s is not None:
                results[t] = s

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_index().ffill()
    first_valid = df.dropna(how="any").index
    if first_valid.empty:
        return pd.DataFrame()
    df = df.loc[first_valid[0]:]
    return (df / df.iloc[0] * 100).round(2)


# =============================================================
# センチメント履歴
# =============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def calc_sentiment_history(index_ticker: str = "^NDX", days: int = 365) -> pd.DataFrame:
    """RSI / MACD / VIX / モメンタム / SMA50 から日次センチメントスコアを計算"""
    ohlcv     = get_ohlcv_data(index_ticker, days + 60)
    vix_ohlcv = get_ohlcv_data("^VIX", days + 60)
    if ohlcv.empty:
        return pd.DataFrame()

    close = ohlcv["Close"]

    # RSI スコア（±2）
    rsi    = calc_rsi(close)
    rsi_sc = pd.Series(0.0, index=rsi.index)
    rsi_sc[rsi > 78] = -2; rsi_sc[(rsi > 70) & (rsi <= 78)] = -1
    rsi_sc[rsi < 22]  =  2; rsi_sc[(rsi < 30) & (rsi >= 22)] = 1

    # MACD スコア（±1）
    _, _, hist = calc_macd(close)
    macd_sc = hist.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

    # 5日モメンタム スコア（±2）
    ret5   = close.pct_change(5) * 100
    mom_sc = pd.Series(0.0, index=ret5.index)
    mom_sc[ret5 > 2]   =  2; mom_sc[(ret5 > 0.5)  & (ret5 <= 2)]    = 1
    mom_sc[ret5 < -2]  = -2; mom_sc[(ret5 < -0.5) & (ret5 >= -2)]   = -1

    # SMA50 スコア（±1）
    sma50_sc = (close > close.rolling(50).mean()).map({True: 1.0, False: -1.0})

    # VIX スコア（±2）
    vix_sc = pd.Series(0.0, index=close.index)
    if not vix_ohlcv.empty:
        vix = vix_ohlcv["Close"].reindex(close.index, method="ffill")
        vix_sc[vix < 14]  =  2; vix_sc[(vix >= 14) & (vix < 18)]  = 1
        vix_sc[vix >= 30] = -2; vix_sc[(vix >= 23) & (vix < 30)]  = -1

    total = (rsi_sc + macd_sc + mom_sc + sma50_sc + vix_sc).rolling(5).mean()

    return pd.DataFrame({
        "指数価格":            close,
        "センチメントスコア":  total,
        "RSI貢献":             rsi_sc.rolling(5).mean(),
        "MACD貢献":            macd_sc.rolling(5).mean(),
        "モメンタム貢献":      mom_sc.rolling(5).mean(),
        "VIX貢献":             vix_sc.rolling(5).mean(),
        "SMA50貢献":           sma50_sc.rolling(5).mean(),
    }).dropna().tail(days)


# =============================================================
# Page functions
# =============================================================

def page_market():
    st.title("📈 マーケット概況")

    ref_col, _ = st.columns([1, 5])
    if ref_col.button("🔄 データ更新"):
        st.cache_data.clear()
        st.rerun()

    # ── センチメントスコア ────────────────────────────────────
    st.markdown("#### 🎯 市場センチメント（テクニカル + 先物/CFD）")
    with st.spinner("センチメント計算中..."):
        sp_sent = calc_market_sentiment("^GSPC", "ES=F")
        nq_sent = calc_market_sentiment("^NDX",  "NQ=F")

    sc1, sc2 = st.columns(2)
    for col, name, sent in [(sc1, "S&P 500", sp_sent), (sc2, "NASDAQ 100", nq_sent)]:
        with col:
            if sent:
                st.metric(f"{name} センチメント", sent["verdict"],
                          f"合計スコア: {sent['total']:+d}  /  最大 ±12",
                          delta_color="off")
                with st.expander("📊 ファクター別スコア詳細"):
                    fdf = pd.DataFrame(sent["factors"])
                    def _sc_style(v):
                        if isinstance(v, (int, float)):
                            return "color:#26A69A;font-weight:bold" if v > 0 else \
                                   "color:#EF5350;font-weight:bold" if v < 0 else ""
                        return ""
                    st.dataframe(fdf.style.applymap(_sc_style, subset=["スコア"]),
                                 use_container_width=True, hide_index=True)
            else:
                st.warning(f"{name}: データ取得失敗")

    st.divider()

    # ── 主要指数 + 先物 CFD ───────────────────────────────────
    st.markdown("#### 主要指数 + 先物/CFD（土日・時間外も反映）")
    all_idx = tuple(sym for sym,_,_ in MARKET_INDICES) + \
              tuple(fut for _,_,fut in MARKET_INDICES)
    with st.spinner("指数・先物データ取得中..."):
        idx_df = get_quote_data(all_idx)

    if not idx_df.empty:
        for sym, label, fut in MARKET_INDICES:
            ri = idx_df[idx_df["シンボル"] == sym]
            rf = idx_df[idx_df["シンボル"] == fut]
            ci, cf = st.columns(2)
            with ci:
                if not ri.empty and pd.notna(ri.iloc[0]["現在値"]):
                    p = ri.iloc[0]["現在値"]; cp = ri.iloc[0]["前日比%"]
                    st.metric(f"📊 {label}", f"{p:,.2f}", f"{cp:+.2f}%" if pd.notna(cp) else None)
            with cf:
                if not rf.empty and pd.notna(rf.iloc[0]["現在値"]):
                    fp = rf.iloc[0]["現在値"]; fcp = rf.iloc[0]["前日比%"]
                    st.metric(f"📉 先物 {fut}", f"{fp:,.2f}", f"{fcp:+.2f}%" if pd.notna(fcp) else None)

    st.divider()

    # ── マクロ経済指標 ────────────────────────────────────────
    st.markdown("#### マクロ経済指標")
    with st.spinner("マクロデータ取得中..."):
        mac_df = get_quote_data(tuple(s for s,_ in MARKET_MACRO))

    if not mac_df.empty:
        mac_cols = st.columns(5)
        for col, (sym, label) in zip(mac_cols, MARKET_MACRO):
            row = mac_df[mac_df["シンボル"] == sym]
            if not row.empty and pd.notna(row.iloc[0]["現在値"]):
                p = row.iloc[0]["現在値"]; cp = row.iloc[0]["前日比%"]
                col.metric(label, f"{p:.2f}", f"{cp:+.2f}%" if pd.notna(cp) else None,
                           delta_color="inverse" if sym == "^VIX" else "normal")

    st.divider()

    # ── 個別銘柄リアルタイム ──────────────────────────────────
    st.markdown("#### 個別銘柄 リアルタイム株価")
    rt_tickers = st.multiselect(
        "銘柄を選択（最大20銘柄）", nasdaq100_tickers,
        default=["AAPL","MSFT","NVDA","META","GOOGL","AMZN"],
        max_selections=20,
    )
    if rt_tickers:
        with st.spinner("株価データ取得中..."):
            rt_df = get_quote_data(tuple(rt_tickers))
        if not rt_df.empty:
            disp = rt_df.copy()
            disp["現在値"] = disp["現在値"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
            disp["前日比"]  = disp["前日比"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
            disp["前日比%"] = disp["前日比%"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
            st.dataframe(disp, use_container_width=True, hide_index=True)

    st.divider()

    # ── ニュース ──────────────────────────────────────────────
    st.markdown("#### 📰 最新ニュース")
    with st.spinner("ニュース取得中（全ソース並列）..."):
        mkt_news = get_market_news()

    def _news_tabs(items: list[dict], key_prefix: str) -> None:
        if not items:
            st.info("取得できませんでした。")
            return
        with st.spinner("タイトル翻訳中..."):
            title_map = ai_translate_titles(tuple(i["見出し"] for i in items))
        sources = list(dict.fromkeys(i["ソース"] for i in items))
        for ntab, src in zip(st.tabs(sources), sources):
            with ntab:
                for idx, item in enumerate(i for i in items if i["ソース"] == src):
                    orig = item["見出し"]; ja = title_map.get(orig, orig)
                    ds   = f"  `{item['日時']}`" if item["日時"] else ""
                    if item["リンク"]:
                        st.markdown(f"**[{ja}]({item['リンク']})**{ds}")
                    else:
                        st.markdown(f"**{ja}**{ds}")
                    st.caption(f"🔤 {orig}")
                    if item.get("概要"):
                        with st.expander("📄 記事概要を翻訳"):
                            st.caption(item["概要"])
                            if st.button("🤖 日本語に翻訳", key=f"desc_{key_prefix}_{src}_{idx}"):
                                with st.spinner("翻訳中..."):
                                    st.info(ai_translate_description(item["概要"]))
                    st.markdown("---")

    nm, nt = st.columns(2)
    with nm:
        st.markdown("**📊 市場全般**")
        _news_tabs([i for i in mkt_news if i["カテゴリ"] == "market"], "m")
    with nt:
        st.markdown("**💻 ハイテク系**")
        _news_tabs([i for i in mkt_news if i["カテゴリ"] == "tech"],   "t")


def page_performance():
    st.title("📊 パフォーマンス分析")
    st.caption("📡 株価データ: Tiingo → Stooq → Yahoo Finance の順で自動フォールバック")

    with st.sidebar:
        st.header("設定")
        years = st.slider("分析対象期間（年）", 1, 10, 3)
        st.info("💡 データ順\n1️⃣ Tiingo → 2️⃣ Stooq → 3️⃣ Yahoo\n\nAI順\n1️⃣ Gemini → 2️⃣ Groq → 3️⃣ OpenRouter")

    end_date   = datetime.today().date()
    start_date = end_date - timedelta(days=years * 365)

    if not st.button("▶ 全銘柄パフォーマンス分析を実行", type="primary"):
        return

    with st.spinner("ベンチマーク（QQQ）データ取得中..."):
        market_returns = get_market_data(TIINGO_API_KEY, start_date, end_date)
    if market_returns is None:
        st.error("市場データ（QQQ）の取得に全プロバイダーで失敗しました")
        return

    progress    = st.progress(0, text="銘柄データ取得中...")
    results     = []
    max_workers = 20 if TIINGO_API_KEY else 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = {executor.submit(analyze_ticker, t, market_returns, TIINGO_API_KEY, start_date, end_date): t
                for t in nasdaq100_tickers}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            r = f.result()
            if r: results.append(r)
            progress.progress((i+1)/len(futs), text=f"取得中... {i+1}/{len(futs)}")
    progress.empty()

    df = pd.DataFrame(results)
    if df.empty:
        st.error("有効な分析結果がありません"); return

    df.insert(0, "シャープレシオ順位", df["シャープレシオ"].rank(ascending=False, method="min").astype(int))
    df.insert(0, "値上がり順位",       df["年間リターン"].rank(ascending=False, method="min").astype(int))
    df = df.sort_values("値上がり順位").reset_index(drop=True)
    if "データソース" in df.columns:
        st.info("📡 " + " | ".join(f"{k}: {v}銘柄" for k,v in df["データソース"].value_counts().items()))
    df["Yahoo Finance"] = df["銘柄"].apply(lambda t: f"https://finance.yahoo.com/quote/{t}/")
    base = ["値上がり順位","シャープレシオ順位","銘柄","Yahoo Finance",
            "年間リターン","年間リスク","シャープレシオ","ベータ","アルファ","レジデュアルリスク"]
    dcols = [c for c in base if c in df.columns]
    if "データソース" in df.columns: dcols.append("データソース")
    st.dataframe(
        df[dcols].style
        .format("{:.2%}", subset=["年間リターン","年間リスク","アルファ","レジデュアルリスク"])
        .format("{:.2f}", subset=["シャープレシオ","ベータ"])
        .format("{:d}",   subset=["値上がり順位","シャープレシオ順位"]),
        use_container_width=True,
        column_config={"Yahoo Finance": st.column_config.LinkColumn("Yahoo Finance", display_text="📈 表示")},
    )
    top = df.sort_values("シャープレシオ", ascending=False).head(20)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    ax1.bar(top["銘柄"], top["年間リターン"]*100, label="リターン(%)", color="#4472C4")
    ax1.bar(top["銘柄"], top["年間リスク"]  *100, label="リスク(%)",   color="#ED7D31", alpha=0.3)
    ax1.legend(); ax1.set_title("年間リターン / リスク（上位20銘柄）")
    ax2.bar(top["銘柄"], top["シャープレシオ"], color="#70AD47")
    ax2.set_title("シャープレシオ（上位20銘柄）")
    plt.tight_layout(); st.pyplot(fig)
    st.subheader("🤖 AIによる企業解説（上位3社）")
    for t in top["銘柄"].head(3):
        with st.expander(f"💡 {t} の概要"):
            st.write(ai_company_summary(t))
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Analysis")
    st.download_button("📥 Excelでダウンロード", data=output.getvalue(),
                       file_name=f"Nasdaq100_Analysis_{end_date}.xlsx")


def page_tech():
    st.title("📉 テクニカル分析チャート")
    tc1, tc2 = st.columns([3, 1])
    with tc1:
        tech_ticker = st.selectbox("銘柄を選択", nasdaq100_tickers,
                                    index=nasdaq100_tickers.index("AAPL") if "AAPL" in nasdaq100_tickers else 0)
    with tc2:
        period_label = st.selectbox("期間", ["3ヶ月","6ヶ月","1年","2年","3年"], index=2)
    period_map = {"3ヶ月":90,"6ヶ月":180,"1年":365,"2年":730,"3年":1095}

    if not st.button("▶ チャートを表示", type="primary"):
        return
    with st.spinner(f"📡 {tech_ticker} OHLCV データ取得中..."):
        ohlcv = get_ohlcv_data(tech_ticker, period_map[period_label])
    if ohlcv.empty:
        st.error("データの取得に失敗しました。"); return

    st.plotly_chart(build_tech_chart(ohlcv, tech_ticker), use_container_width=True)
    close = ohlcv["Close"]; rsi_now = calc_rsi(close).iloc[-1]
    macd_l, _, hist = calc_macd(close)
    s1, s2, s3 = st.columns(3)
    s1.metric("RSI (14)", f"{rsi_now:.1f}",
              "🔴 買われすぎ" if rsi_now>70 else "🟢 売られすぎ" if rsi_now<30 else "⚪ 中立",
              delta_color="off")
    ms = ("🟢 ゴールデンクロス" if hist.iloc[-1]>0 and hist.iloc[-2]<=0 else
          "🔴 デッドクロス"     if hist.iloc[-1]<0 and hist.iloc[-2]>=0 else
          "↑ 強気" if hist.iloc[-1]>0 else "↓ 弱気")
    s2.metric("MACD", f"{macd_l.iloc[-1]:.3f}", ms, delta_color="off")
    sma50 = close.rolling(50).mean().iloc[-1]
    s3.metric("現在値 vs SMA50", f"${close.iloc[-1]:.2f}",
              "↑ SMA50 上回り" if close.iloc[-1]>sma50 else "↓ SMA50 下回り",
              delta_color="normal" if close.iloc[-1]>sma50 else "inverse")


def page_earnings():
    st.title("📋 決算分析（SEC EDGAR + Yahoo Finance）")
    with st.sidebar:
        st.header("決算分析設定")
        selected_tickers = st.multiselect("分析する銘柄を選択", nasdaq100_tickers,
                                           default=["AAPL","MSFT","NVDA"], max_selections=10)
        filing_type  = st.selectbox("SEC フィリング種別", ["10-K","10-Q"])
        st.divider()
        enable_email = st.checkbox("決算レポートをメールで通知", value=False)

    end_date = datetime.today().date()
    if not selected_tickers:
        st.info("サイドバーで銘柄を選択してください"); return
    if not st.button("▶ 決算データを取得・分析", type="primary"):
        return

    all_reports: dict[str, str] = {}
    for ticker in selected_tickers:
        st.markdown(f"---\n### 🔍 {ticker}")
        with st.spinner(f"📡 {ticker} XBRL 取得中..."):      xbrl_df = get_xbrl_financials(ticker)
        with st.spinner(f"📡 {ticker} 決算・EPS 取得中..."):  earnings_info = get_earnings_data(ticker)
        with st.spinner(f"📡 {ticker} ニュース取得中..."):    headlines = get_news_headlines(ticker, 5)
        cols = st.columns([1,1])
        with cols[0]:
            st.markdown("**📈 XBRL 財務データ（SEC EDGAR）**")
            if not xbrl_df.empty:
                ddf = xbrl_df.copy()
                for c in ["売上高","純利益"]:
                    if c in ddf.columns:
                        ddf[c] = ddf[c].apply(lambda v: f"${v/1e9:.2f}B" if pd.notna(v) and v!=0 else "N/A")
                if "EPS（基本）" in ddf.columns:
                    ddf["EPS（基本）"] = ddf["EPS（基本）"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "N/A")
                st.dataframe(ddf, use_container_width=True)
                fig_q = plot_xbrl_quarterly(xbrl_df, ticker)
                if fig_q: st.pyplot(fig_q)
            else:
                st.warning("XBRL データが取得できませんでした")
        eps_history = earnings_info.get("EPS履歴", [])
        with cols[1]:
            st.markdown("**📅 決算発表日・EPS サプライズ**")
            per_val = earnings_info.get("PER(予想)"); pbr_val = earnings_info.get("PBR")
            st.markdown(f"- **次回決算日**: {earnings_info.get('次回決算日','N/A')}")
            st.markdown(f"- **予想PER**: {f'{per_val:.1f}' if per_val else 'N/A'}")
            st.markdown(f"- **PBR**: {f'{pbr_val:.2f}' if pbr_val else 'N/A'}")
            if eps_history:
                st.dataframe(pd.DataFrame(eps_history), use_container_width=True)
                fig_eps = plot_eps_surprise(eps_history, ticker)
                if fig_eps: st.pyplot(fig_eps)
                beats = sum(1 for e in eps_history if e.get("サプライズ%") and e["サプライズ%"]>0)
                total = len([e for e in eps_history if e.get("サプライズ%") is not None])
                if total: st.metric("直近4四半期 EPS Beat率", f"{beats}/{total} ({beats/total*100:.0f}%)")
            else:
                st.info("EPS データが取得できませんでした")
        st.markdown("**🤖 AI 決算総合分析（日本語）**")
        xbrl_str = xbrl_df.to_string(index=False) if not xbrl_df.empty else "データなし"
        eps_str  = json.dumps(eps_history, ensure_ascii=False, indent=2) if eps_history else "データなし"
        with st.spinner("🤖 AI 分析中..."): ai_analysis = ai_earnings_analysis(ticker, xbrl_str, eps_str)
        st.info(ai_analysis)
        st.markdown("**🎯 ニュース センチメント**")
        with st.spinner("🤖 センチメント分析中..."): sentiment = ai_sentiment(tuple(headlines), ticker)
        st.info(sentiment)
        filings = get_edgar_filings(ticker, filing_type, count=4)
        if filings:
            st.markdown(f"**📄 SEC {filing_type} フィリング（直近4件）**")
            fdf = pd.DataFrame(filings)[["form","date","url"]]
            fdf.columns = ["種別","提出日","EDGARリンク"]
            st.dataframe(fdf, use_container_width=True,
                         column_config={"EDGARリンク": st.column_config.LinkColumn("EDGARリンク")})
        all_reports[ticker] = build_text_report(ticker, earnings_info, xbrl_df, headlines, sentiment, ai_analysis)

    if all_reports:
        st.divider()
        combined = "\n\n".join(all_reports.values())
        st.download_button("📥 全銘柄レポートをテキストでダウンロード",
                           data=combined.encode("utf-8"),
                           file_name=f"EarningsReport_{end_date}.txt", mime="text/plain")
        if enable_email:
            subject = f"【NASDAQ決算速報】{', '.join(selected_tickers)} | {end_date}"
            ok, msg = send_email_notification(subject, combined,
                                               attachment_bytes=combined.encode("utf-8"),
                                               filename=f"EarningsReport_{end_date}.txt")
            st.success(f"✅ {msg}") if ok else st.warning(f"⚠️ {msg}")


def page_news():
    st.title("📰 ニュース翻訳・センチメント分析")
    news_ticker = st.selectbox("銘柄を選択", nasdaq100_tickers,
                                index=nasdaq100_tickers.index("AAPL") if "AAPL" in nasdaq100_tickers else 0)
    max_news = st.slider("取得ニュース件数", 3, 20, 8)
    if not st.button("▶ ニュースを取得・翻訳", type="primary"):
        return
    with st.spinner("ニュース取得中..."):
        headlines = get_news_headlines(news_ticker, max_news)
    if not headlines:
        st.warning("ニュースが取得できませんでした（Yahoo Finance 制限の可能性）"); return
    st.markdown(f"### 📰 {news_ticker} 最新ニュース（英語原文）")
    for i, h in enumerate(headlines, 1): st.markdown(f"{i}. {h}")
    st.markdown("---")
    st.markdown("### 🎯 センチメント分析（AI）")
    with st.spinner("センチメント分析中..."): sentiment_result = ai_sentiment(tuple(headlines), news_ticker)
    st.info(sentiment_result)
    st.markdown("---")
    st.markdown("### 🌐 AI 日本語翻訳・要約")
    with st.spinner("翻訳・要約中..."):
        translation = ai_translate_and_summarize("\n".join(headlines), context=f"{news_ticker}に関するニュース見出し一覧")
    st.success(translation)
    end_date    = datetime.today().date()
    report_news = f"# {news_ticker} ニュースレポート ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
    report_news += "## 英語原文\n" + "\n".join(f"- {h}" for h in headlines)
    report_news += f"\n\n## センチメント分析\n{sentiment_result}"
    report_news += f"\n\n## 日本語翻訳・要約\n{translation}"
    st.download_button("📥 ニュースレポートを保存",
                       data=report_news.encode("utf-8"),
                       file_name=f"{news_ticker}_news_{end_date}.txt", mime="text/plain")


def page_momentum():
    st.title("🚀 モメンタムランキング")
    st.caption("当日・5日・20日リターンの加重平均スコア（0.5/0.3/0.2）で上昇力の強い銘柄を表示")

    with st.sidebar:
        st.header("設定")
        index_choice = st.selectbox("対象インデックス",
                                     ["NASDAQ-100", "Dow Jones 30", "S&P 500"])
        top_n = st.slider("表示件数", 10, 50, 20)

    if index_choice == "NASDAQ-100":
        tickers = nasdaq100_tickers
    elif index_choice == "Dow Jones 30":
        tickers = DOW30_TICKERS
    else:
        with st.spinner("S&P500 構成銘柄リスト取得中..."):
            tickers = get_sp500_tickers()
        if not tickers:
            st.warning("S&P500 リスト取得失敗。NASDAQ-100 で代替します。")
            tickers = nasdaq100_tickers

    if not st.button("▶ ランキングを計算", type="primary"):
        st.info(f"ボタンを押すと {len(tickers)} 銘柄のモメンタムを計算します。")
        return

    prog = st.progress(0, text="計算中...")
    with st.spinner(f"{len(tickers)} 銘柄のスコア計算中..."):
        mom_df = calc_momentum_scores(tuple(tickers))
    prog.empty()

    if mom_df.empty:
        st.error("データの取得に失敗しました。"); return

    top_df = mom_df.head(top_n)
    bot_df = mom_df.tail(top_n).sort_values("スコア")

    ct, cb = st.columns(2)
    with ct:
        st.markdown(f"**↑ 上昇モメンタム上位 {top_n}**")
        st.dataframe(top_df.style
                     .format("{:+.2f}", subset=["当日(%)","5日(%)","20日(%)","スコア"])
                     .background_gradient(subset=["スコア"], cmap="Greens"),
                     use_container_width=True, hide_index=True)
    with cb:
        st.markdown(f"**↓ 下落モメンタム上位 {top_n}**")
        st.dataframe(bot_df.style
                     .format("{:+.2f}", subset=["当日(%)","5日(%)","20日(%)","スコア"])
                     .background_gradient(subset=["スコア"], cmap="Reds_r"),
                     use_container_width=True, hide_index=True)

    fig = go.Figure([go.Bar(x=top_df["銘柄"], y=top_df["スコア"],
                            marker_color="#26A69A", name="上昇")])
    fig.update_layout(title=f"モメンタムスコア上位 {top_n}（{index_choice}）",
                      template="plotly_dark", height=400,
                      margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)


def page_sector_comparison():
    st.title("📡 セクター比較チャート")
    st.caption("光通信 vs 半導体バスケット / 期間正規化リターン比較（始値=100）")

    period_map = {"3ヶ月": 90, "6ヶ月": 180, "1年": 365, "2年": 730, "3年": 1095}

    with st.sidebar:
        st.header("設定")
        period        = st.selectbox("期間", list(period_map.keys()), index=2)
        show_indiv    = st.checkbox("個別銘柄を表示（凡例でON/OFF）", value=True)
        custom_tickers = st.multiselect("追加比較銘柄", nasdaq100_tickers, default=[])

    # バスケット構成を表示
    with st.expander("📋 バスケット構成"):
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**📡 光通信バスケット（等加重）**")
            for t in OPTICAL_TICKERS: st.markdown(f"- `{t}`")
        with bc2:
            st.markdown("**💾 半導体バスケット（等加重）**")
            for t in SEMI_TICKERS:  st.markdown(f"- `{t}`")
            for t in SEMI_ETF:      st.markdown(f"- `{t}` *(ETF)*")

    if not st.button("▶ 比較チャートを表示", type="primary"):
        return

    all_t = list(dict.fromkeys(OPTICAL_TICKERS + SEMI_TICKERS + SEMI_ETF + ["^NDX"] + custom_tickers))
    with st.spinner(f"価格データ取得中（{len(all_t)} 銘柄）..."):
        norm_df = get_normalized_prices(tuple(all_t), period_map[period])

    if norm_df.empty:
        st.error("データの取得に失敗しました。"); return

    # バスケット計算
    opt_cols  = [t for t in OPTICAL_TICKERS if t in norm_df.columns]
    semi_cols = [t for t in SEMI_TICKERS    if t in norm_df.columns]
    if opt_cols:  norm_df["📡 光通信（等加重）"] = norm_df[opt_cols].mean(axis=1)
    if semi_cols: norm_df["💾 半導体（等加重）"] = norm_df[semi_cols].mean(axis=1)

    # Plotly チャート
    fig = go.Figure()

    # バスケット（太線）
    for name, color in [("📡 光通信（等加重）","#FF6B35"), ("💾 半導体（等加重）","#42A5F5")]:
        if name in norm_df.columns:
            fig.add_scatter(x=norm_df.index, y=norm_df[name], name=name,
                            line=dict(color=color, width=3))

    # ベンチマーク
    if "^NDX" in norm_df.columns:
        fig.add_scatter(x=norm_df.index, y=norm_df["^NDX"], name="NASDAQ 100",
                        line=dict(color="#CCCCCC", width=1.5, dash="dot"))

    # ETF
    for t, color in [("SMH","#FFA726"), ("SOXX","#AB47BC")]:
        if t in norm_df.columns:
            fig.add_scatter(x=norm_df.index, y=norm_df[t], name=t,
                            line=dict(color=color, width=2, dash="dash"))

    # 個別銘柄（凡例ONのみ表示）
    if show_indiv:
        opt_colors  = ["#FF8C5A","#FFA87A","#FFC49A","#FFD9B3","#FFF0E0"]
        semi_colors = ["#64B5F6","#90CAF9","#BBDEFB","#C5E3F7","#E3F2FD"]
        for t, c in zip(opt_cols, opt_colors):
            fig.add_scatter(x=norm_df.index, y=norm_df[t], name=t,
                            line=dict(color=c, width=1, dash="dot"), visible="legendonly")
        for t, c in zip(semi_cols, semi_colors):
            fig.add_scatter(x=norm_df.index, y=norm_df[t], name=t,
                            line=dict(color=c, width=1, dash="dot"), visible="legendonly")

    # カスタム銘柄
    for t in custom_tickers:
        if t in norm_df.columns:
            fig.add_scatter(x=norm_df.index, y=norm_df[t], name=f"★ {t}",
                            line=dict(width=2))

    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig.update_layout(
        title=f"正規化価格比較（{period} / 始値 = 100）",
        template="plotly_dark", height=560,
        legend=dict(orientation="h", y=-0.25, x=0),
        yaxis_title="価格（始値=100）",
        margin=dict(l=50, r=20, t=60, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 期間リターン表
    st.markdown("#### 期間リターン一覧")
    ret_rows = []
    groups = [
        ("バスケット",    ["📡 光通信（等加重）","💾 半導体（等加重）"]),
        ("ETF",           SEMI_ETF),
        ("光通信 個別",   opt_cols),
        ("半導体 個別",   semi_cols),
        ("ベンチマーク",  ["^NDX"]),
        ("カスタム",      [t for t in custom_tickers if t in norm_df.columns]),
    ]
    for grp, tlist in groups:
        for t in tlist:
            if t in norm_df.columns:
                ret_rows.append({"グループ": grp, "銘柄/バスケット": t,
                                 "期間リターン(%)": norm_df[t].iloc[-1] - 100})
    if ret_rows:
        rdf = pd.DataFrame(ret_rows).sort_values("期間リターン(%)", ascending=False)
        st.dataframe(
            rdf.style.format("{:+.1f}%", subset=["期間リターン(%)"])
                     .background_gradient(subset=["期間リターン(%)"], cmap="RdYlGn", vmin=-50, vmax=200),
            use_container_width=True, hide_index=True,
        )


def page_sentiment_chart():
    st.title("📊 センチメント推移チャート")
    st.caption("テクニカル複合センチメント（RSI / MACD / VIX / モメンタム / SMA50）vs 指数価格")

    with st.sidebar:
        st.header("設定")
        index_choice = st.selectbox("対象指数",
                                     ["NASDAQ 100 (^NDX)", "S&P 500 (^GSPC)", "Dow Jones (^DJI)"])
        period_label = st.selectbox("期間", ["3ヶ月", "6ヶ月", "1年"], index=2)

    ticker_map = {
        "NASDAQ 100 (^NDX)": "^NDX",
        "S&P 500 (^GSPC)":   "^GSPC",
        "Dow Jones (^DJI)":  "^DJI",
    }
    days_map = {"3ヶ月": 90, "6ヶ月": 180, "1年": 365}

    with st.spinner("センチメント履歴を計算中..."):
        hist_df = calc_sentiment_history(ticker_map[index_choice], days_map[period_label])

    if hist_df.empty:
        st.error("データの取得に失敗しました。"); return

    # ── デュアル軸チャート ─────────────────────────────────────
    from plotly.subplots import make_subplots as _msp
    fig = _msp(specs=[[{"secondary_y": True}]])

    score  = hist_df["センチメントスコア"]
    bar_colors = ["#26A69A" if v >= 0 else "#EF5350" for v in score]
    fig.add_bar(x=hist_df.index, y=score, name="センチメントスコア",
                marker_color=bar_colors, opacity=0.7, secondary_y=False)
    fig.add_scatter(x=hist_df.index, y=hist_df["指数価格"],
                    name=index_choice.split("(")[0].strip(),
                    line=dict(color="#FFFFFF", width=2), secondary_y=True)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", secondary_y=False)

    fig.update_layout(
        title=f"テクニカルセンチメント vs {index_choice}（{period_label} / 5日移動平均）",
        template="plotly_dark", height=500,
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=60, r=60, t=60, b=20),
    )
    fig.update_yaxes(title_text="センチメントスコア（±7）", secondary_y=False)
    fig.update_yaxes(title_text="指数価格", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── ファクター別貢献チャート ──────────────────────────────
    st.markdown("#### ファクター別貢献度（5日移動平均）")
    factor_cols = ["RSI貢献","MACD貢献","モメンタム貢献","VIX貢献","SMA50貢献"]
    avail = [c for c in factor_cols if c in hist_df.columns]
    if avail:
        fig2 = go.Figure()
        colors_f = ["#AB47BC","#42A5F5","#FFA726","#26A69A","#EC407A"]
        for col, color in zip(avail, colors_f):
            fig2.add_scatter(x=hist_df.index, y=hist_df[col], name=col,
                             line=dict(color=color, width=1.3))
        fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig2.update_layout(template="plotly_dark", height=280,
                           legend=dict(orientation="h", y=1.05, x=0),
                           margin=dict(l=50, r=20, t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    # ── 直近メトリクス ────────────────────────────────────────
    st.markdown("#### 直近センチメント状況")
    latest = hist_df.iloc[-1]
    mc = st.columns(5)
    for col, key in zip(mc, ["センチメントスコア","RSI貢献","MACD貢献","VIX貢献","モメンタム貢献"]):
        val = latest.get(key, 0)
        col.metric(key, f"{val:.2f}")

    # ── スコア判定凡例 ────────────────────────────────────────
    with st.expander("📖 スコア判定基準"):
        st.markdown("""
| スコア | 判定 |
|--------|------|
| +7 以上 | 🟢 強気 |
| +3 〜 +6 | 🟡 やや強気 |
| -2 〜 +2 | ⚪ 中立 |
| -6 〜 -3 | 🟠 やや弱気 |
| -7 以下 | 🔴 弱気 |

**ファクター構成**: RSI(±2) + MACD(±1) + 5日モメンタム(±2) + SMA50(±1) + VIX(±2) = 最大 ±8
""")


# =============================================================
# マーケットスクリーン
# =============================================================

_SCREEN_SECTIONS = [
    ("🇺🇸 米国指数", [
        ("^DJI",      "ダウ平均"),
        ("^IXIC",     "ナスダック"),
        ("^GSPC",     "S&P 500"),
        ("^NDX",      "NASDAQ 100"),
        ("^RUT",      "Russell 2000"),
    ]),
    ("⚡ 先物 / CFD（24時間）", [
        ("ES=F",      "S&P先物"),
        ("NQ=F",      "NQ先物"),
        ("YM=F",      "ダウ先物"),
        ("RTY=F",     "Russell先物"),
    ]),
    ("📊 マクロ / 商品 / 為替", [
        ("^VIX",      "VIX"),
        ("^TNX",      "米10年債利回り"),
        ("GC=F",      "金 (Gold)"),
        ("CL=F",      "原油 (WTI)"),
        ("DX-Y.NYB",  "ドル指数 (DXY)"),
        ("USDJPY=X",  "USD/JPY"),
        ("EURUSD=X",  "EUR/USD"),
        ("GBPUSD=X",  "GBP/USD"),
    ]),
    ("₿ 仮想通貨", [
        ("BTC-USD",   "Bitcoin"),
        ("ETH-USD",   "Ethereum"),
        ("SOL-USD",   "Solana"),
        ("BNB-USD",   "BNB"),
    ]),
    ("💻 テック大型株", [
        ("AAPL",      "Apple"),
        ("MSFT",      "Microsoft"),
        ("NVDA",      "NVIDIA"),
        ("META",      "Meta"),
        ("TSLA",      "Tesla"),
        ("GOOGL",     "Alphabet"),
        ("AMZN",      "Amazon"),
        ("AVGO",      "Broadcom"),
    ]),
]


def _make_sparkline_fig(closes: pd.Series, is_up: bool) -> "go.Figure":
    color = "#26A69A" if is_up else "#EF5350"
    fill  = "rgba(38,166,154,0.15)" if is_up else "rgba(239,83,80,0.15)"
    fig = go.Figure(go.Scatter(
        x=closes.index,
        y=closes.values,
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x|%m/%d}  %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=110,
        margin=dict(l=50, r=5, t=4, b=28),
        xaxis=dict(
            visible=True, fixedrange=True,
            tickformat="%m/%d", nticks=3,
            tickfont=dict(size=9), showgrid=False,
        ),
        yaxis=dict(
            visible=True, fixedrange=True,
            tickfont=dict(size=9), nticks=3,
            showgrid=True, gridcolor="rgba(128,128,128,0.2)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _make_intraday_fig(prices: pd.Series, prev_close: float) -> "go.Figure":
    """当日5分足の価格チャート。±1%/±3%参照線付き、Y軸右側に価格表示"""
    last = prices.iloc[-1]
    color_line = "#26A69A" if last >= prev_close else "#EF5350"

    fig = go.Figure()

    # ±1%, ±3% 水平参照線（シアン系）
    for pct_level in [3, 1, -1, -3]:
        level = prev_close * (1 + pct_level / 100)
        clr   = "rgba(0,200,200,0.55)" if pct_level > 0 else "rgba(200,80,80,0.45)"
        dash  = "dot" if abs(pct_level) == 1 else "dash"
        fig.add_hline(
            y=level, line_width=0.8, line_dash=dash, line_color=clr,
            annotation_text=f"{pct_level:+d}%",
            annotation_position="right",
            annotation_font_size=8, annotation_font_color=clr,
        )

    # 前日終値ライン（白グレー）
    fig.add_hline(y=prev_close, line_width=1, line_dash="solid",
                  line_color="rgba(200,200,200,0.45)")

    # 価格ライン
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", line=dict(color=color_line, width=1.5),
        showlegend=False,
        hovertemplate="%{x|%H:%M ET}  %{y:,.2f}<extra></extra>",
    ))

    fig.update_layout(
        height=130,
        margin=dict(l=4, r=38, t=6, b=26),
        xaxis=dict(
            visible=True, fixedrange=True,
            tickformat="%H:%M", nticks=4,
            tickfont=dict(size=9), showgrid=False,
        ),
        yaxis=dict(
            visible=True, fixedrange=True, side="right",
            tickfont=dict(size=9), nticks=4, showgrid=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def page_market_screen():
    st.title("🖥️ マーケットスクリーン")
    st.caption("世界主要指数・先物・マクロ・仮想通貨・個別株のリアルタイム一覧（30日スパークライン付き）")

    NCOLS = 4

    # 全ティッカー収集
    all_instruments: list[tuple[str, str]] = []
    for _sec, items in _SCREEN_SECTIONS:
        all_instruments.extend(items)
    all_tickers = tuple(dict.fromkeys(t for t, _ in all_instruments))  # 重複除去・順序保持

    col_refresh, col_ts = st.columns([1, 5])
    if col_refresh.button("🔄 更新", key="screen_refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("データ取得中..."):
        quote_df  = get_quote_data(all_tickers)
        intraday  = get_intraday_data(all_tickers)
        sparklines = get_sparkline_data(all_tickers)

    if not quote_df.empty or intraday:
        ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        col_ts.caption(f"最終取得: {ts} (UTC)")

    for section_title, instruments in _SCREEN_SECTIONS:
        st.subheader(section_title)

        for row_start in range(0, len(instruments), NCOLS):
            row_items = instruments[row_start:row_start + NCOLS]
            # 端数を埋めるための空列調整
            cols = st.columns(NCOLS)

            for ci, (ticker, name) in enumerate(row_items):
                with cols[ci]:
                    q     = quote_df[quote_df["シンボル"] == ticker] if "シンボル" in quote_df.columns else pd.DataFrame()
                    price = float(q.iloc[0]["現在値"])  if not q.empty and pd.notna(q.iloc[0]["現在値"])  else None
                    chg   = float(q.iloc[0]["前日比"])  if not q.empty and pd.notna(q.iloc[0]["前日比"])  else None
                    chgp  = float(q.iloc[0]["前日比%"]) if not q.empty and pd.notna(q.iloc[0]["前日比%"]) else None

                    # v7 quote が失敗した場合はイントラデイ(v8)から現在値・前日比を算出
                    idata = intraday.get(ticker)
                    if price is None and idata is not None and len(idata["prices"]) > 0:
                        price      = float(idata["prices"].iloc[-1])
                        prev_close = idata["prev_close"]
                        chg        = price - prev_close
                        chgp       = (price / prev_close - 1) * 100 if prev_close else None

                    is_up     = (chgp or 0.0) >= 0
                    color_hex = "#26A69A" if is_up else "#EF5350"

                    # 価格フォーマット
                    if price is not None:
                        if price < 0.01:
                            price_str = f"{price:.6f}"
                        elif price < 1:
                            price_str = f"{price:.4f}"
                        elif price >= 10000:
                            price_str = f"{price:,.0f}"
                        else:
                            price_str = f"{price:,.2f}"
                    else:
                        price_str = "---"

                    chgp_str = f"{chgp:+.2f}%" if chgp is not None else "---"
                    chg_str  = f"{chg:+,.2f}"  if chg  is not None else ""

                    # ── カードヘッダー：銘柄名 + 前日比%（大きく）──
                    st.markdown(
                        f"<div style='font-size:11px;color:#999;margin-bottom:0px'>"
                        f"{name}&nbsp;<code style='font-size:10px;color:#555'>{ticker}</code></div>"
                        f"<div style='font-size:26px;font-weight:bold;color:{color_hex};line-height:1.15'>"
                        f"{chgp_str}</div>",
                        unsafe_allow_html=True,
                    )

                    # ── チャート：当日5分足（価格ベース）→ なければ30日スパークライン ──
                    if idata is not None and len(idata["prices"]) > 2:
                        fig = _make_intraday_fig(idata["prices"], idata["prev_close"])
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"staticPlot": False, "displayModeBar": False},
                            key=f"intra_{ticker}",
                        )
                    else:
                        closes = sparklines.get(ticker)
                        if closes is not None and len(closes) > 2:
                            spark = _make_sparkline_fig(closes, is_up)
                            st.plotly_chart(
                                spark,
                                use_container_width=True,
                                config={"staticPlot": True, "displayModeBar": False},
                                key=f"spark_{ticker}",
                            )

                    # ── カードフッター：現在値（大きく）+ 前日比絶対値 ──
                    st.markdown(
                        f"<div style='font-size:18px;font-weight:bold;color:var(--text-color);margin-top:2px'>"
                        f"{price_str}&nbsp;"
                        f"<span style='font-size:13px;color:{color_hex}'>{chg_str}</span></div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")


# =============================================================
# Navigation（サイドバー）
# =============================================================
with st.sidebar:
    st.markdown("### 🚀 米国株ダッシュボード")
    page_sel = st.radio(
        "ページ選択",
        ["🖥️ マーケットスクリーン",
         "📈 マーケット概況",
         "📊 パフォーマンス分析",
         "📉 テクニカル分析",
         "📋 決算分析",
         "📰 ニュース翻訳",
         "🚀 モメンタムランキング",
         "📡 セクター比較",
         "📊 センチメント推移"],
        label_visibility="collapsed",
    )
    st.divider()
    with st.expander("🤖 AI プロバイダー状態"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Gemini",     "✅" if GEMINI_API_KEY     else "❌")
        c2.metric("Groq",       "✅" if GROQ_API_KEY       else "❌")
        c3.metric("OpenRouter", "✅" if OPENROUTER_API_KEY else "❌")
    st.caption("v3.0 | Tiingo / Stooq / Yahoo / SEC EDGAR")

PAGE_MAP = {
    "🖥️ マーケットスクリーン":  page_market_screen,
    "📈 マーケット概況":        page_market,
    "📊 パフォーマンス分析":     page_performance,
    "📉 テクニカル分析":         page_tech,
    "📋 決算分析":               page_earnings,
    "📰 ニュース翻訳":           page_news,
    "🚀 モメンタムランキング":   page_momentum,
    "📡 セクター比較":           page_sector_comparison,
    "📊 センチメント推移":       page_sentiment_chart,
}
PAGE_MAP[page_sel]()

# ── サイドバー ────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 設定")
    years = st.slider("分析対象期間（年）", 1, 10, 3)
    st.divider()
    st.subheader("📋 決算分析設定")
    selected_tickers = st.multiselect(
        "分析する銘柄を選択", nasdaq100_tickers,
        default=["AAPL", "MSFT", "NVDA"], max_selections=10,
    )
    filing_type  = st.selectbox("SEC フィリング種別", ["10-K", "10-Q"])
    st.divider()
    enable_email = st.checkbox("決算レポートをメールで通知", value=False)
    st.divider()
    st.info(
        "💡 株価データ フォールバック順\n"
        "1️⃣ Tiingo（APIキー必要）\n"
        "2️⃣ Stooq（無料・高速）\n"
        "3️⃣ Yahoo Finance（無料）\n\n"
        "AI フォールバック順\n"
        "1️⃣ Gemini → 2️⃣ Groq → 3️⃣ OpenRouter"
    )
    st.caption("v2.5 | SEC EDGAR / Tiingo / Stooq / Yahoo Finance")

end_date   = datetime.today().date()
start_date = end_date - timedelta(days=years * 365)

# =============================================================
# Tab 0: マーケット概況
# =============================================================
with tab_market:
    st.subheader("📈 マーケット概況")

    refresh_col, _ = st.columns([1, 5])
    if refresh_col.button("🔄 データ更新", key="refresh_mkt"):
        st.cache_data.clear()
        st.rerun()

    all_mkt_tickers = tuple(t for t, _ in MARKET_INDICES + MARKET_MACRO)
    with st.spinner("市場データ取得中..."):
        mkt_df = get_quote_data(all_mkt_tickers)

    if not mkt_df.empty:
        st.markdown("#### 主要指数")
        idx_cols = st.columns(4)
        for col, (sym, label) in zip(idx_cols, MARKET_INDICES):
            row = mkt_df[mkt_df["シンボル"] == sym]
            if not row.empty and pd.notna(row.iloc[0]["現在値"]):
                price = row.iloc[0]["現在値"]
                chgp  = row.iloc[0]["前日比%"]
                col.metric(label, f"{price:,.2f}", f"{chgp:+.2f}%" if pd.notna(chgp) else None)

        st.markdown("#### マクロ経済指標")
        mac_cols = st.columns(5)
        for col, (sym, label) in zip(mac_cols, MARKET_MACRO):
            row = mkt_df[mkt_df["シンボル"] == sym]
            if not row.empty and pd.notna(row.iloc[0]["現在値"]):
                price = row.iloc[0]["現在値"]
                chgp  = row.iloc[0]["前日比%"]
                col.metric(label, f"{price:.2f}", f"{chgp:+.2f}%" if pd.notna(chgp) else None,
                           delta_color="inverse" if sym == "^VIX" else "normal")
    else:
        st.warning("市場データの取得に失敗しました。しばらくしてから更新ボタンを押してください。")

    st.divider()
    st.markdown("#### 個別銘柄 リアルタイム株価")
    rt_tickers = st.multiselect(
        "銘柄を選択（最大20銘柄）",
        nasdaq100_tickers,
        default=["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN"],
        max_selections=20,
        key="rt_tickers",
    )
    if rt_tickers:
        with st.spinner("株価データ取得中..."):
            rt_df = get_quote_data(tuple(rt_tickers))
        if not rt_df.empty:
            disp = rt_df.copy()
            disp["現在値"] = disp["現在値"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
            disp["前日比"]  = disp["前日比"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
            disp["前日比%"] = disp["前日比%"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
            st.dataframe(disp, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 📰 最新ニュース")
    with st.spinner("ニュース取得中（全ソース並列）..."):
        mkt_news = get_market_news()

    def _render_news_tabs(items: list[dict], ai_key: str) -> None:
        if not items:
            st.info("取得できませんでした。しばらくしてから更新ボタンを押してください。")
            return

        # タイトル一括翻訳（キャッシュ済みなら即表示）
        with st.spinner("タイトル翻訳中..."):
            title_map = ai_translate_titles(tuple(item["見出し"] for item in items))

        sources = list(dict.fromkeys(item["ソース"] for item in items))
        ntabs = st.tabs(sources)
        for ntab, src in zip(ntabs, sources):
            with ntab:
                for idx, item in enumerate(i for i in items if i["ソース"] == src):
                    orig = item["見出し"]
                    ja   = title_map.get(orig, orig)
                    date_str = f"  `{item['日時']}`" if item["日時"] else ""

                    if item["リンク"]:
                        st.markdown(f"**[{ja}]({item['リンク']})**{date_str}")
                    else:
                        st.markdown(f"**{ja}**{date_str}")
                    st.caption(f"🔤 {orig}")

                    desc = item.get("概要", "")
                    if desc:
                        with st.expander("📄 記事概要を翻訳"):
                            st.caption(desc)
                            if st.button("🤖 日本語に翻訳", key=f"desc_{ai_key}_{src}_{idx}"):
                                with st.spinner("翻訳中..."):
                                    desc_ja = ai_translate_description(desc)
                                st.info(desc_ja)
                    st.markdown("---")

    market_items = [i for i in mkt_news if i["カテゴリ"] == "market"]
    tech_items   = [i for i in mkt_news if i["カテゴリ"] == "tech"]

    col_m, col_t = st.columns(2)
    with col_m:
        st.markdown("**📊 市場全般**")
        _render_news_tabs(market_items, "market")
    with col_t:
        st.markdown("**💻 ハイテク系**")
        _render_news_tabs(tech_items, "tech")

# =============================================================
# Tab 1: パフォーマンス分析
# =============================================================
with tab_perf:
    st.subheader("📊 NASDAQ-100 パフォーマンス分析")
    st.caption("📡 株価データ: Tiingo → Stooq → Yahoo Finance の順で自動フォールバック")

    if st.button("▶ 全銘柄パフォーマンス分析を実行", type="primary"):
        with st.spinner("ベンチマーク（QQQ）データ取得中..."):
            market_returns = get_market_data(TIINGO_API_KEY, start_date, end_date)

        if market_returns is None:
            st.error("市場データ（QQQ）の取得に全プロバイダーで失敗しました")
        else:
            progress = st.progress(0, text="銘柄データ取得中...")
            results  = []
            # Tiingo は並列耐性が高いので 20 まで引き上げて高速化
            # Stooq/Yahoo はレート制限があるため 5 に抑える
            max_workers = 20 if TIINGO_API_KEY else 5
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        analyze_ticker, t, market_returns, TIINGO_API_KEY, start_date, end_date
                    ): t
                    for t in nasdaq100_tickers
                }
                for i, f in enumerate(concurrent.futures.as_completed(futures)):
                    r = f.result()
                    if r:
                        results.append(r)
                    progress.progress((i + 1) / len(futures), text=f"取得中... {i+1}/{len(futures)}")
            progress.empty()

            df = pd.DataFrame(results)
            if df.empty:
                st.error("有効な分析結果がありません")
            else:
                df.insert(0, "シャープレシオ順位", df["シャープレシオ"].rank(ascending=False, method="min").astype(int))
                df.insert(0, "値上がり順位",       df["年間リターン"].rank(ascending=False, method="min").astype(int))
                df = df.sort_values("値上がり順位").reset_index(drop=True)

                # データソース内訳を表示
                if "データソース" in df.columns:
                    source_counts = df["データソース"].value_counts()
                    source_info = " | ".join([f"{k}: {v}銘柄" for k, v in source_counts.items()])
                    st.info(f"📡 使用データソース内訳: {source_info}")

                # 銘柄コード → Yahoo Finance URL 列を追加
                df["Yahoo Finance"] = df["銘柄"].apply(
                    lambda t: f"https://finance.yahoo.com/quote/{t}/"
                )

                # 表示列順: データソースを末尾、Yahoo Financeを銘柄の直後
                base_cols = ["値上がり順位", "シャープレシオ順位", "銘柄", "Yahoo Finance",
                             "年間リターン", "年間リスク", "シャープレシオ", "ベータ", "アルファ", "レジデュアルリスク"]
                display_cols = [c for c in base_cols if c in df.columns]
                if "データソース" in df.columns:
                    display_cols.append("データソース")

                st.dataframe(
                    df[display_cols].style
                    .format("{:.2%}", subset=["年間リターン", "年間リスク", "アルファ", "レジデュアルリスク"])
                    .format("{:.2f}", subset=["シャープレシオ", "ベータ"])
                    .format("{:d}",   subset=["値上がり順位", "シャープレシオ順位"]),
                    use_container_width=True,
                    column_config={
                        "Yahoo Finance": st.column_config.LinkColumn(
                            "Yahoo Finance",
                            display_text="📈 表示",   # セルに表示するラベル
                        ),
                    },
                )

                top = df.sort_values("シャープレシオ", ascending=False).head(20)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
                ax1.bar(top["銘柄"], top["年間リターン"] * 100, label="リターン(%)", color="#4472C4")
                ax1.bar(top["銘柄"], top["年間リスク"]   * 100, label="リスク(%)",   color="#ED7D31", alpha=0.3)
                ax1.legend(); ax1.set_title("年間リターン / リスク（上位20銘柄）")
                ax2.bar(top["銘柄"], top["シャープレシオ"], color="#70AD47")
                ax2.set_title("シャープレシオ（上位20銘柄）")
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("🤖 AIによる企業解説（上位3社）")
                for t in top["銘柄"].head(3):
                    with st.expander(f"💡 {t} の概要"):
                        st.write(ai_company_summary(t))

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Analysis")
                st.download_button("📥 Excelでダウンロード", data=output.getvalue(),
                                   file_name=f"Nasdaq100_Analysis_{end_date}.xlsx")

# =============================================================
# Tab 2: テクニカル分析
# =============================================================
with tab_tech:
    st.subheader("📉 テクニカル分析チャート")

    tc1, tc2 = st.columns([3, 1])
    with tc1:
        tech_ticker = st.selectbox(
            "銘柄を選択",
            nasdaq100_tickers,
            index=nasdaq100_tickers.index("AAPL") if "AAPL" in nasdaq100_tickers else 0,
            key="tech_ticker_sel",
        )
    with tc2:
        period_label = st.selectbox(
            "期間", ["3ヶ月", "6ヶ月", "1年", "2年", "3年"], index=2, key="tech_period_sel"
        )

    period_map = {"3ヶ月": 90, "6ヶ月": 180, "1年": 365, "2年": 730, "3年": 1095}

    if st.button("▶ チャートを表示", type="primary", key="btn_tech"):
        with st.spinner(f"📡 {tech_ticker} OHLCV データ取得中..."):
            ohlcv = get_ohlcv_data(tech_ticker, period_map[period_label])

        if ohlcv.empty:
            st.error("データの取得に失敗しました。しばらくしてから再試行してください。")
        else:
            st.plotly_chart(build_tech_chart(ohlcv, tech_ticker), use_container_width=True)

            # 直近のテクニカルシグナル
            close    = ohlcv["Close"]
            rsi_now  = calc_rsi(close).iloc[-1]
            macd_l, _, hist = calc_macd(close)

            sig_cols = st.columns(3)
            rsi_status = "🔴 買われすぎ" if rsi_now > 70 else "🟢 売られすぎ" if rsi_now < 30 else "⚪ 中立"
            sig_cols[0].metric("RSI (14)", f"{rsi_now:.1f}", rsi_status, delta_color="off")

            macd_signal = (
                "🟢 ゴールデンクロス" if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0 else
                "🔴 デッドクロス"     if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0 else
                "↑ 強気"             if hist.iloc[-1] > 0 else "↓ 弱気"
            )
            sig_cols[1].metric("MACD", f"{macd_l.iloc[-1]:.3f}", macd_signal, delta_color="off")

            sma50 = close.rolling(50).mean().iloc[-1]
            vs_sma = "↑ SMA50 上回り" if close.iloc[-1] > sma50 else "↓ SMA50 下回り"
            sig_cols[2].metric("現在値 vs SMA50", f"${close.iloc[-1]:.2f}", vs_sma,
                               delta_color="normal" if close.iloc[-1] > sma50 else "inverse")

# =============================================================
# Tab 3: 決算分析
# =============================================================
with tab_earnings:
    st.subheader("📋 決算分析（SEC EDGAR + Yahoo Finance）")
    if not selected_tickers:
        st.info("サイドバーで銘柄を選択してください")
    elif st.button("▶ 決算データを取得・分析", type="primary"):
        all_reports: dict[str, str] = {}

        for ticker in selected_tickers:
            st.markdown(f"---\n### 🔍 {ticker}")

            with st.spinner(f"📡 {ticker} XBRL データ取得中..."):
                xbrl_df = get_xbrl_financials(ticker)

            with st.spinner(f"📡 {ticker} 決算・EPS データ取得中..."):
                earnings_info = get_earnings_data(ticker)

            with st.spinner(f"📡 {ticker} ニュース取得中..."):
                headlines = get_news_headlines(ticker, 5)

            cols = st.columns([1, 1])

            with cols[0]:
                st.markdown("**📈 XBRL 財務データ（SEC EDGAR）**")
                if not xbrl_df.empty:
                    disp_df = xbrl_df.copy()
                    for c in ["売上高", "純利益"]:
                        if c in disp_df.columns:
                            disp_df[c] = disp_df[c].apply(
                                lambda v: f"${v/1e9:.2f}B" if pd.notna(v) and v != 0 else "N/A"
                            )
                    if "EPS（基本）" in disp_df.columns:
                        disp_df["EPS（基本）"] = disp_df["EPS（基本）"].apply(
                            lambda v: f"${v:.2f}" if pd.notna(v) else "N/A"
                        )
                    st.dataframe(disp_df, use_container_width=True)
                    fig_q = plot_xbrl_quarterly(xbrl_df, ticker)
                    if fig_q:
                        st.pyplot(fig_q)
                else:
                    st.warning("XBRL データが取得できませんでした")

            eps_history = earnings_info.get("EPS履歴", [])
            with cols[1]:
                st.markdown("**📅 決算発表日・EPS サプライズ**")
                per_val = earnings_info.get("PER(予想)")
                pbr_val = earnings_info.get("PBR")
                st.markdown(f"- **次回決算日**: {earnings_info.get('次回決算日', 'N/A')}")
                st.markdown(f"- **予想PER**: {f'{per_val:.1f}' if per_val else 'N/A'}")
                st.markdown(f"- **PBR**: {f'{pbr_val:.2f}' if pbr_val else 'N/A'}")
                if eps_history:
                    st.dataframe(pd.DataFrame(eps_history), use_container_width=True)
                    fig_eps = plot_eps_surprise(eps_history, ticker)
                    if fig_eps:
                        st.pyplot(fig_eps)
                    beats = sum(1 for e in eps_history if e.get("サプライズ%") and e["サプライズ%"] > 0)
                    total = len([e for e in eps_history if e.get("サプライズ%") is not None])
                    if total:
                        st.metric("直近4四半期 EPS Beat率", f"{beats}/{total} ({beats/total*100:.0f}%)")
                else:
                    st.info("EPS データが取得できませんでした（Yahoo Finance 制限の可能性）")

            st.markdown("**🤖 AI 決算総合分析（日本語）**")
            xbrl_str = xbrl_df.to_string(index=False) if not xbrl_df.empty else "データなし"
            eps_str  = json.dumps(eps_history, ensure_ascii=False, indent=2) if eps_history else "データなし"
            with st.spinner("🤖 AI 分析中..."):
                ai_analysis = ai_earnings_analysis(ticker, xbrl_str, eps_str)
            st.info(ai_analysis)

            st.markdown("**🎯 ニュース センチメント**")
            with st.spinner("🤖 センチメント分析中..."):
                sentiment = ai_sentiment(tuple(headlines), ticker)
            st.info(sentiment)

            filings = get_edgar_filings(ticker, filing_type, count=4)
            if filings:
                st.markdown(f"**📄 SEC {filing_type} フィリング（直近4件）**")
                fdf = pd.DataFrame(filings)[["form", "date", "url"]]
                fdf.columns = ["種別", "提出日", "EDGARリンク"]
                st.dataframe(fdf, use_container_width=True,
                             column_config={"EDGARリンク": st.column_config.LinkColumn("EDGARリンク")})

            all_reports[ticker] = build_text_report(
                ticker, earnings_info, xbrl_df, headlines, sentiment, ai_analysis
            )

        if all_reports:
            st.divider()
            combined = "\n\n".join(all_reports.values())
            st.download_button(
                "📥 全銘柄レポートをテキストでダウンロード",
                data=combined.encode("utf-8"),
                file_name=f"EarningsReport_{end_date}.txt",
                mime="text/plain",
            )
            if enable_email:
                subject = f"【NASDAQ決算速報】{', '.join(selected_tickers)} | {end_date}"
                ok, msg = send_email_notification(subject, combined,
                                                   attachment_bytes=combined.encode("utf-8"),
                                                   filename=f"EarningsReport_{end_date}.txt")
                st.success(f"✅ {msg}") if ok else st.warning(f"⚠️ {msg}")

# =============================================================
# Tab 4: ニュース翻訳・センチメント
# =============================================================
with tab_news:
    st.subheader("📰 最新ニュース 自動日本語翻訳・センチメント分析")
    news_ticker = st.selectbox(
        "銘柄を選択", nasdaq100_tickers,
        index=nasdaq100_tickers.index("AAPL") if "AAPL" in nasdaq100_tickers else 0,
    )
    max_news = st.slider("取得ニュース件数", 3, 20, 8)

    if st.button("▶ ニュースを取得・翻訳", type="primary"):
        with st.spinner("ニュース取得中..."):
            headlines = get_news_headlines(news_ticker, max_news)

        if not headlines:
            st.warning("ニュースが取得できませんでした（Yahoo Finance 制限の可能性）")
        else:
            st.markdown(f"### 📰 {news_ticker} 最新ニュース（英語原文）")
            for i, h in enumerate(headlines, 1):
                st.markdown(f"{i}. {h}")

            st.markdown("---")
            st.markdown("### 🎯 センチメント分析（AI）")
            with st.spinner("センチメント分析中..."):
                sentiment_result = ai_sentiment(tuple(headlines), news_ticker)
            st.info(sentiment_result)

            st.markdown("---")
            st.markdown("### 🌐 AI 日本語翻訳・要約")
            with st.spinner("翻訳・要約中..."):
                translation = ai_translate_and_summarize(
                    "\n".join(headlines), context=f"{news_ticker}に関するニュース見出し一覧"
                )
            st.success(translation)

            report_news  = f"# {news_ticker} ニュースレポート ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
            report_news += "## 英語原文\n" + "\n".join(f"- {h}" for h in headlines)
            report_news += f"\n\n## センチメント分析\n{sentiment_result}"
            report_news += f"\n\n## 日本語翻訳・要約\n{translation}"
            st.download_button(
                "📥 ニュースレポートを保存",
                data=report_news.encode("utf-8"),
                file_name=f"{news_ticker}_news_{end_date}.txt",
                mime="text/plain",
            )
