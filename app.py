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
import os, io, json, time, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import xml.etree.ElementTree as ET

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
    "ANET","APH","GLW","COHR","AAOI","LITE",
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
# UI
# =============================================================
st.title("🚀 NASDAQ-100 プロ仕様 銘柄分析ダッシュボード v2.5")

with st.expander("🤖 AI プロバイダー状態", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("1️⃣ Gemini",     "✅ 設定済" if GEMINI_API_KEY     else "❌ 未設定", "gemini-2.0-flash → lite")
    c2.metric("2️⃣ Groq",       "✅ 設定済" if GROQ_API_KEY       else "❌ 未設定", "llama-3.3-70b → 8b")
    c3.metric("3️⃣ OpenRouter", "✅ 設定済" if OPENROUTER_API_KEY else "❌ 未設定", "gemini-flash → llama free")
    if not any([GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY]):
        st.warning("⚠️ AI APIキーが未設定です")

tab1, tab2, tab3 = st.tabs(["📊 パフォーマンス分析", "📋 決算分析（EDGAR・EPS）", "📰 ニュース翻訳・センチメント"])

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
# Tab 1: パフォーマンス分析
# =============================================================
with tab1:
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

                # 表示時はデータソース列を末尾に移動
                display_cols = [c for c in df.columns if c != "データソース"] + (
                    ["データソース"] if "データソース" in df.columns else []
                )

                st.dataframe(
                    df[display_cols].style
                    .format("{:.2%}", subset=["年間リターン", "年間リスク", "アルファ", "レジデュアルリスク"])
                    .format("{:.2f}", subset=["シャープレシオ", "ベータ"])
                    .format("{:d}",   subset=["値上がり順位", "シャープレシオ順位"]),
                    use_container_width=True,
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
# Tab 2: 決算分析
# =============================================================
with tab2:
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
# Tab 3: ニュース翻訳・センチメント
# =============================================================
with tab3:
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
