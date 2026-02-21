
# -*- coding: utf-8 -*-
"""
NASDAQ-100 プロ仕様 銘柄分析ダッシュボード v2.1
─────────────────────────────────────────────────────
  ① SEC EDGAR 10-Q/10-K 取得 & XBRL 財務数値抽出
  ② Yahoo Finance 決算発表日・EPSサプライズ
  ③ 英文ニュース自動日本語翻訳
  ④ 過去4四半期比較グラフ
  ⑤ EPSサプライズ / コンセンサス比較
  ⑥ センチメント分析
  ⑦ テキスト保存 & メール通知
  ⑧ 複数銘柄一括処理
  ⑨ AIフォールバック: Gemini → Groq → OpenRouter
─────────────────────────────────────────────────────
Streamlit Secrets:
  TIINGO_API_KEY      : 必須
  GEMINI_API_KEY      : 任意（第1優先AI）
  GROQ_API_KEY        : 任意（第2優先AI）
  OPENROUTER_API_KEY  : 任意（第3優先AI）
  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / NOTIFY_EMAIL : 任意
"""

# ===============================
# imports
# ===============================
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import concurrent.futures
import os, io, json, re, time, smtplib, textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import xml.etree.ElementTree as ET

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ===============================
# フォント設定
# ===============================
FONT_CANDIDATES = ["fonts/NotoSansJP-VariableFont_wght.ttf"]
for _font in FONT_CANDIDATES:
    if os.path.exists(_font):
        fm.fontManager.addfont(_font)
        _prop = fm.FontProperties(fname=_font)
        plt.rcParams["font.family"] = _prop.get_name()
        break
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["IPAexGothic", "TakaoPGothic", "DejaVu Sans"]

# ===============================
# ページ設定
# ===============================
st.set_page_config(page_title="NASDAQ-100 プロ分析ツール", layout="wide", page_icon="🚀")

# ===============================
# Secrets 読み込み
# ===============================
TIINGO_API_KEY      = st.secrets.get("TIINGO_API_KEY", "")
GEMINI_API_KEY      = st.secrets.get("GEMINI_API_KEY", "")
GROQ_API_KEY        = st.secrets.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = st.secrets.get("OPENROUTER_API_KEY", "")
SMTP_HOST           = st.secrets.get("SMTP_HOST", "")
SMTP_PORT           = int(st.secrets.get("SMTP_PORT", 587))
SMTP_USER           = st.secrets.get("SMTP_USER", "")
SMTP_PASS           = st.secrets.get("SMTP_PASS", "")
NOTIFY_EMAIL        = st.secrets.get("NOTIFY_EMAIL", "")

if not TIINGO_API_KEY:
    st.error("⚠️ TIINGO_API_KEY が設定されていません")
    st.stop()

# Gemini 初期化
if GEMINI_API_KEY and _GENAI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

# 利用可能バックエンド一覧
_ai_backends = []
if GEMINI_API_KEY and _GENAI_AVAILABLE:
    _ai_backends.append("🟢 Gemini")
if GROQ_API_KEY:
    _ai_backends.append("🟡 Groq")
if OPENROUTER_API_KEY:
    _ai_backends.append("🔵 OpenRouter")
if not _ai_backends:
    _ai_backends.append("⚪ AI未設定")

# ===============================
# NASDAQ-100 ティッカー
# ===============================
nasdaq100_tickers = sorted(set([
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","AVGO","COST","PEP",
    "ADBE","NFLX","AMD","INTC","CSCO","QCOM","TXN","AMAT","HON","INTU",
    "SBUX","BKNG","ADP","REGN","VRTX","LRCX","MU","PANW","KLAC","CDNS",
    "ADI","NXPI","FTNT","WDAY","SNPS","MELI","CRWD","CTAS","VRT","CEG",
    "ANET","SMCI","PLTR","APH","GLW",
]))

EDGAR_HEADERS = {
    "User-Agent": "NasdaqDashboard/2.1 research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# ================================================================
# ✨ AI ユニファイドクライアント（Gemini → Groq → OpenRouter）
# ================================================================

# Gemini で試せる最新モデルリスト（上から順に試す）
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

# Groq モデル
GROQ_MODEL = "llama-3.3-70b-versatile"

# OpenRouter モデル（無料/低コスト枠）
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"


def _try_gemini(prompt: str) -> tuple[str | None, str | None]:
    """Gemini で複数モデルを順番に試す。成功 → (text, model名)、失敗 → (None, error)"""
    if not (GEMINI_API_KEY and _GENAI_AVAILABLE):
        return None, "Gemini未設定"
    last_err = ""
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return resp.text, model_name
        except Exception as e:
            last_err = str(e)
            # 404 / not found → 次のモデルへ
            if "404" in last_err or "not found" in last_err.lower() or "not supported" in last_err.lower():
                continue
            # その他エラーはすぐ諦める
            break
    return None, f"Gemini全モデル失敗: {last_err}"


def _try_groq(prompt: str) -> tuple[str | None, str | None]:
    """Groq API（OpenAI互換）を呼ぶ"""
    if not GROQ_API_KEY:
        return None, "Groq未設定"
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
            timeout=30,
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            return text, f"Groq/{GROQ_MODEL}"
        return None, f"Groq HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"Groq例外: {e}"


def _try_openrouter(prompt: str) -> tuple[str | None, str | None]:
    """OpenRouter API を呼ぶ"""
    if not OPENROUTER_API_KEY:
        return None, "OpenRouter未設定"
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://nasdaq-dashboard.streamlit.app",
                "X-Title": "NASDAQ Dashboard",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
            },
            timeout=30,
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            return text, f"OpenRouter/{OPENROUTER_MODEL}"
        return None, f"OpenRouter HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"OpenRouter例外: {e}"


def ai_generate(prompt: str, label: str = "") -> tuple[str, str]:
    """
    Gemini → Groq → OpenRouter の順で試みる統合AI関数。
    Returns: (生成テキスト, 使用バックエンド名)
    """
    errors = []

    # 1) Gemini
    text, info = _try_gemini(prompt)
    if text:
        return text, info

    errors.append(f"Gemini: {info}")

    # 2) Groq
    text, info = _try_groq(prompt)
    if text:
        return text, info
    errors.append(f"Groq: {info}")

    # 3) OpenRouter
    text, info = _try_openrouter(prompt)
    if text:
        return text, info
    errors.append(f"OpenRouter: {info}")

    # 全失敗
    err_detail = " | ".join(errors)
    return f"⚠️ AI生成失敗（{label}）: {err_detail}", "失敗"


# ================================================================
# AI 高レベル関数
# ================================================================

def ai_company_summary(ticker: str) -> tuple[str, str]:
    """企業概要（日本語）"""
    prompt = f"""
米国株「{ticker}」について以下を投資家向けに日本語300文字以内で要約してください。
・事業内容 ・強み ・主な収益源
"""
    return ai_generate(prompt, label=f"{ticker}企業概要")


def ai_translate_and_summarize(text: str, context: str = "") -> tuple[str, str]:
    """英文翻訳・要約"""
    prompt = f"""以下の英文テキストを日本語に翻訳し、投資家向けに300文字以内で要約してください。
{f'背景情報: {context}' if context else ''}

英文:
{text}

出力形式:
【翻訳要約】
（ここに日本語で記載）
"""
    return ai_generate(prompt, label="翻訳要約")


def ai_sentiment(headlines: list, ticker: str) -> tuple[str, str]:
    """センチメント分析"""
    if not headlines:
        return "（ニュースデータなし）", "スキップ"
    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""以下は米国株「{ticker}」の最新ニュース見出しです。
センチメント（強気/弱気/中立）を判定し、理由を日本語200文字以内で説明してください。

ニュース見出し:
{headlines_text}

出力形式:
センチメント: [強気 / 弱気 / 中立]
理由: （日本語で説明）
"""
    return ai_generate(prompt, label=f"{ticker}センチメント")


def ai_earnings_analysis(ticker: str, xbrl_df: pd.DataFrame, eps_data: list) -> tuple[str, str]:
    """決算総合分析"""
    xbrl_str = xbrl_df.to_string(index=False) if not xbrl_df.empty else "データなし"
    eps_str = json.dumps(eps_data, ensure_ascii=False, indent=2) if eps_data else "データなし"
    prompt = f"""米国株「{ticker}」の決算データを分析し、投資家向け日本語レポートを作成してください。

## XBRL財務データ（過去4四半期）
{xbrl_str}

## EPS履歴・サプライズ
{eps_str}

以下の観点で400文字以内で分析：
1. 売上・利益トレンド
2. EPSサプライズの傾向（beat/miss）
3. 投資判断上の注目点
"""
    return ai_generate(prompt, label=f"{ticker}決算分析")


# ================================================================
# データ取得関数
# ================================================================

# ================================================================
# 株価データ取得 フォールバックチェーン
# Tiingo → Stooq → Yahoo Finance
# ================================================================

def _extract_close(df: pd.DataFrame) -> pd.Series:
    """adjClose -> close -> 数値カラム の順で価格カラムを探す"""
    for col in ["adjClose", "close", "adjclose", "Close", "AdjClose"]:
        if col in df.columns:
            return df[col]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        return df[num_cols[0]]
    raise KeyError(f"価格カラムが見つかりません。カラム: {list(df.columns)}")


def _returns_from_series(price: pd.Series) -> pd.Series:
    """価格 Series → 日次リターン Series（日付インデックスをtz-naive化）"""
    price = price.copy()
    price.index = pd.to_datetime(price.index).tz_localize(None)
    price = price.sort_index()
    return price.pct_change().dropna()


# ── Tiingo ──────────────────────────────────────────────────────
def _fetch_tiingo(ticker: str, api_key: str, start, end) -> tuple:
    """戻り値: (price_series or None, error_str or None)"""
    if not api_key:
        return None, "Tiingo APIキー未設定"
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": str(start),
        "endDate": str(end),
        "resampleFreq": "daily",
        "token": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=20,
                         headers={"Content-Type": "application/json"})
        if r.status_code == 401:
            return None, "Tiingo: 認証エラー（APIキーを確認）"
        if r.status_code == 403:
            return None, "Tiingo: アクセス拒否（プラン制限の可能性）"
        if r.status_code != 200:
            return None, f"Tiingo: HTTP {r.status_code}"
        data = r.json()
        if not data:
            return None, "Tiingo: レスポンスが空"
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return _extract_close(df), None
    except requests.exceptions.Timeout:
        return None, "Tiingo: タイムアウト"
    except requests.exceptions.ConnectionError:
        return None, "Tiingo: 接続エラー"
    except Exception as e:
        return None, f"Tiingo: {type(e).__name__}: {e}"


# ── Stooq ───────────────────────────────────────────────────────
def _fetch_stooq(ticker: str, start, end) -> tuple:
    """
    Stooq CSV エンドポイント（APIキー不要・完全無料）
    NASDAQ銘柄は "{ticker}.US" 形式
    """
    # Stooq のティッカー形式に変換（QQQ → QQQ.US）
    stooq_symbol = f"{ticker}.US"
    url = (
        f"https://stooq.com/q/d/l/"
        f"?s={stooq_symbol.lower()}"
        f"&d1={str(start).replace('-', '')}"
        f"&d2={str(end).replace('-', '')}"
        f"&i=d"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None, f"Stooq: HTTP {r.status_code}"
        # Stooq はデータなしのとき "No data" を返す
        if "No data" in r.text or len(r.text.strip()) < 30:
            return None, f"Stooq: データなし ({stooq_symbol})"
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        if df.empty:
            return None, "Stooq: CSVが空"
        # カラム名を正規化
        df.columns = [c.strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        # Close カラムを使用
        close_col = next((c for c in df.columns if c.lower() == "close"), None)
        if close_col is None:
            return None, f"Stooq: Closeカラムが見つかりません。カラム: {list(df.columns)}"
        return df[close_col], None
    except Exception as e:
        return None, f"Stooq: {type(e).__name__}: {e}"


# ── Yahoo Finance ────────────────────────────────────────────────
def _fetch_yahoo(ticker: str, start, end) -> tuple:
    """
    Yahoo Finance v8 チャートAPI（APIキー不要）
    """
    import time as _time
    start_ts = int(_time.mktime(start.timetuple()))
    end_ts   = int(_time.mktime(end.timetuple()))
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=adjsplits"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return None, f"Yahoo Finance: HTTP {r.status_code}"
        body = r.json()
        result = body.get("chart", {}).get("result")
        if not result:
            err = body.get("chart", {}).get("error", {})
            return None, f"Yahoo Finance: {err.get('description', 'データなし')}"
        timestamps = result[0].get("timestamp", [])
        indicators  = result[0].get("indicators", {})
        # adjclose があれば優先、なければ close を使用
        adj = indicators.get("adjclose", [{}])[0].get("adjclose")
        cls = indicators.get("quote", [{}])[0].get("close")
        prices = adj if adj else cls
        if not prices or not timestamps:
            return None, "Yahoo Finance: 価格データが空"
        dates = pd.to_datetime(timestamps, unit="s")
        series = pd.Series(prices, index=dates, dtype=float).dropna()
        series.index = series.index.tz_localize(None)
        return series, None
    except Exception as e:
        return None, f"Yahoo Finance: {type(e).__name__}: {e}"


# ── 統合フォールバック関数 ────────────────────────────────────────
def fetch_price_with_fallback(ticker: str, api_key: str, start, end) -> tuple:
    """
    Tiingo → Stooq → Yahoo Finance の順で株価を取得。
    戻り値: (日次リターン Series or None, 使用ソース名, エラー詳細dict)
    """
    errors = {}

    # 1) Tiingo
    price, err = _fetch_tiingo(ticker, api_key, start, end)
    if price is not None:
        return _returns_from_series(price), "Tiingo", errors
    errors["Tiingo"] = err

    # 2) Stooq
    price, err = _fetch_stooq(ticker, start, end)
    if price is not None:
        return _returns_from_series(price), "Stooq", errors
    errors["Stooq"] = err

    # 3) Yahoo Finance
    price, err = _fetch_yahoo(ticker, start, end)
    if price is not None:
        return _returns_from_series(price), "Yahoo Finance", errors
    errors["Yahoo Finance"] = err

    return None, "全ソース失敗", errors


def get_market_data(api_key, start, end):
    """
    市場ベンチマーク (QQQ) の日次リターンを取得。
    戻り値: (Series or None, summary_str)
    """
    returns, source, errors = fetch_price_with_fallback("QQQ", api_key, start, end)
    if returns is not None and len(returns) > 0:
        return returns, f"✅ {source} から取得（{len(returns)}日分）"
    # 全失敗時のエラーメッセージ生成
    detail = " | ".join(f"{k}: {v}" for k, v in errors.items())
    return None, f"全ソース失敗: {detail}"


def analyze_ticker(ticker, market_returns, api_key, start, end):
    """個別銘柄分析。Tiingo → Stooq → Yahoo Finance でフォールバック"""
    returns, source, _ = fetch_price_with_fallback(ticker, api_key, start, end)
    if returns is None:
        return None
    try:
        common = returns.index.intersection(market_returns.index)
        if len(common) < 60:
            return None
        x = returns.loc[common].values
        y = market_returns.loc[common].values
        annual_return = np.mean(x) * 252
        annual_risk   = np.std(x) * np.sqrt(252)
        if annual_risk == 0:
            return None
        beta     = np.cov(x, y)[0, 1] / np.var(y)
        alpha    = annual_return - (0.01 + beta * (np.mean(y) * 252 - 0.01))
        sharpe   = (annual_return - 0.01) / annual_risk
        residual = np.std(x - beta * y) * np.sqrt(252)
        return {
            "銘柄": ticker,
            "データソース": source,
            "年間リターン": annual_return,
            "年間リスク": annual_risk,
            "シャープレシオ": sharpe,
            "ベータ": beta,
            "アルファ": alpha,
            "レジデュアルリスク": residual,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_cik(ticker: str):
    try:
        r = requests.get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&output=atom",
            headers={"User-Agent": "NasdaqDashboard/2.1 research@example.com"},
            timeout=15,
        )
        match = re.search(r"/cgi-bin/browse-edgar\?action=getcompany&CIK=(\d+)", r.text)
        if match:
            return match.group(1).zfill(10)
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def get_edgar_filings(ticker: str, form_type: str = "10-K", count: int = 4) -> list:
    cik = get_cik(ticker)
    if not cik:
        return []
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms      = filings.get("form", [])
        dates      = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        results = []
        for i, f in enumerate(forms):
            if f == form_type:
                acc = accessions[i].replace("-", "")
                results.append({
                    "form": f,
                    "date": dates[i],
                    "accession": accessions[i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/",
                })
                if len(results) >= count:
                    break
        return results
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_xbrl_financials(ticker: str) -> pd.DataFrame:
    cik = get_cik(ticker)
    if not cik:
        return pd.DataFrame()
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        facts = r.json().get("facts", {})

        def extract_series(concept_path: str, label: str) -> pd.DataFrame:
            ns, concept = concept_path.split(":")
            units = facts.get(ns, {}).get(concept, {}).get("units", {})
            entries = []
            for unit_key in ["USD", "USD/shares", "shares"]:
                entries = units.get(unit_key, [])
                if entries:
                    break
            q_entries = [e for e in entries if e.get("form") in ("10-Q", "10-K") and e.get("fp")]
            q_entries.sort(key=lambda x: x.get("end", ""), reverse=True)
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

        revenue_df = extract_series("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "売上高")
        if revenue_df.empty:
            revenue_df = extract_series("us-gaap:Revenues", "売上高")
        net_income_df = extract_series("us-gaap:NetIncomeLoss", "純利益")
        eps_df        = extract_series("us-gaap:EarningsPerShareBasic", "EPS（基本）")

        dfs = [d for d in [revenue_df, net_income_df, eps_df] if not d.empty]
        if not dfs:
            return pd.DataFrame()
        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.join(d, how="outer")
        merged = merged.sort_index(ascending=False).head(5).reset_index()
        merged["銘柄"] = ticker
        return merged
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_earnings_data(ticker: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=calendarEvents,earningsHistory,defaultKeyStatistics"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json().get("quoteSummary", {}).get("result", [{}])[0]
        earnings_dates = data.get("calendarEvents", {}).get("earnings", {}).get("earningsDate", [])
        next_date = earnings_dates[0].get("fmt", "N/A") if earnings_dates else "N/A"
        history = data.get("earningsHistory", {}).get("history", [])
        eps_history = []
        for h in history[-4:]:
            eps_history.append({
                "期間": h.get("quarter", {}).get("fmt", ""),
                "EPS実績": h.get("epsActual", {}).get("raw"),
                "EPS予想": h.get("epsEstimate", {}).get("raw"),
                "サプライズ%": h.get("surprisePercent", {}).get("raw"),
            })
        stats = data.get("defaultKeyStatistics", {})
        return {
            "次回決算日": next_date,
            "EPS履歴": eps_history,
            "PER(予想)": stats.get("forwardPE", {}).get("raw"),
            "PBR": stats.get("priceToBook", {}).get("raw"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=900)
def get_news_headlines(ticker: str, max_items: int = 10) -> list:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        titles = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text.strip())
            if len(titles) >= max_items:
                break
        return titles
    except Exception:
        return []


# ================================================================
# メール通知
# ================================================================

def send_email_notification(subject: str, body: str, attachment_bytes=None, filename="report.txt"):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, NOTIFY_EMAIL]):
        return False, "SMTP設定が不完全です"
    try:
        msg = MIMEMultipart()
        msg["From"]    = SMTP_USER
        msg["To"]      = NOTIFY_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        if attachment_bytes:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            msg.attach(part)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, NOTIFY_EMAIL, msg.as_string())
        return True, "送信成功"
    except Exception as e:
        return False, f"送信失敗: {e}"


# ================================================================
# レポートテキスト生成
# ================================================================

def build_text_report(ticker, earnings_info, xbrl_df, headlines, sentiment, ai_analysis) -> str:
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
    lines.append("\n【最新ニュース】")
    lines += [f"  - {h}" for h in headlines]
    lines += [f"\n【センチメント】\n{sentiment}", f"\n【AI決算分析】\n{ai_analysis}", "=" * 60]
    return "\n".join(lines)


# ================================================================
# グラフ描画
# ================================================================

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
        periods = xbrl_df.loc[vals.index, "期間末"] if "期間末" in xbrl_df.columns else range(len(vals))
        ax.bar(range(len(vals)), vals / 1e9, color="#4472C4")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(list(periods), rotation=30, ha="right", fontsize=8)
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


# ================================================================
# UI
# ================================================================
st.title("🚀 NASDAQ-100 プロ仕様 銘柄分析ダッシュボード")

# AI バックエンド状態表示
with st.sidebar:
    st.header("📊 設定")
    years = st.sidebar.slider("分析対象期間（年）", 1, 10, 3)
    st.divider()

    st.subheader("🤖 AI バックエンド")
    st.caption("優先順位: Gemini → Groq → OpenRouter")
    for b in _ai_backends:
        st.markdown(f"- {b}")
    st.divider()

    st.subheader("📋 決算分析設定")
    selected_tickers = st.multiselect(
        "分析する銘柄",
        nasdaq100_tickers,
        default=["AAPL", "MSFT", "NVDA"],
        max_selections=10,
    )
    filing_type = st.selectbox("SEC フィリング種別", ["10-K", "10-Q"])
    st.divider()

    st.subheader("📧 メール通知")
    enable_email = st.checkbox("決算レポートをメールで通知", value=False)
    st.divider()
    st.caption("v2.1 | SEC EDGAR / Yahoo Finance / Gemini / Groq / OpenRouter")

end_date   = datetime.today().date()
start_date = end_date - timedelta(days=years * 365)

tab1, tab2, tab3 = st.tabs(["📊 パフォーマンス分析", "📋 決算分析（EDGAR・EPS）", "📰 ニュース翻訳・センチメント"])

# ─── Tab 1: パフォーマンス分析 ───────────────────────────────
with tab1:
    st.subheader("📊 NASDAQ-100 パフォーマンス分析")

    # ── APIキー簡易チェック ──
    if TIINGO_API_KEY:
        st.caption(f"🔑 Tiingo APIキー: `{TIINGO_API_KEY[:6]}...{TIINGO_API_KEY[-4:]}` (設定済み)")
    else:
        st.error("TIINGO_API_KEY が未設定です")

    if st.button("▶ 全銘柄パフォーマンス分析を実行", type="primary"):
        with st.spinner("市場データ取得中 (QQQ)... Tiingo → Stooq → Yahoo Finance の順で試みます"):
            market_returns, src_msg = get_market_data(TIINGO_API_KEY, start_date, end_date)

        if market_returns is None:
            st.error("❌ 全データソースで市場データ(QQQ)の取得に失敗しました")
            st.error(f"詳細: {src_msg}")
            st.info(
                "**対処法:**\n"
                "- Tiingo: Secrets の `TIINGO_API_KEY` を確認\n"
                "- Stooq: stooq.com にブラウザからアクセスして疎通確認\n"
                "- Yahoo Finance: ネットワーク環境を確認\n"
                "- しばらく待ってから再試行してください"
            )
            # 診断ツール
            with st.expander("🔍 各ソース診断"):
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.markdown("**Tiingo**")
                    try:
                        test_url = f"https://api.tiingo.com/tiingo/daily/QQQ/prices?startDate=2024-01-01&endDate=2024-01-05&token={TIINGO_API_KEY}"
                        resp = requests.get(test_url, timeout=10, headers={"Content-Type": "application/json"})
                        st.code(f"HTTP {resp.status_code}\n{resp.text[:200]}")
                    except Exception as ex:
                        st.code(f"Error: {ex}")
                with col_d2:
                    st.markdown("**Stooq**")
                    try:
                        resp = requests.get("https://stooq.com/q/d/l/?s=qqq.us&d1=20240101&d2=20240105&i=d", timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                        st.code(f"HTTP {resp.status_code}\n{resp.text[:200]}")
                    except Exception as ex:
                        st.code(f"Error: {ex}")
                with col_d3:
                    st.markdown("**Yahoo Finance**")
                    try:
                        resp = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/QQQ?period1=1704067200&period2=1704326400&interval=1d", timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                        st.code(f"HTTP {resp.status_code}\n{resp.text[:200]}")
                    except Exception as ex:
                        st.code(f"Error: {ex}")
        else:
            st.success(f"市場データ取得: {src_msg}")
            with st.spinner("全銘柄データ取得・分析中（各銘柄もフォールバック対応）..."):
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(analyze_ticker, t, market_returns, TIINGO_API_KEY, start_date, end_date)
                               for t in nasdaq100_tickers]
                    for f in futures:
                        r = f.result()
                        if r:
                            results.append(r)

                df = pd.DataFrame(results)
                if df.empty:
                    st.error("有効な分析結果がありません")
                else:
                    # データソース別の集計を表示
                    if "データソース" in df.columns:
                        src_counts = df["データソース"].value_counts()
                        src_info = " / ".join(f"{s}: {c}銘柄" for s, c in src_counts.items())
                        st.caption(f"📡 データ取得元: {src_info}")
                    display_cols = [c for c in df.columns if c != "データソース"]
                    st.dataframe(
                        df[display_cols].style
                        .format("{:.2%}", subset=["年間リターン", "年間リスク", "アルファ", "レジデュアルリスク"])
                        .format("{:.2f}", subset=["シャープレシオ", "ベータ"]),
                        use_container_width=True,
                    )

                    top = df.sort_values("シャープレシオ", ascending=False).head(20)
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
                    ax1.bar(top["銘柄"], top["年間リターン"] * 100, label="リターン(%)", color="#4472C4")
                    ax1.bar(top["銘柄"], top["年間リスク"] * 100, alpha=0.3, label="リスク(%)", color="#ED7D31")
                    ax1.legend(); ax1.set_title("年間リターン / リスク（上位20銘柄）")
                    ax2.bar(top["銘柄"], top["シャープレシオ"], color="#70AD47")
                    ax2.set_title("シャープレシオ（上位20銘柄）")
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.subheader("🤖 AIによる企業解説（上位3社）")
                    for ticker in top["銘柄"].head(3):
                        with st.expander(f"💡 {ticker} の概要"):
                            text, backend = ai_company_summary(ticker)
                            st.write(text)
                            st.caption(f"AI: {backend}")

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="Analysis")
                    st.download_button(
                        "📥 Excelでダウンロード",
                        data=output.getvalue(),
                        file_name=f"Nasdaq100_Analysis_{end_date}.xlsx",
                    )

# ─── Tab 2: 決算分析 ─────────────────────────────────────────
with tab2:
    st.subheader("📋 決算分析（SEC EDGAR + Yahoo Finance）")
    if not selected_tickers:
        st.info("サイドバーで銘柄を選択してください")
    elif st.button("▶ 決算データを取得・分析", type="primary"):
        all_reports = {}

        for ticker in selected_tickers:
            st.markdown(f"---\n### 🔍 {ticker}")
            cols = st.columns([1, 1])

            with st.spinner(f"{ticker} データ取得中..."):
                xbrl_df = get_xbrl_financials(ticker)

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

                earnings_info = get_earnings_data(ticker)
                with cols[1]:
                    st.markdown("**📅 決算発表日・EPS サプライズ**")
                    st.markdown(f"- **次回決算日**: {earnings_info.get('次回決算日', 'N/A')}")
                    per_val = earnings_info.get('PER(予想)')
                    pbr_val = earnings_info.get('PBR')
                    st.markdown(f"- **予想PER**: {f'{per_val:.1f}x' if per_val else 'N/A'}")
                    st.markdown(f"- **PBR**: {f'{pbr_val:.2f}x' if pbr_val else 'N/A'}")
                    eps_history = earnings_info.get("EPS履歴", [])
                    if eps_history:
                        eps_df_show = pd.DataFrame(eps_history)
                        st.dataframe(eps_df_show, use_container_width=True)
                        fig_eps = plot_eps_surprise(eps_history, ticker)
                        if fig_eps:
                            st.pyplot(fig_eps)
                        beats = sum(1 for e in eps_history if e.get("サプライズ%") and e["サプライズ%"] > 0)
                        total = len([e for e in eps_history if e.get("サプライズ%") is not None])
                        if total:
                            beat_pct = beats / total * 100
                            color = "🟢" if beat_pct >= 75 else "🟡" if beat_pct >= 50 else "🔴"
                            st.metric("直近4Q EPS Beat率", f"{color} {beats}/{total} ({beat_pct:.0f}%)")

                # AI 決算分析
                st.markdown("**🤖 AI 決算総合分析（日本語）**")
                ai_result, ai_backend = ai_earnings_analysis(ticker, xbrl_df, eps_history)
                st.info(ai_result)
                st.caption(f"AI バックエンド: {ai_backend}")

                # SEC フィリング
                filings = get_edgar_filings(ticker, filing_type, count=4)
                if filings:
                    st.markdown(f"**📄 SEC {filing_type} フィリング（直近4件）**")
                    fdf = pd.DataFrame(filings)[["form", "date", "url"]]
                    fdf.columns = ["種別", "提出日", "EDGARリンク"]
                    st.dataframe(fdf, use_container_width=True, column_config={
                        "EDGARリンク": st.column_config.LinkColumn("EDGARリンク")
                    })

                headlines = get_news_headlines(ticker, 5)
                sentiment, _ = ai_sentiment(headlines, ticker)
                report_txt = build_text_report(ticker, earnings_info, xbrl_df, headlines, sentiment, ai_result)
                all_reports[ticker] = report_txt

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
                ok, msg = send_email_notification(
                    f"【NASDAQ決算速報】{', '.join(selected_tickers)} | {end_date}",
                    combined,
                    attachment_bytes=combined.encode("utf-8"),
                    filename=f"EarningsReport_{end_date}.txt",
                )
                st.success(f"✅ {msg}") if ok else st.warning(f"⚠️ {msg}")

# ─── Tab 3: ニュース翻訳・センチメント ───────────────────────
with tab3:
    st.subheader("📰 最新ニュース 自動日本語翻訳・センチメント分析")
    news_ticker = st.selectbox("銘柄を選択", nasdaq100_tickers, index=nasdaq100_tickers.index("AAPL"))
    max_news = st.slider("取得ニュース件数", 3, 20, 8)

    if st.button("▶ ニュースを取得・翻訳", type="primary"):
        with st.spinner("ニュース取得中..."):
            headlines = get_news_headlines(news_ticker, max_news)

        if not headlines:
            st.warning("ニュースが取得できませんでした")
        else:
            st.markdown(f"### 📰 {news_ticker} 最新ニュース（英語原文）")
            for i, h in enumerate(headlines, 1):
                st.markdown(f"{i}. {h}")

            st.divider()
            st.markdown("### 🎯 センチメント分析（AI）")
            with st.spinner("センチメント分析中..."):
                sentiment_result, sent_backend = ai_sentiment(headlines, news_ticker)
            st.info(sentiment_result)
            st.caption(f"AI バックエンド: {sent_backend}")

            st.divider()
            st.markdown("### 🌐 AI 日本語翻訳・要約")
            with st.spinner("翻訳・要約中..."):
                translation, trans_backend = ai_translate_and_summarize(
                    "\n".join(headlines),
                    context=f"{news_ticker}に関するニュース見出し一覧"
                )
            st.success(translation)
            st.caption(f"AI バックエンド: {trans_backend}")

            report_news = (
                f"# {news_ticker} ニュースレポート ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
                f"## 英語原文\n" + "\n".join(f"- {h}" for h in headlines)
                + f"\n\n## センチメント分析\n{sentiment_result}"
                + f"\n\n## 日本語翻訳・要約\n{translation}"
            )
            st.download_button(
                "📥 ニュースレポートを保存",
                data=report_news.encode("utf-8"),
                file_name=f"{news_ticker}_news_{end_date}.txt",
                mime="text/plain",
            )
