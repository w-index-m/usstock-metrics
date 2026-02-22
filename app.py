# -*- coding: utf-8 -*-
"""
NASDAQ-100 プロ仕様 銘柄分析ダッシュボード
─────────────────────────────────────────────────────
追加機能（v2.0）
  ① SEC EDGAR から 10-Q / 10-K 取得 & XBRL 財務数値抽出
  ② 決算発表日・概要を Yahoo Finance から取得
  ③ 英文ニュース/決算サマリーを自動日本語翻訳（Gemini）
  ④ 過去4四半期比較グラフ
  ⑤ EPS サプライズ / コンセンサス比較
  ⑥ センチメント分析（ニュース見出し）
  ⑦ テキスト保存 & メール通知（SMTP）
  ⑧ 複数銘柄一括処理
─────────────────────────────────────────────────────
必要 Secrets (Streamlit Secrets):
  TIINGO_API_KEY    : Tiingo API
  GEMINI_API_KEY    : Google Gemini API（任意、翻訳・要約に使用）
  SMTP_HOST         : メール送信ホスト（任意）
  SMTP_PORT         : ポート番号（例: 587）
  SMTP_USER         : 送信元メールアドレス
  SMTP_PASS         : SMTPパスワード
  NOTIFY_EMAIL      : 通知先メールアドレス
"""

# ===============================
# import
# ===============================
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import concurrent.futures
import google.generativeai as genai
import os
import io
import json
import re
import time
import smtplib
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import xml.etree.ElementTree as ET

# ===============================
# フォント設定
# ===============================
FONT_CANDIDATES = ["fonts/NotoSansJP-VariableFont_wght.ttf"]
for font in FONT_CANDIDATES:
    if os.path.exists(font):
        fm.fontManager.addfont(font)
        prop = fm.FontProperties(fname=font)
        plt.rcParams["font.family"] = prop.get_name()
        break
else:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["IPAexGothic", "TakaoPGothic", "DejaVu Sans"]

# ===============================
# ページ設定
# ===============================
st.set_page_config(page_title="NASDAQ-100 プロ分析ツール", layout="wide", page_icon="🚀")

# ===============================
# Secrets
# ===============================
TIINGO_API_KEY = st.secrets.get("TIINGO_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
SMTP_HOST      = st.secrets.get("SMTP_HOST", "")
SMTP_PORT      = int(st.secrets.get("SMTP_PORT", 587))
SMTP_USER      = st.secrets.get("SMTP_USER", "")
SMTP_PASS      = st.secrets.get("SMTP_PASS", "")
NOTIFY_EMAIL   = st.secrets.get("NOTIFY_EMAIL", "")

if not TIINGO_API_KEY:
    st.error("⚠️ TIINGO_API_KEY が設定されていません（Secretsを確認してください）")
    st.stop()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ===============================
# NASDAQ-100 ティッカー
# ===============================
nasdaq100_tickers = sorted(set([
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","AVGO","COST","PEP",
    "ADBE","NFLX","AMD","INTC","CSCO","QCOM","TXN","AMAT","HON","INTU",
    "SBUX","BKNG","ADP","REGN","VRTX","LRCX","MU","PANW","KLAC","CDNS",
    "ADI","NXPI","FTNT","WDAY","SNPS","MELI","CRWD","CTAS","VRT","CEG",
    "MRVL","ORLY","MNST","ASML","CSX","ROST","PAYX","AEP","KDP","ODFL",
    "DXCM","TEAM","FAST","BIIB","CHTR","EA","EXC","FANG","GEHC","GFS",
    "IDXX","KHC","LCID","LULU","MCHP","MDLZ","MRNA","PCAR","PDD","PYPL",
    "RIVN","SIRI","TTD","WBD","XEL","ZS","ANSS","AZN","BKR","CDW",
    "CPRT","DLTR","EBAY","ENPH","ILMN","JD","KHC","MAR","MSTR","OKTA",
    "ON","SPLK","TTWO","UAL","WBA","ZM","APP","ARM","SMCI","PLTR"
    "ADI","NXPI","FTNT","WDAY","SNPS","MELI","CRWD","CTAS","VRT","CEG",
    "ANET","SMCI","PLTR","APH","GLW",
]))

# ===============================
# SEC EDGAR ヘッダー（User-Agent 必須）
# ===============================
EDGAR_HEADERS = {
    "User-Agent": "NasdaqDashboard/2.0 research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# ===============================
# ① 株価・リターン取得
# ===============================
def get_market_data(api_key, start, end):
    url = "https://api.tiingo.com/tiingo/daily/QQQ/prices"
    params = {"startDate": str(start), "endDate": str(end), "resampleFreq": "daily", "token": api_key}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df["adjClose"].pct_change().dropna()


def analyze_ticker(ticker, market_returns, api_key, start, end):
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": str(start), "endDate": str(end), "resampleFreq": "daily", "token": api_key}
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        df = pd.DataFrame(r.json())
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        returns = df["adjClose"].pct_change().dropna()
        common = returns.index.intersection(market_returns.index)
        if len(common) < 60:
            return None
        x = returns.loc[common].values
        y = market_returns.loc[common].values
        annual_return = np.mean(x) * 252
        annual_risk   = np.std(x) * np.sqrt(252)
        beta    = np.cov(x, y)[0, 1] / np.var(y)
        alpha   = annual_return - (0.01 + beta * (np.mean(y) * 252 - 0.01))
        sharpe  = (annual_return - 0.01) / annual_risk
        residual = np.std(x - beta * y) * np.sqrt(252)
        return {"銘柄": ticker, "年間リターン": annual_return, "年間リスク": annual_risk,
                "シャープレシオ": sharpe, "ベータ": beta, "アルファ": alpha, "レジデュアルリスク": residual}
    except Exception:
        return None

# ===============================
# ② SEC EDGAR : CIK 取得
# ===============================
@st.cache_data(ttl=3600)
def get_cik(ticker: str) -> str | None:
    """ティッカー → CIK（10桁ゼロ埋め）"""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "NasdaqDashboard/2.0 research@example.com"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None

# ===============================
# ③ SEC EDGAR : 最新 10-K / 10-Q フィリング取得
# ===============================
@st.cache_data(ttl=3600)
def get_edgar_filings(ticker: str, form_type: str = "10-K", count: int = 4) -> list[dict]:
    """SEC EDGAR から直近フィリング一覧を返す"""
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
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
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

# ===============================
# ④ XBRL から財務数値抽出
# ===============================
@st.cache_data(ttl=3600)
def get_xbrl_financials(ticker: str) -> pd.DataFrame:
    """
    SEC EDGAR Company Facts API を使い、
    売上・純利益・EPS の過去4四半期を取得
    """
    cik = get_cik(ticker)
    if not cik:
        return pd.DataFrame()
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame()
        facts = r.json().get("facts", {})

        def extract_series(concept_path: str, label: str) -> pd.Series:
            """us-gaap:概念名 → 四半期値 Series"""
            ns, concept = concept_path.split(":")
            units = facts.get(ns, {}).get(concept, {}).get("units", {})
            # 通貨単位（USD）または純粋数値（shares）
            for unit_key in ["USD", "USD/shares", "shares"]:
                entries = units.get(unit_key, [])
                if entries:
                    break
            # 四半期（form=10-Q or 10-K）のみ、直近8件
            q_entries = [e for e in entries if e.get("form") in ("10-Q", "10-K") and e.get("fp")]
            q_entries.sort(key=lambda x: x.get("end", ""), reverse=True)
            rows = []
            seen = set()
            for e in q_entries:
                key = e.get("end")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"期間末": key, label: e.get("val")})
                if len(rows) >= 5:
                    break
            return pd.DataFrame(rows).set_index("期間末") if rows else pd.DataFrame()

        # 主要指標
        revenue_df = extract_series("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "売上高")
        if revenue_df.empty:
            revenue_df = extract_series("us-gaap:Revenues", "売上高")
        net_income_df = extract_series("us-gaap:NetIncomeLoss", "純利益")
        eps_df        = extract_series("us-gaap:EarningsPerShareBasic", "EPS（基本）")

        dfs = [df for df in [revenue_df, net_income_df, eps_df] if not df.empty]
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

# ===============================
# ⑤ Yahoo Finance : 決算発表日 & EPS サプライズ
# ===============================
@st.cache_data(ttl=1800)
def get_earnings_data(ticker: str) -> dict:
    """
    Yahoo Finance の非公式エンドポイントから
    決算発表日、EPS実績、EPS予想を取得
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=calendarEvents,earningsHistory,defaultKeyStatistics"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json().get("quoteSummary", {}).get("result", [{}])[0]

        # 次回決算日
        earnings_dates = (
            data.get("calendarEvents", {}).get("earnings", {}).get("earningsDate", [])
        )
        next_date = earnings_dates[0].get("fmt", "N/A") if earnings_dates else "N/A"

        # 過去EPS
        history = data.get("earningsHistory", {}).get("history", [])
        eps_history = []
        for h in history[-4:]:
            eps_history.append({
                "期間": h.get("quarter", {}).get("fmt", ""),
                "EPS実績": h.get("epsActual", {}).get("raw"),
                "EPS予想": h.get("epsEstimate", {}).get("raw"),
                "サプライズ%": h.get("surprisePercent", {}).get("raw"),
            })

        # 統計
        stats = data.get("defaultKeyStatistics", {})
        return {
            "次回決算日": next_date,
            "EPS履歴": eps_history,
            "PER(予想)": stats.get("forwardPE", {}).get("raw"),
            "PBR": stats.get("priceToBook", {}).get("raw"),
        }
    except Exception:
        return {}

# ===============================
# ⑥ Yahoo Finance : 最新ニュース取得
# ===============================
@st.cache_data(ttl=900)
def get_news_headlines(ticker: str, max_items: int = 10) -> list[str]:
    """Yahoo Finance RSS から最新ニュース見出しを取得"""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        ns = {"media": "http://search.yahoo.com/mrss/"}
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

# ===============================
# ⑦ Gemini : 翻訳 & サマリー & センチメント
# ===============================
def gemini_translate_and_summarize(text: str, context: str = "") -> str:
    """英文テキストを日本語に翻訳し要約する"""
    if not GEMINI_API_KEY:
        return "（Gemini API Key 未設定）"
    prompt = f"""
以下の英文テキストを日本語に翻訳し、投資家向けに300文字以内で要約してください。
{f'背景情報: {context}' if context else ''}

英文:
{text}

出力形式:
【翻訳要約】
（ここに日本語で記載）
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini Error: {e}"


def gemini_sentiment(headlines: list[str], ticker: str) -> str:
    """ニュース見出しリストのセンチメント分析"""
    if not GEMINI_API_KEY or not headlines:
        return "（データなし）"
    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""
以下は米国株「{ticker}」に関する最新ニュース見出しです。
センチメント（強気/弱気/中立）を判定し、理由を日本語200文字以内で説明してください。

ニュース見出し:
{headlines_text}

出力形式:
センチメント: [強気 / 弱気 / 中立]
理由: （日本語で説明）
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini Error: {e}"


def gemini_earnings_analysis(ticker: str, xbrl_df: pd.DataFrame, eps_data: list[dict]) -> str:
    """決算データを総合分析して日本語レポートを生成"""
    if not GEMINI_API_KEY:
        return "（Gemini API Key 未設定）"
    xbrl_str = xbrl_df.to_string(index=False) if not xbrl_df.empty else "データなし"
    eps_str = json.dumps(eps_data, ensure_ascii=False, indent=2) if eps_data else "データなし"
    prompt = f"""
米国株「{ticker}」の決算データを分析し、投資家向けの日本語レポートを作成してください。

## XBRL財務データ（過去4四半期）
{xbrl_str}

## EPS履歴・サプライズ
{eps_str}

以下の観点で400文字以内で分析してください：
1. 売上・利益トレンド
2. EPSサプライズの傾向（beat/miss）
3. 投資判断上の注目点
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini Error: {e}"


def gemini_company_summary(ticker: str) -> str:
    if not GEMINI_API_KEY:
        return "（Gemini API Key 未設定）"
    prompt = f"""
米国株「{ticker}」について、
・事業内容 ・強み ・主な収益源
を投資家向けに日本語300文字以内で要約してください。
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Gemini Error: {e}"

# ===============================
# ⑧ メール通知
# ===============================
def send_email_notification(subject: str, body: str, attachment_bytes: bytes | None = None, filename: str = "report.xlsx"):
    """決算速報をメールで通知する"""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, NOTIFY_EMAIL]):
        return False, "SMTP設定が不完全です（Secretsを確認）"
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
        return True, "メール送信成功"
    except Exception as e:
        return False, f"メール送信失敗: {e}"

# ===============================
# ⑨ テキストレポート保存
# ===============================
def build_text_report(ticker: str, earnings_info: dict, xbrl_df: pd.DataFrame,
                      headlines: list[str], sentiment: str, ai_analysis: str) -> str:
    lines = [
        f"=" * 60,
        f"  {ticker} 決算レポート（生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        f"=" * 60,
        f"\n【次回決算日】 {earnings_info.get('次回決算日', 'N/A')}",
        f"【PER(予想)】  {earnings_info.get('PER(予想)', 'N/A')}",
        f"【PBR】       {earnings_info.get('PBR', 'N/A')}",
        "\n【EPS履歴・サプライズ】",
    ]
    for e in earnings_info.get("EPS履歴", []):
        lines.append(f"  {e['期間']} | 実績: {e['EPS実績']} | 予想: {e['EPS予想']} | サプライズ: {e.get('サプライズ%', 'N/A')}%")

    if not xbrl_df.empty:
        lines.append("\n【XBRL財務データ（SEC EDGAR）】")
        lines.append(xbrl_df.to_string(index=False))

    lines.append("\n【最新ニュース見出し】")
    for h in headlines:
        lines.append(f"  - {h}")

    lines.append(f"\n【センチメント分析】\n{sentiment}")
    lines.append(f"\n【AI決算分析（日本語）】\n{ai_analysis}")
    lines.append("=" * 60)
    return "\n".join(lines)

# ===============================
# グラフ描画ヘルパー
# ===============================
def plot_xbrl_quarterly(xbrl_df: pd.DataFrame, ticker: str):
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


def plot_eps_surprise(eps_history: list[dict], ticker: str):
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

# ===============================
# UI
# ===============================
st.title("🚀 NASDAQ-100 プロ仕様 銘柄分析ダッシュボード")

# タブ構成
tab1, tab2, tab3 = st.tabs(["📊 パフォーマンス分析", "📋 決算分析（EDGAR・EPS）", "📰 ニュース翻訳・センチメント"])

# ─── サイドバー ───────────────────────────────────────────────
with st.sidebar:
    st.header("📊 設定")
    years = st.slider("分析対象期間（年）", 1, 10, 3)
    st.divider()
    st.subheader("📋 決算分析設定")
    selected_tickers = st.multiselect(
        "分析する銘柄を選択",
        nasdaq100_tickers,
        default=["AAPL", "MSFT", "NVDA"],
        max_selections=10,
    )
    filing_type = st.selectbox("SEC フィリング種別", ["10-K", "10-Q"])
    st.divider()
    st.subheader("📧 メール通知")
    enable_email = st.checkbox("決算レポートをメールで通知", value=False)
    st.divider()
    st.caption("v2.0 | SEC EDGAR / Yahoo Finance / Gemini AI")

end_date   = datetime.today().date()
start_date = end_date - timedelta(days=years * 365)

# ─────────────────────────────────────────────────────────────
# Tab 1: パフォーマンス分析（既存機能）
# ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 NASDAQ-100 パフォーマンス分析")
    if st.button("▶ 全銘柄パフォーマンス分析を実行", type="primary"):
        with st.spinner("データ取得・計算中..."):
            market_returns = get_market_data(TIINGO_API_KEY, start_date, end_date)
            if market_returns is None:
                st.error("市場データの取得に失敗しました")
            else:
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
                    # 順位列を追加（値上がり順位・シャープレシオ順位）
                    df.insert(0, "シャープレシオ順位", df["シャープレシオ"].rank(ascending=False, method="min").astype(int))
                    df.insert(0, "値上がり順位", df["年間リターン"].rank(ascending=False, method="min").astype(int))
                    # デフォルト表示は値上がり順位でソート
                    df = df.sort_values("値上がり順位").reset_index(drop=True)
                    st.dataframe(
                        df.style
                        .format("{:.2%}", subset=["年間リターン", "年間リスク", "アルファ", "レジデュアルリスク"])
                        .format("{:.2f}", subset=["シャープレシオ", "ベータ"])
                        .format("{:d}", subset=["値上がり順位", "シャープレシオ順位"]),
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
                    for t in top["銘柄"].head(3):
                        with st.expander(f"💡 {t} の概要"):
                            st.write(gemini_company_summary(t))

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="Analysis")
                    st.download_button("📥 Excelでダウンロード", data=output.getvalue(),
                                       file_name=f"Nasdaq100_Analysis_{end_date}.xlsx")

# ─────────────────────────────────────────────────────────────
# Tab 2: 決算分析
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📋 決算分析（SEC EDGAR + Yahoo Finance）")
    if not selected_tickers:
        st.info("サイドバーで銘柄を選択してください")
    elif st.button("▶ 決算データを取得・分析", type="primary"):
        all_reports = {}

        for ticker in selected_tickers:
            st.markdown(f"---\n### 🔍 {ticker}")
            cols = st.columns([1, 1])

            with st.spinner(f"{ticker} のデータ取得中..."):
                # ── XBRL 財務データ ──
                xbrl_df = get_xbrl_financials(ticker)
                with cols[0]:
                    st.markdown("**📈 XBRL 財務データ（SEC EDGAR）**")
                    if not xbrl_df.empty:
                        # 表示用に単位変換（Bドル）
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

                # ── EPS / 決算日 ──
                earnings_info = get_earnings_data(ticker)
                with cols[1]:
                    st.markdown("**📅 決算発表日・EPS サプライズ**")
                    st.markdown(f"- **次回決算日**: {earnings_info.get('次回決算日', 'N/A')}")
                    st.markdown(f"- **予想PER**: {earnings_info.get('PER(予想)', 'N/A')}")
                    st.markdown(f"- **PBR**: {earnings_info.get('PBR', 'N/A')}")

                    eps_history = earnings_info.get("EPS履歴", [])
                    if eps_history:
                        eps_df = pd.DataFrame(eps_history)
                        st.dataframe(eps_df, use_container_width=True)
                        fig_eps = plot_eps_surprise(eps_history, ticker)
                        if fig_eps:
                            st.pyplot(fig_eps)

                        # コンセンサス比較サマリー
                        beats = sum(1 for e in eps_history if e.get("サプライズ%") and e["サプライズ%"] > 0)
                        total = len([e for e in eps_history if e.get("サプライズ%") is not None])
                        if total:
                            st.metric("直近4四半期 EPS Beat率", f"{beats}/{total} ({beats/total*100:.0f}%)")

                # ── AI 決算分析 ──
                st.markdown("**🤖 AI 決算総合分析（日本語）**")
                ai_analysis = gemini_earnings_analysis(ticker, xbrl_df, eps_history)
                st.info(ai_analysis)

                # ── SEC フィリング一覧 ──
                filings = get_edgar_filings(ticker, filing_type, count=4)
                if filings:
                    st.markdown(f"**📄 SEC {filing_type} フィリング（直近4件）**")
                    fdf = pd.DataFrame(filings)[["form", "date", "url"]]
                    fdf.columns = ["種別", "提出日", "EDGARリンク"]
                    st.dataframe(fdf, use_container_width=True, column_config={
                        "EDGARリンク": st.column_config.LinkColumn("EDGARリンク")
                    })

                # ── レポートテキスト蓄積 ──
                headlines = get_news_headlines(ticker, 5)
                sentiment = gemini_sentiment(headlines, ticker)
                report_txt = build_text_report(ticker, earnings_info, xbrl_df, headlines, sentiment, ai_analysis)
                all_reports[ticker] = report_txt

        # ── 一括ダウンロード（テキスト）──
        if all_reports:
            st.divider()
            combined = "\n\n".join(all_reports.values())
            st.download_button(
                "📥 全銘柄レポートをテキストでダウンロード",
                data=combined.encode("utf-8"),
                file_name=f"EarningsReport_{end_date}.txt",
                mime="text/plain",
            )

            # ── メール通知 ──
            if enable_email:
                subject = f"【NASDAQ決算速報】{', '.join(selected_tickers)} | {end_date}"
                ok, msg = send_email_notification(
                    subject, combined,
                    attachment_bytes=combined.encode("utf-8"),
                    filename=f"EarningsReport_{end_date}.txt",
                )
                if ok:
                    st.success(f"✅ {msg} → {NOTIFY_EMAIL}")
                else:
                    st.warning(f"⚠️ {msg}")

# ─────────────────────────────────────────────────────────────
# Tab 3: ニュース翻訳・センチメント
# ─────────────────────────────────────────────────────────────
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

            # センチメント
            st.markdown("---")
            st.markdown("### 🎯 センチメント分析（AI）")
            with st.spinner("センチメント分析中..."):
                sentiment_result = gemini_sentiment(headlines, news_ticker)
            st.info(sentiment_result)

            # 一括翻訳
            st.markdown("---")
            st.markdown("### 🌐 AI 日本語翻訳・要約")
            all_headlines_text = "\n".join(headlines)
            with st.spinner("翻訳・要約中..."):
                translation = gemini_translate_and_summarize(
                    all_headlines_text,
                    context=f"{news_ticker}に関するニュース見出し一覧"
                )
            st.success(translation)

            # テキスト保存
            report_news = f"# {news_ticker} ニュースレポート ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
            report_news += "## 英語原文\n" + "\n".join(f"- {h}" for h in headlines)
            report_news += f"\n\n## センチメント分析\n{sentiment_result}"
            report_news += f"\n\n## 日本語翻訳・要約\n{translation}"
            st.download_button(
                "📥 ニュースレポートを保存",
                data=report_news.encode("utf-8"),
                file_name=f"{news_ticker}_news_{end_date}.txt",
                mime="text/plain",
            )
