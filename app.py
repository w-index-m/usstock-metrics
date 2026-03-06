"""
us_stock_agent.py
─────────────────────────────────────────────────────────────────
🤖 AI銘柄分析エージェント（米国株対応）
  ユーザーが銘柄ティッカーを入力すると、AIエージェントが自律的に
  ① 価格取得 ② ニュース収集 ③ 決算確認 ④ テーマ分析
  を実行し、統合レポートを生成する。

依存パッケージ:
  pip install streamlit yfinance pandas numpy matplotlib plotly
              google-generativeai requests
─────────────────────────────────────────────────────────────────
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import xml.etree.ElementTree as ET
import re
import time
from datetime import datetime, timedelta
from io import BytesIO
import google.generativeai as genai

# ─── ページ設定 ──────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="🤖 AI銘柄分析エージェント",
    page_icon="🤖",
)

# ─── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* チャットバブル */
.agent-step {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    color: #e0e0e0;
}
.agent-step .step-icon { color: #00d4ff; font-size: 1rem; margin-right: 8px; }
.report-card {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #00d4ff33;
}
.metric-card {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    border: 1px solid #ffffff22;
}
.metric-value { font-size: 1.5rem; font-weight: bold; color: #00d4ff; }
.metric-label { font-size: 0.75rem; color: #aaaaaa; margin-top: 4px; }
.positive { color: #00ff88 !important; }
.negative { color: #ff4444 !important; }
.neutral  { color: #ffaa00 !important; }
.thinking-box {
    background: #0d1117;
    border: 1px dashed #00d4ff55;
    border-radius: 8px;
    padding: 10px 14px;
    color: #888;
    font-size: 0.8rem;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ─── タイトル ─────────────────────────────────────────────────────
st.title("🤖 AI銘柄分析エージェント")
st.caption("銘柄ティッカーを入力すると、AIが自律的に価格・ニュース・決算・テーマを分析してレポートを生成します")

# ─── APIキー設定 ──────────────────────────────────────────────────
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    API_OK = True
except Exception:
    API_OK = False
    gemini_model = None

# ════════════════════════════════════════════════════════════════
# ツール定義（エージェントが呼び出す関数群）
# ════════════════════════════════════════════════════════════════

def tool_get_price_data(ticker: str, period_days: int = 365) -> dict:
    """価格データ・基本指標を取得"""
    try:
        end = datetime.today()
        start = end - timedelta(days=period_days + 30)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 20:
            return {"error": f"{ticker} の価格データを取得できませんでした"}

        close = df["Close"].dropna()
        current = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else current
        week_ago = float(close.iloc[-5]) if len(close) >= 5 else current
        month_ago = float(close.iloc[-20]) if len(close) >= 20 else current
        year_ago = float(close.iloc[0])

        high_52w = float(df["High"].max())
        low_52w  = float(df["Low"].min())
        avg_vol  = int(df["Volume"].mean())

        # 移動平均
        ma25  = float(close.rolling(25).mean().iloc[-1]) if len(close) >= 25 else None
        ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # シャープレシオ
        returns = close.pct_change().dropna()
        annual_ret = float(returns.mean() * 252)
        annual_vol = float(returns.std() * np.sqrt(252))
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs   = gain / (loss + 1e-10)
        rsi  = float(100 - 100 / (1 + rs.iloc[-1]))

        return {
            "ticker": ticker,
            "current_price": round(current, 2),
            "prev_close": round(prev_close, 2),
            "change_1d_pct": round((current - prev_close) / prev_close * 100, 2),
            "change_1w_pct": round((current - week_ago) / week_ago * 100, 2),
            "change_1m_pct": round((current - month_ago) / month_ago * 100, 2),
            "change_1y_pct": round((current - year_ago) / year_ago * 100, 2),
            "high_52w": round(high_52w, 2),
            "low_52w":  round(low_52w, 2),
            "from_high_pct": round((current - high_52w) / high_52w * 100, 2),
            "from_low_pct":  round((current - low_52w) / low_52w * 100, 2),
            "avg_volume": avg_vol,
            "ma25": round(ma25, 2) if ma25 else None,
            "ma50": round(ma50, 2) if ma50 else None,
            "ma200": round(ma200, 2) if ma200 else None,
            "sharpe_ratio": round(sharpe, 3),
            "annual_return_pct": round(annual_ret * 100, 2),
            "annual_volatility_pct": round(annual_vol * 100, 2),
            "rsi_14": round(rsi, 1),
            "data_points": len(close),
            "ohlcv_df": df,  # チャート用
        }
    except Exception as e:
        return {"error": str(e)}


def tool_get_fundamentals(ticker: str) -> dict:
    """ファンダメンタルズ・決算データを取得"""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        result = {
            "company_name": info.get("longName") or info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "employees": info.get("fullTimeEmployees"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "revenue_ttm": info.get("totalRevenue"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "free_cashflow": info.get("freeCashflow"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
            "analyst_target": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "business_summary": (info.get("longBusinessSummary") or "")[:500],
        }

        # 直近決算
        try:
            qfinancials = tk.quarterly_financials
            if qfinancials is not None and not qfinancials.empty:
                result["recent_revenue"] = []
                result["recent_net_income"] = []
                for col in qfinancials.columns[:4]:
                    rev = qfinancials.loc["Total Revenue", col] if "Total Revenue" in qfinancials.index else None
                    ni  = qfinancials.loc["Net Income", col] if "Net Income" in qfinancials.index else None
                    result["recent_revenue"].append({
                        "quarter": str(col)[:10],
                        "value": int(rev) if rev and not pd.isna(rev) else None
                    })
                    result["recent_net_income"].append({
                        "quarter": str(col)[:10],
                        "value": int(ni) if ni and not pd.isna(ni) else None
                    })
        except Exception:
            pass

        return result
    except Exception as e:
        return {"error": str(e)}


def tool_get_news(ticker: str, company_name: str = "") -> dict:
    """Yahoo Finance RSSからニュースを取得"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    items = []

    # Yahoo Finance RSS
    try:
        query = ticker if not company_name else f"{company_name} stock"
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:10]:
                title   = item.findtext("title", "").strip()
                link    = item.findtext("link", "").strip()
                pubdate = item.findtext("pubDate", "").strip()
                desc    = re.sub(r"<[^>]+>", "", item.findtext("description", ""))[:150].strip()
                if title:
                    items.append({
                        "source": "Yahoo Finance",
                        "title": title,
                        "link": link,
                        "date": pubdate[:16] if pubdate else "",
                        "summary": desc,
                    })
    except Exception:
        pass

    # Reuters Business RSS
    try:
        r2 = requests.get("https://feeds.reuters.com/reuters/businessNews", headers=headers, timeout=10)
        if r2.status_code == 200:
            root2 = ET.fromstring(r2.content)
            ticker_upper = ticker.upper()
            name_short = (company_name.split()[0] if company_name else "").upper()
            for item in root2.findall(".//item")[:20]:
                title = item.findtext("title", "").strip()
                if ticker_upper in title.upper() or (name_short and name_short in title.upper()):
                    link    = item.findtext("link", "").strip()
                    pubdate = item.findtext("pubDate", "")[:16]
                    items.append({
                        "source": "Reuters",
                        "title": title,
                        "link": link,
                        "date": pubdate,
                        "summary": "",
                    })
    except Exception:
        pass

    return {
        "ticker": ticker,
        "news_count": len(items),
        "news_items": items[:12],
    }


def tool_sector_theme_analysis(ticker: str, sector: str, industry: str) -> dict:
    """セクター・テーマ文脈を分析（類似銘柄との比較）"""
    SECTOR_MAP = {
        "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
        "Healthcare":  ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
        "Financials":  ["JPM", "BAC", "WFC", "GS", "MS"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
        "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Industrials": ["BA", "HON", "UPS", "CAT", "GE"],
        "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST"],
        "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
        "Basic Materials": ["LIN", "APD", "ECL", "SHW", "FCX"],
    }

    peers = SECTOR_MAP.get(sector, ["SPY"])[:4]
    peers = [p for p in peers if p != ticker][:3]

    peer_data = {}
    for peer in peers:
        try:
            df = yf.download(peer, period="3mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                ret = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
                peer_data[peer] = round(float(ret), 2)
        except Exception:
            pass

    # ベンチマーク（S&P500）
    try:
        spy = yf.download("SPY", period="3mo", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.droplevel(1)
        spy_ret = float((spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0] * 100)
    except Exception:
        spy_ret = None

    return {
        "sector": sector,
        "industry": industry,
        "peer_tickers": peers,
        "peer_3m_returns": peer_data,
        "spy_3m_return": round(spy_ret, 2) if spy_ret else None,
    }


# ════════════════════════════════════════════════════════════════
# AIエージェントコア
# ════════════════════════════════════════════════════════════════

def run_agent(ticker: str, status_container, log_container) -> dict:
    """
    AIエージェントが自律的にツールを呼び出し、
    最終レポートデータを返す
    """
    ticker = ticker.upper().strip()
    results = {}
    logs = []

    def log(icon, msg, detail=""):
        logs.append({"icon": icon, "msg": msg, "detail": detail})
        html = "".join(
            f'<div class="agent-step"><span class="step-icon">{l["icon"]}</span>{l["msg"]}'
            f'{"<br><span style=\'color:#888;font-size:0.78rem\'>" + l["detail"] + "</span>" if l["detail"] else ""}'
            f'</div>'
            for l in logs
        )
        log_container.markdown(html, unsafe_allow_html=True)

    # ── Step 1: 価格データ取得 ───────────────────────────────────
    status_container.info("🔍 **Step 1/4** 価格データを取得中...")
    log("📡", "ツール呼び出し: `tool_get_price_data`", f"ticker={ticker}, period=365日")
    time.sleep(0.3)

    price_data = tool_get_price_data(ticker, 365)
    if "error" in price_data:
        log("❌", f"価格取得エラー: {price_data['error']}")
        return {"error": price_data["error"]}

    log("✅", f"価格取得完了",
        f"現在値: ${price_data['current_price']:,.2f} | "
        f"1日: {price_data['change_1d_pct']:+.1f}% | "
        f"RSI: {price_data['rsi_14']}")
    results["price"] = price_data

    # ── Step 2: ファンダメンタルズ・決算取得 ─────────────────────
    status_container.info("🔍 **Step 2/4** 決算・財務データを確認中...")
    log("📊", "ツール呼び出し: `tool_get_fundamentals`", f"ticker={ticker}")
    time.sleep(0.3)

    fundamentals = tool_get_fundamentals(ticker)
    if "error" in fundamentals:
        log("⚠️", f"ファンダメンタルズ一部取得失敗（続行）")
        fundamentals = {"company_name": ticker, "sector": "N/A", "industry": "N/A"}
    else:
        company_name = fundamentals.get("company_name", ticker)
        pe = fundamentals.get("pe_ratio")
        margin = fundamentals.get("profit_margin")
        log("✅", f"財務データ取得完了",
            f"{company_name} | セクター: {fundamentals.get('sector','N/A')} | "
            f"PER: {f'{pe:.1f}x' if pe else 'N/A'} | "
            f"純利益率: {f'{margin*100:.1f}%' if margin else 'N/A'}")
    results["fundamentals"] = fundamentals

    # ── Step 3: ニュース収集 ─────────────────────────────────────
    status_container.info("🔍 **Step 3/4** ニュースを収集中...")
    log("📰", "ツール呼び出し: `tool_get_news`",
        f"ticker={ticker}, company={fundamentals.get('company_name','')} | ソース: Yahoo Finance, Reuters")
    time.sleep(0.3)

    news_data = tool_get_news(ticker, fundamentals.get("company_name", ""))
    log("✅", f"ニュース収集完了",
        f"{news_data['news_count']}件取得")
    results["news"] = news_data

    # ── Step 4: テーマ・セクター分析 ────────────────────────────
    status_container.info("🔍 **Step 4/4** セクター・テーマ分析中...")
    log("🔬", "ツール呼び出し: `tool_sector_theme_analysis`",
        f"sector={fundamentals.get('sector','N/A')}, industry={fundamentals.get('industry','N/A')}")
    time.sleep(0.3)

    theme_data = tool_sector_theme_analysis(
        ticker,
        fundamentals.get("sector", ""),
        fundamentals.get("industry", ""),
    )
    peer_summary = ", ".join(
        f"{k}: {v:+.1f}%" for k, v in (theme_data.get("peer_3m_returns") or {}).items()
    )
    spy_ret_str = f"{theme_data.get('spy_3m_return',0):+.1f}%" if theme_data.get('spy_3m_return') else 'N/A'
    log("✅", "セクター・テーマ分析完了",
        f"同セクター比較: {peer_summary or 'データなし'} | SPY 3M: {spy_ret_str}")
    results["theme"] = theme_data

    # ── Step 5: AIが統合分析レポートを生成 ──────────────────────
    status_container.info("🤖 **AIが統合分析レポートを生成中...**")
    log("🤖", "AIエージェントが全データを統合してレポート生成中...", "Gemini 2.5 Pro を使用")

    ai_report = generate_ai_report(results)
    results["ai_report"] = ai_report
    log("🎉", "レポート生成完了！")
    status_container.success("✅ 分析完了！")

    return results


def generate_ai_report(data: dict) -> str:
    """Geminiが全データを統合して投資レポートを生成"""
    if not API_OK or gemini_model is None:
        return _fallback_report(data)

    price = data.get("price", {})
    fund  = data.get("fundamentals", {})
    news  = data.get("news", {})
    theme = data.get("theme", {})

    news_headlines = "\n".join(
        f"- [{n['source']}] {n['title']}" for n in (news.get("news_items") or [])[:8]
    )

    peer_str = "\n".join(
        f"  {k}: {v:+.2f}%" for k, v in (theme.get("peer_3m_returns") or {}).items()
    )

    prompt = f"""
あなたはウォール街のトップアナリストです。
以下のデータをもとに、{fund.get('company_name', price.get('ticker',''))}({price.get('ticker','')})の
機関投資家向け総合分析レポートを作成してください。

## 価格データ
- 現在値: ${price.get('current_price', 'N/A')}
- 1日変化: {price.get('change_1d_pct', 'N/A')}%
- 1週変化: {price.get('change_1w_pct', 'N/A')}%
- 1ヶ月変化: {price.get('change_1m_pct', 'N/A')}%
- 1年変化: {price.get('change_1y_pct', 'N/A')}%
- 52週高値: ${price.get('high_52w', 'N/A')} ({price.get('from_high_pct', 'N/A')}% from high)
- RSI(14): {price.get('rsi_14', 'N/A')}
- シャープレシオ: {price.get('sharpe_ratio', 'N/A')}
- MA25: ${price.get('ma25', 'N/A')}, MA50: ${price.get('ma50', 'N/A')}, MA200: ${price.get('ma200', 'N/A')}

## ファンダメンタルズ
- セクター: {fund.get('sector', 'N/A')} / {fund.get('industry', 'N/A')}
- 時価総額: {_fmt_large(fund.get('market_cap'))}
- PER: {_fmt_num(fund.get('pe_ratio'))}x / フォワードPER: {_fmt_num(fund.get('forward_pe'))}x
- PBR: {_fmt_num(fund.get('pb_ratio'))}x / PEG: {_fmt_num(fund.get('peg_ratio'))}
- 売上高（TTM）: {_fmt_large(fund.get('revenue_ttm'))}
- 粗利益率: {_fmt_pct(fund.get('gross_margin'))} / 営業利益率: {_fmt_pct(fund.get('operating_margin'))} / 純利益率: {_fmt_pct(fund.get('profit_margin'))}
- ROE: {_fmt_pct(fund.get('roe'))} / ROA: {_fmt_pct(fund.get('roa'))}
- 売上成長率: {_fmt_pct(fund.get('revenue_growth'))} / 利益成長率: {_fmt_pct(fund.get('earnings_growth'))}
- D/Eレシオ: {_fmt_num(fund.get('debt_to_equity'))}
- アナリスト目標株価: ${_fmt_num(fund.get('analyst_target'))} ({fund.get('num_analysts', 'N/A')}人)
- アナリスト推奨: {fund.get('recommendation', 'N/A')}
- ベータ: {_fmt_num(fund.get('beta'))}
- 配当利回り: {_fmt_pct(fund.get('dividend_yield'))}

## 最新ニュース（{news.get('news_count', 0)}件）
{news_headlines if news_headlines else "ニュースなし"}

## セクター・テーマ比較（3ヶ月）
同業他社:
{peer_str if peer_str else "データなし"}
SPY（S&P500）: {f"{theme.get('spy_3m_return'):+.2f}%" if theme.get('spy_3m_return') is not None else "N/A"}

---
以下の構成でレポートを作成してください（日本語）:

### 📌 エグゼクティブサマリー
（3〜4文で現状評価と投資判断の要点）

### 💰 株価・テクニカル分析
（トレンド、サポート・レジスタンス、RSI・移動平均の評価）

### 📊 ファンダメンタルズ評価
（バリュエーション・収益性・成長性の評価）

### 📰 ニュース・センチメント分析
（最新ニュースからの市場センチメント判定：強気/弱気/中立）

### 🌐 セクター・テーマポジション
（同業他社対比でのポジション評価）

### ⚠️ リスク要因
（主要リスク3点）

### 🎯 投資判断サマリー
強気 / やや強気 / 中立 / やや弱気 / 弱気 の5段階で判定し、
その根拠を簡潔に述べてください。
目標株価レンジ（3〜6ヶ月）も示してください。

※ 投資判断はあくまで参考情報であり、最終的な投資決定は自己責任でお願いします。
"""
    try:
        response = gemini_model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates") and response.candidates:
            text = response.candidates[0].content.parts[0].text
        return text or _fallback_report(data)
    except Exception as e:
        return f"_AIレポート生成エラー: {e}_\n\n" + _fallback_report(data)


def _fallback_report(data: dict) -> str:
    price = data.get("price", {})
    fund  = data.get("fundamentals", {})
    return f"""
### 📌 基本分析サマリー
**{fund.get('company_name', price.get('ticker',''))}** の基本データを取得しました。

- **現在値**: ${price.get('current_price', 'N/A'):,}
- **1日変化**: {price.get('change_1d_pct', 'N/A'):+.1f}%
- **RSI(14)**: {price.get('rsi_14', 'N/A')}
- **シャープレシオ**: {price.get('sharpe_ratio', 'N/A')}
- **セクター**: {fund.get('sector', 'N/A')}
- **PER**: {_fmt_num(fund.get('pe_ratio'))}x

※ Gemini APIキーが未設定のため詳細AIレポートは生成されませんでした。
"""


def _fmt_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}"

def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:.1f}%"

def _fmt_large(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


# ════════════════════════════════════════════════════════════════
# チャート生成
# ════════════════════════════════════════════════════════════════

def make_candlestick_chart(df: pd.DataFrame, ticker: str):
    """ローソク足チャート + 移動平均"""
    df = df.tail(120).copy()
    df.index = pd.to_datetime(df.index)

    fig = go.Figure()

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4444",
    ))

    # 移動平均
    for period, color in [(25, "#00d4ff"), (50, "#ffaa00"), (200, "#ff88ff")]:
        if len(df) >= period:
            ma = df["Close"].rolling(period).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=ma,
                name=f"MA{period}", line=dict(color=color, width=1.5),
                opacity=0.8,
            ))

    fig.update_layout(
        title=f"{ticker} 株価チャート（直近120日）",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#333", showgrid=True, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#333", showgrid=True, title="価格 (USD)"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#333"),
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def make_volume_chart(df: pd.DataFrame, ticker: str):
    """出来高チャート"""
    df = df.tail(60).copy()
    df.index = pd.to_datetime(df.index)
    colors = ["#00ff88" if c >= o else "#ff4444"
              for c, o in zip(df["Close"], df["Open"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, name="出来高",
    ))
    fig.update_layout(
        title=f"{ticker} 出来高（直近60日）",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333", title="出来高"),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ════════════════════════════════════════════════════════════════
# HTMLレポート生成
# ════════════════════════════════════════════════════════════════

def generate_html_report(data: dict, ticker: str) -> str:
    price = data.get("price", {})
    fund  = data.get("fundamentals", {})
    news  = data.get("news", {})
    theme = data.get("theme", {})
    ai_report = data.get("ai_report", "")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    change_1d = price.get("change_1d_pct", 0)
    color_1d  = "#00ff88" if change_1d >= 0 else "#ff4444"
    sign_1d   = "+" if change_1d >= 0 else ""

    news_html = ""
    for n in (news.get("news_items") or [])[:8]:
        src_color = "#00d4ff" if n["source"] == "Yahoo Finance" else "#ffaa00"
        news_html += f"""
        <div style="border-left:3px solid {src_color};padding:8px 12px;margin:6px 0;background:#1a1a2e;border-radius:4px">
            <div style="font-size:0.75rem;color:{src_color}">{n['source']} | {n.get('date','')}</div>
            <div style="margin-top:4px"><a href="{n.get('link','#')}" style="color:#e0e0e0;text-decoration:none" target="_blank">{n['title']}</a></div>
            {"<div style='font-size:0.8rem;color:#888;margin-top:3px'>" + n['summary'] + "</div>" if n.get('summary') else ""}
        </div>"""

    peer_html = ""
    for peer, ret in (theme.get("peer_3m_returns") or {}).items():
        c = "#00ff88" if ret >= 0 else "#ff4444"
        peer_html += f'<span style="background:#1a1a2e;padding:6px 12px;border-radius:20px;margin:4px;display:inline-block;border:1px solid {c}">{peer}: <span style="color:{c}">{ret:+.1f}%</span></span>'

    # AI レポートのMarkdown→HTML（簡易変換）
    ai_html = ai_report.replace("\n", "<br>")
    ai_html = re.sub(r"### (.+?)(<br>|$)", r"<h3 style='color:#00d4ff;margin-top:18px'>\1</h3>", ai_html)
    ai_html = re.sub(r"\*\*(.+?)\*\*", r"<strong style='color:#ffaa00'>\1</strong>", ai_html)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{fund.get('company_name', ticker)} ({ticker}) 分析レポート</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
  .header {{ background: linear-gradient(135deg,#1a1a2e,#16213e); border-radius:12px; padding:24px; margin-bottom:20px; border:1px solid #00d4ff33; }}
  .header h1 {{ font-size:1.8rem; color:#00d4ff; }}
  .header .meta {{ color:#888; font-size:0.85rem; margin-top:6px; }}
  .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }}
  .grid-2 {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:20px; }}
  .card {{ background:#1a1a2e; border-radius:8px; padding:16px; border:1px solid #ffffff11; }}
  .card h3 {{ color:#00d4ff; margin-bottom:12px; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; }}
  .metric {{ text-align:center; padding:12px; background:#0d1117; border-radius:6px; margin:4px; }}
  .metric .val {{ font-size:1.4rem; font-weight:bold; }}
  .metric .lbl {{ font-size:0.72rem; color:#888; margin-top:4px; }}
  .table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  .table th {{ background:#16213e; padding:8px; text-align:left; color:#888; }}
  .table td {{ padding:8px; border-bottom:1px solid #ffffff11; }}
  .section {{ background:linear-gradient(135deg,#0f3460,#16213e); border-radius:12px; padding:20px; margin-bottom:20px; border:1px solid #00d4ff22; }}
  .section h2 {{ color:#00d4ff; margin-bottom:16px; font-size:1.1rem; }}
  .badge {{ display:inline-block; padding:4px 10px; border-radius:20px; font-size:0.75rem; font-weight:bold; }}
  .footer {{ text-align:center; color:#555; font-size:0.75rem; margin-top:30px; padding:16px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>📈 {fund.get('company_name', ticker)} <span style="color:#888;font-size:1rem">({ticker})</span></h1>
    <div class="meta">
      {fund.get('sector','N/A')} | {fund.get('industry','N/A')} | {fund.get('country','US')}
      &nbsp;·&nbsp; 生成日時: {now}
    </div>
  </div>

  <div class="grid-3">
    <div class="card" style="text-align:center">
      <div style="font-size:2rem;font-weight:bold;color:#00d4ff">${price.get('current_price',0):,.2f}</div>
      <div style="font-size:1rem;color:{color_1d};font-weight:bold;margin-top:4px">{sign_1d}{change_1d:.2f}% (1日)</div>
      <div style="font-size:0.75rem;color:#888;margin-top:4px">現在値</div>
    </div>
    <div class="card">
      <table class="table">
        <tr><td>1週間</td><td style="color:{'#00ff88' if price.get('change_1w_pct',0)>=0 else '#ff4444'};font-weight:bold">{price.get('change_1w_pct',0):+.2f}%</td></tr>
        <tr><td>1ヶ月</td><td style="color:{'#00ff88' if price.get('change_1m_pct',0)>=0 else '#ff4444'};font-weight:bold">{price.get('change_1m_pct',0):+.2f}%</td></tr>
        <tr><td>1年</td><td style="color:{'#00ff88' if price.get('change_1y_pct',0)>=0 else '#ff4444'};font-weight:bold">{price.get('change_1y_pct',0):+.2f}%</td></tr>
      </table>
    </div>
    <div class="card">
      <table class="table">
        <tr><td>52週高値</td><td>${price.get('high_52w',0):,.2f} <span style="color:#ff4444;font-size:0.8rem">({price.get('from_high_pct',0):+.1f}%)</span></td></tr>
        <tr><td>52週安値</td><td>${price.get('low_52w',0):,.2f} <span style="color:#00ff88;font-size:0.8rem">({price.get('from_low_pct',0):+.1f}%)</span></td></tr>
        <tr><td>RSI(14)</td><td><span style="color:{'#ff4444' if price.get('rsi_14',50)>70 else '#00ff88' if price.get('rsi_14',50)<30 else '#ffaa00'}">{price.get('rsi_14',0):.1f}</span></td></tr>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>📊 バリュエーション</h3>
      <table class="table">
        <tr><td>時価総額</td><td>{_fmt_large(fund.get('market_cap'))}</td></tr>
        <tr><td>PER (実績)</td><td>{_fmt_num(fund.get('pe_ratio'))}x</td></tr>
        <tr><td>フォワードPER</td><td>{_fmt_num(fund.get('forward_pe'))}x</td></tr>
        <tr><td>PBR</td><td>{_fmt_num(fund.get('pb_ratio'))}x</td></tr>
        <tr><td>PSR</td><td>{_fmt_num(fund.get('ps_ratio'))}x</td></tr>
        <tr><td>EV/EBITDA</td><td>{_fmt_num(fund.get('ev_ebitda'))}x</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>📈 収益性・成長性</h3>
      <table class="table">
        <tr><td>売上高 (TTM)</td><td>{_fmt_large(fund.get('revenue_ttm'))}</td></tr>
        <tr><td>粗利益率</td><td>{_fmt_pct(fund.get('gross_margin'))}</td></tr>
        <tr><td>営業利益率</td><td>{_fmt_pct(fund.get('operating_margin'))}</td></tr>
        <tr><td>純利益率</td><td>{_fmt_pct(fund.get('profit_margin'))}</td></tr>
        <tr><td>ROE</td><td>{_fmt_pct(fund.get('roe'))}</td></tr>
        <tr><td>売上成長率</td><td>{_fmt_pct(fund.get('revenue_growth'))}</td></tr>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>📐 テクニカル指標</h3>
      <table class="table">
        <tr><td>MA25</td><td>${price.get('ma25') or 'N/A'}</td></tr>
        <tr><td>MA50</td><td>${price.get('ma50') or 'N/A'}</td></tr>
        <tr><td>MA200</td><td>${price.get('ma200') or 'N/A'}</td></tr>
        <tr><td>シャープレシオ</td><td>{price.get('sharpe_ratio', 'N/A')}</td></tr>
        <tr><td>年間リターン</td><td>{price.get('annual_return_pct', 'N/A')}%</td></tr>
        <tr><td>年間ボラティリティ</td><td>{price.get('annual_volatility_pct', 'N/A')}%</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>🏦 財務健全性</h3>
      <table class="table">
        <tr><td>フリーCF</td><td>{_fmt_large(fund.get('free_cashflow'))}</td></tr>
        <tr><td>D/Eレシオ</td><td>{_fmt_num(fund.get('debt_to_equity'))}</td></tr>
        <tr><td>流動比率</td><td>{_fmt_num(fund.get('current_ratio'))}</td></tr>
        <tr><td>ベータ</td><td>{_fmt_num(fund.get('beta'))}</td></tr>
        <tr><td>配当利回り</td><td>{_fmt_pct(fund.get('dividend_yield'))}</td></tr>
        <tr><td>アナリスト推奨</td><td>{fund.get('recommendation','N/A').upper() if fund.get('recommendation') else 'N/A'}</td></tr>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>🌐 セクター・ピア比較（3ヶ月）</h2>
    <div style="margin-bottom:8px;color:#888;font-size:0.85rem">セクター: {theme.get('sector','N/A')} | SPY (S&P500): <span style="color:{'#00ff88' if (theme.get('spy_3m_return') or 0)>=0 else '#ff4444'};font-weight:bold">{f"{theme.get('spy_3m_return'):+.1f}%" if theme.get('spy_3m_return') is not None else 'N/A'}</span></div>
    <div>{peer_html if peer_html else '<span style="color:#888">データなし</span>'}</div>
  </div>

  <div class="section">
    <h2>📰 最新ニュース ({news.get('news_count',0)}件)</h2>
    {news_html if news_html else '<p style="color:#888">ニュースを取得できませんでした</p>'}
  </div>

  <div class="section">
    <h2>🤖 AI統合分析レポート</h2>
    <div style="line-height:1.8;font-size:0.9rem">{ai_html}</div>
  </div>

  <div class="footer">
    ⚠️ 本レポートは情報提供のみを目的としており、投資助言ではありません。投資判断はご自身の責任で行ってください。<br>
    データソース: Yahoo Finance, Reuters | AI: Gemini 2.5 Pro | 生成: {now}
  </div>

</div>
</body>
</html>"""
    return html


# ════════════════════════════════════════════════════════════════
# メインUI
# ════════════════════════════════════════════════════════════════

# サイドバー
with st.sidebar:
    st.header("⚙️ エージェント設定")
    st.divider()
    st.subheader("💡 サンプル銘柄")
    sample_tickers = {
        "🍎 Apple":    "AAPL",
        "🤖 NVIDIA":   "NVDA",
        "⚡ Tesla":    "TSLA",
        "🔍 Google":   "GOOGL",
        "📦 Amazon":   "AMZN",
        "💳 Visa":     "V",
        "💊 J&J":      "JNJ",
        "🏦 JPMorgan": "JPM",
    }
    for label, sym in sample_tickers.items():
        if st.button(label, key=f"sample_{sym}", use_container_width=True):
            st.session_state["selected_ticker"] = sym

    st.divider()
    st.caption("データソース: Yahoo Finance, Reuters RSS")
    st.caption("AI: Google Gemini 2.5 Pro")

# セッション状態
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = ""

# ─── 入力エリア ───────────────────────────────────────────────
col_input, col_btn = st.columns([4, 1])
with col_input:
    ticker_input = st.text_input(
        "🔍 銘柄ティッカーを入力（例：AAPL, NVDA, TSLA）",
        value=st.session_state.get("selected_ticker", ""),
        placeholder="例: AAPL",
        key="ticker_input",
        label_visibility="visible",
    )
with col_btn:
    st.write("")
    run_btn = st.button("🚀 分析開始", type="primary", use_container_width=True)

# ─── 実行 ─────────────────────────────────────────────────────
if run_btn and ticker_input.strip():
    ticker = ticker_input.strip().upper()
    st.divider()

    st.markdown(f"## 🤖 `{ticker}` のエージェント分析を開始")

    # エージェントログ表示エリア
    col_log, col_result = st.columns([1, 2])

    with col_log:
        st.markdown("### 🔧 エージェントの思考プロセス")
        status_container = st.empty()
        log_container    = st.empty()

    with col_result:
        st.markdown("### 📊 リアルタイム結果")
        result_placeholder = st.empty()
        result_placeholder.markdown(
            '<div class="thinking-box">🤔 エージェントが分析中です...</div>',
            unsafe_allow_html=True
        )

    # エージェント実行
    data = run_agent(ticker, status_container, log_container)

    if "error" in data:
        st.error(f"❌ エラー: {data['error']}")
        st.stop()

    result_placeholder.empty()

    # ─── 結果表示 ──────────────────────────────────────────────
    st.divider()
    st.markdown(f"## 📈 {data['fundamentals'].get('company_name', ticker)} ({ticker}) 分析レポート")

    price = data["price"]
    fund  = data["fundamentals"]
    news  = data["news"]
    theme = data["theme"]

    # KPIカード
    change_1d  = price.get("change_1d_pct", 0)
    change_cls = "positive" if change_1d >= 0 else "negative"
    rsi        = price.get("rsi_14", 50)
    rsi_cls    = "negative" if rsi > 70 else "positive" if rsi < 30 else "neutral"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${price.get('current_price',0):,.2f}</div>
            <div class="metric-label">現在値</div>
            <div class="{change_cls}" style="font-size:0.85rem;margin-top:4px">{change_1d:+.2f}% (1日)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        pe = fund.get("pe_ratio")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f'{pe:.1f}x' if pe else 'N/A'}</div>
            <div class="metric-label">PER（実績）</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {rsi_cls}">{rsi:.1f}</div>
            <div class="metric-label">RSI (14日)</div>
            <div style="font-size:0.72rem;color:#888">{'過買' if rsi>70 else '過売' if rsi<30 else '中立'}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        sharpe = price.get("sharpe_ratio", 0)
        s_cls  = "positive" if sharpe > 1 else "negative" if sharpe < 0 else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {s_cls}">{sharpe:.2f}</div>
            <div class="metric-label">シャープレシオ</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        mc = fund.get("market_cap")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{_fmt_large(mc)}</div>
            <div class="metric-label">時価総額</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    # チャートタブ
    tab_chart, tab_fund, tab_news_tab, tab_theme, tab_report = st.tabs([
        "📈 チャート",
        "📊 財務指標",
        "📰 ニュース",
        "🌐 テーマ分析",
        "🤖 AIレポート",
    ])

    with tab_chart:
        ohlcv = price.get("ohlcv_df")
        if ohlcv is not None and not ohlcv.empty:
            st.plotly_chart(make_candlestick_chart(ohlcv, ticker), use_container_width=True)
            st.plotly_chart(make_volume_chart(ohlcv, ticker), use_container_width=True)
        else:
            st.warning("チャートデータがありません")

    with tab_fund:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("#### 📐 バリュエーション")
            metrics_val = {
                "時価総額": _fmt_large(fund.get("market_cap")),
                "EV": _fmt_large(fund.get("enterprise_value")),
                "PER (実績)": f"{_fmt_num(fund.get('pe_ratio'))}x",
                "フォワードPER": f"{_fmt_num(fund.get('forward_pe'))}x",
                "PBR": f"{_fmt_num(fund.get('pb_ratio'))}x",
                "PSR": f"{_fmt_num(fund.get('ps_ratio'))}x",
                "EV/EBITDA": f"{_fmt_num(fund.get('ev_ebitda'))}x",
                "PEG": _fmt_num(fund.get("peg_ratio")),
            }
            st.table(pd.DataFrame(list(metrics_val.items()), columns=["指標", "値"]))

        with col_f2:
            st.markdown("#### 📈 収益性・成長性")
            metrics_prof = {
                "売上高 (TTM)": _fmt_large(fund.get("revenue_ttm")),
                "粗利益率": _fmt_pct(fund.get("gross_margin")),
                "営業利益率": _fmt_pct(fund.get("operating_margin")),
                "純利益率": _fmt_pct(fund.get("profit_margin")),
                "ROE": _fmt_pct(fund.get("roe")),
                "ROA": _fmt_pct(fund.get("roa")),
                "売上成長率": _fmt_pct(fund.get("revenue_growth")),
                "利益成長率": _fmt_pct(fund.get("earnings_growth")),
            }
            st.table(pd.DataFrame(list(metrics_prof.items()), columns=["指標", "値"]))

        st.markdown("#### 🏦 財務健全性")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            metrics_health = {
                "フリーCF": _fmt_large(fund.get("free_cashflow")),
                "D/Eレシオ": _fmt_num(fund.get("debt_to_equity")),
                "流動比率": _fmt_num(fund.get("current_ratio")),
            }
            st.table(pd.DataFrame(list(metrics_health.items()), columns=["指標", "値"]))
        with col_h2:
            metrics_other = {
                "ベータ": _fmt_num(fund.get("beta")),
                "配当利回り": _fmt_pct(fund.get("dividend_yield")),
                "アナリスト推奨": (fund.get("recommendation") or "N/A").upper(),
                "目標株価（平均）": f"${_fmt_num(fund.get('analyst_target'))}",
            }
            st.table(pd.DataFrame(list(metrics_other.items()), columns=["指標", "値"]))

        # 事業概要
        if fund.get("business_summary"):
            st.markdown("#### 📝 事業概要")
            st.info(fund["business_summary"])

    with tab_news_tab:
        news_items = news.get("news_items", [])
        if not news_items:
            st.info("ニュースを取得できませんでした")
        else:
            src_colors = {"Yahoo Finance": "🟦", "Reuters": "🟫"}
            from collections import Counter
            src_cnt = Counter(n["source"] for n in news_items)
            cols_s = st.columns(len(src_cnt))
            for i, (src, cnt) in enumerate(src_cnt.items()):
                cols_s[i].metric(f"{src_colors.get(src,'⚪')} {src}", f"{cnt}件")

            st.divider()
            for item in news_items:
                icon = src_colors.get(item["source"], "⚪")
                with st.expander(f"{icon} {item['title'][:70]}{'...' if len(item['title'])>70 else ''}"):
                    st.markdown(f"**{item['title']}**")
                    if item.get("summary"):
                        st.caption(item["summary"])
                    c1, c2 = st.columns(2)
                    with c1:
                        if item.get("date"):
                            st.caption(f"📅 {item['date']}")
                    with c2:
                        if item.get("link"):
                            st.markdown(f"[🔗 記事を開く]({item['link']})")

    with tab_theme:
        st.markdown(f"#### 🌐 セクター: {theme.get('sector','N/A')} | 業種: {theme.get('industry','N/A')}")
        peer_returns = theme.get("peer_3m_returns", {})
        spy_ret      = theme.get("spy_3m_return")

        if peer_returns or spy_ret is not None:
            # 比較バーチャート
            all_tickers = list(peer_returns.keys()) + (["SPY"] if spy_ret is not None else [])
            all_returns = list(peer_returns.values()) + ([spy_ret] if spy_ret is not None else [])
            
            # 対象銘柄の3Mリターンを追加
            target_ret = price.get("change_1m_pct", 0)  # 1Mで代替
            all_tickers = [ticker] + all_tickers
            all_returns = [target_ret] + all_returns

            colors = ["#00d4ff" if t == ticker else ("#00ff88" if r >= 0 else "#ff4444")
                      for t, r in zip(all_tickers, all_returns)]

            fig_peer = go.Figure(go.Bar(
                x=all_tickers, y=all_returns,
                marker_color=colors,
                text=[f"{r:+.1f}%" for r in all_returns],
                textposition="outside",
            ))
            fig_peer.update_layout(
                title="同セクター 3ヶ月リターン比較",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e0e0e0"),
                yaxis=dict(gridcolor="#333", title="リターン (%)"),
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_peer, use_container_width=True)
        else:
            st.info("比較データなし")

    with tab_report:
        st.markdown(data.get("ai_report", "レポートが生成されませんでした"))

    # ─── HTMLダウンロード ────────────────────────────────────
    st.divider()
    html_content = generate_html_report(data, ticker)
    st.download_button(
        label="📥 HTMLレポートをダウンロード",
        data=html_content.encode("utf-8"),
        file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
    )

    # 履歴保存
    st.session_state.analysis_history.append({
        "ticker": ticker,
        "name": fund.get("company_name", ticker),
        "time": datetime.now().strftime("%H:%M"),
        "price": price.get("current_price"),
        "change": price.get("change_1d_pct"),
    })


# ─── 分析履歴 ─────────────────────────────────────────────────
if st.session_state.analysis_history:
    st.divider()
    st.markdown("#### 🕐 分析履歴（今セッション）")
    cols = st.columns(min(len(st.session_state.analysis_history), 5))
    for i, h in enumerate(reversed(st.session_state.analysis_history[-5:])):
        c = h.get("change", 0) or 0
        color = "#00ff88" if c >= 0 else "#ff4444"
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="metric-card" style="cursor:pointer">
                <div style="font-size:1rem;font-weight:bold;color:#00d4ff">{h['ticker']}</div>
                <div style="font-size:0.75rem;color:#888">{h['name'][:15]}</div>
                <div style="font-size:0.85rem;color:{color};margin-top:4px">${h['price']:,.2f} ({c:+.1f}%)</div>
                <div style="font-size:0.7rem;color:#555">{h['time']}</div>
            </div>""", unsafe_allow_html=True)

# ─── 初期画面 ─────────────────────────────────────────────────
elif not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#888">
        <div style="font-size:4rem">🤖</div>
        <div style="font-size:1.2rem;margin-top:16px;color:#e0e0e0">AI銘柄分析エージェント</div>
        <div style="margin-top:12px;font-size:0.9rem">
            銘柄ティッカーを入力するか、左のサンプルを選択してください<br>
            AIが自律的に以下の4つのツールを呼び出して分析します
        </div>
        <div style="display:flex;justify-content:center;gap:20px;margin-top:30px;flex-wrap:wrap">
            <div style="background:#1a1a2e;padding:16px 24px;border-radius:10px;border:1px solid #00d4ff33">
                <div style="color:#00d4ff;font-size:1.5rem">📡</div>
                <div style="margin-top:8px;font-size:0.85rem">価格取得</div>
                <div style="color:#888;font-size:0.75rem">yfinance API</div>
            </div>
            <div style="background:#1a1a2e;padding:16px 24px;border-radius:10px;border:1px solid #00d4ff33">
                <div style="color:#00d4ff;font-size:1.5rem">📰</div>
                <div style="margin-top:8px;font-size:0.85rem">ニュース収集</div>
                <div style="color:#888;font-size:0.75rem">Yahoo Finance / Reuters RSS</div>
            </div>
            <div style="background:#1a1a2e;padding:16px 24px;border-radius:10px;border:1px solid #00d4ff33">
                <div style="color:#00d4ff;font-size:1.5rem">📊</div>
                <div style="margin-top:8px;font-size:0.85rem">決算確認</div>
                <div style="color:#888;font-size:0.75rem">財務データ取得</div>
            </div>
            <div style="background:#1a1a2e;padding:16px 24px;border-radius:10px;border:1px solid #00d4ff33">
                <div style="color:#00d4ff;font-size:1.5rem">🌐</div>
                <div style="margin-top:8px;font-size:0.85rem">テーマ分析</div>
                <div style="color:#888;font-size:0.75rem">セクター・ピア比較</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
