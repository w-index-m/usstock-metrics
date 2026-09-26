# 🇺🇸 usstock-metrics

NASDAQ-100を中心とした米国株分析ダッシュボードです。Streamlitで構築されており、株価・指数・先物・マクロ指標・ニュース・SEC提出書類を取得し、テクニカル分析やAIによる日本語解説をブラウザ上で確認できます。

> **注意:** 本アプリは投資判断を補助するための分析ツールです。表示される情報は正確性・完全性・最新性を保証するものではなく、投資判断および損益は利用者自身の責任で行ってください。

## ✨ 主な機能

サイドバーから以下のページを切り替えて利用できます。

### 🖥️ マーケットスクリーン

- 米国主要指数（Dow Jones、NASDAQ、S&P 500、NASDAQ-100、Russell 2000）
- 先物・CFD（S&P、NASDAQ、Dow、Russell）
- VIX、米10年債利回り、金、WTI原油、ドル指数、為替
- Bitcoin、Ethereum、Solanaなどの暗号資産
- 大型テック株のリアルタイム株価
- 当日5分足チャート、または30日スパークライン
- 日本時間・米国東部時間のデータ時刻表示

### 📈 マーケット概況

- S&P 500 / NASDAQ-100の市場センチメント
- 株価、移動平均、RSI、MACD、5日モメンタム、VIX、先物/CFDを組み合わせたスコアリング
- 主要指数・先物・マクロ経済指標
- 市場ニュースとテクノロジーニュースの一覧・AI翻訳
- NASDAQ-100銘柄のリアルタイム株価

### 📊 パフォーマンス分析

- NASDAQ-100構成銘柄を対象にした期間別分析（1〜10年）
- QQQをベンチマークにした比較
- 年間リターン、年間リスク、シャープレシオ、ベータ、アルファ、レジデュアルリスク
- 銘柄ごとのデータソース表示
- 上位銘柄のリターン・リスク・シャープレシオのグラフ
- 上位銘柄のAI企業解説
- 分析結果のExcelダウンロード

株価データは次の順番で自動フォールバックします。

1. Tiingo
2. Stooq
3. Yahoo Finance

### 📉 テクニカル分析

指定した銘柄について、以下をPlotlyチャートで表示します。

- ローソク足（OHLC）
- SMA 20 / 50 / 200
- 出来高
- RSI（14）
- MACD（12, 26, 9）
- RSIの買われすぎ・売られすぎ判定
- MACDのゴールデンクロス・デッドクロス判定
- 現在値とSMA50の位置関係

### 📋 決算分析

- Yahoo Financeから決算日、EPS履歴、予想PER、PBRを取得
- SEC EDGARから10-K / 10-Qの提出書類を取得
- SEC XBRLから売上高、純利益、EPSを取得
- 売上高・純利益の推移グラフ
- EPS実績と予想の比較、EPS Beat率
- AIによる決算分析とニュースセンチメント分析
- テキストレポートのダウンロード
- SMTP設定時は決算レポートをメール通知

### 📰 ニュース翻訳

- 指定銘柄のYahoo Financeニュースを取得
- AIによるセンチメント判定（強気・弱気・中立）
- 英語見出しの日本語翻訳
- 投資家向けの日本語要約
- ニュースレポートの保存

### 🚀 モメンタムランキング

NASDAQ-100、Dow Jones 30、S&P 500を対象に、次の加重スコアでランキングします。

- 当日リターン: 50%
- 5日リターン: 30%
- 20日リターン: 20%

上昇モメンタム上位・下位銘柄とスコアの棒グラフを表示します。

### 📡 セクター比較

以下のバスケットを始値=100に正規化して比較します。

- 光通信: CIEN、COHR、LITE、VIAV、AAOI
- 半導体: NVDA、AMD、AVGO、INTC、QCOM
- 半導体ETF: SMH、SOXX
- NASDAQ-100および任意の追加銘柄

### 📊 センチメント推移

NASDAQ-100、S&P 500、Dow Jonesについて、RSI・MACD・VIX・モメンタム・SMA50から算出したセンチメントの時系列を表示します。

## 🛠️ 技術構成

- Python
- Streamlit
- pandas / NumPy
- Plotly / Matplotlib
- Yahoo Finance API
- Tiingo API（任意）
- Stooq（フォールバック）
- SEC EDGAR / SEC XBRL
- Google Gemini、Groq、OpenRouter（AI機能）
- RSSニュースフィード

## 🚀 セットアップ

### 1. リポジトリを取得

```bash
git clone https://github.com/w-index-m/usstock-metrics.git
cd usstock-metrics
```

### 2. 仮想環境を作成して依存関係をインストール

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Excelダウンロード機能を利用する場合は、`xlsxwriter`もインストールしてください。

```bash
pip install xlsxwriter
```

### 3. Streamlit Secretsを設定

ローカルでは `.streamlit/secrets.toml` を作成し、必要なAPIキーを設定します。

```toml
TIINGO_API_KEY = ""
GEMINI_API_KEY = ""
GROQ_API_KEY = ""
OPENROUTER_API_KEY = ""

SMTP_HOST = ""
SMTP_PORT = 587
SMTP_USER = ""
SMTP_PASS = ""
NOTIFY_EMAIL = ""
```

AI機能は設定済みのプロバイダーを利用し、次の順にフォールバックします。

1. Gemini
2. Groq
3. OpenRouter

`TIINGO_API_KEY`が未設定の場合は、Stooq、Yahoo Financeへ自動的に切り替わります。

### 4. アプリを起動

```bash
streamlit run app.py
```

ブラウザで表示されたStreamlitのURLを開いてください。

## 🔐 APIキー・Secretsに関する注意

- APIキーやSMTPパスワードをソースコード、README、Git履歴に記録しないでください。
- Streamlitでは `.streamlit/secrets.toml` またはデプロイ先のSecrets管理機能を使用してください。
- 既に公開リポジトリやGit履歴にキーを登録した場合は、直ちにキーを無効化・再発行してください。
- `Secrets` のようなキーを含むファイルは、Gitへコミットせず `.gitignore` に追加してください。

## 📁 ファイル構成

```text
usstock-metrics/
├── app.py            # Streamlitアプリ本体
├── requirements.txt  # Python依存パッケージ
├── font/             # 日本語表示用フォント
├── .streamlit/
│   └── secrets.toml  # ローカル用Secrets（Git管理対象外）
└── README.md         # プロジェクト説明
```

## 📡 外部データについて

本アプリは以下の外部サービスからデータを取得します。各サービスの仕様変更、レート制限、障害、利用規約の影響によりデータを取得できない場合があります。

- Yahoo Finance
- Tiingo
- Stooq
- SEC EDGAR / SEC XBRL
- Yahoo Finance RSS
- Wikipedia（S&P 500構成銘柄）
- Gemini / Groq / OpenRouter

## 📄 ライセンス

ライセンスは現在明示されていません。利用・再配布する場合は、リポジトリ管理者に確認してください。
