Skip to content
w-index-m
usstock-metrics
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
1
 (1)
Insights
Settings
Files
Go to file
t
font
README.md
Secrets
app.py
requirements.txt
usstock-metrics
/
app.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

2
Line wrap mode

No wrap
Editing app.py file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
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

Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.

説明
