"""
AdCrux  ·  Marketing Mix Modeling Platform
==============================================
The problem we solve
--------------------
Google says Google generated your leads.
Meta says Meta generated your leads.
Both are wrong — because both have an incentive to overclaim.

AdCrux uses Marketing Mix Modeling to show you the real picture:
which channels are actually driving incremental results, which ones
are being overcredited by last-click, and where you should move budget.

Modules
-------
1. Overview      — The truth about your attribution (hero)
2. Attribution   — Deep-dive: platform claims vs model reality
3. Saturation    — Where each channel sits on its response curve
4. Optimizer     — Model-driven budget recommendation
5. Scenarios     — What-if budget simulator
6. Diagnostics   — Model quality, confidence, and data health
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from diagnostics import (
    compute_confidence_score,
    compute_collinearity,
    detect_tracking_breaks,
)
from model import MMMLeadGen, HyperParams
from optimizer import optimize_budget_advanced, optimize_budget_mvp
from tuning import auto_tune_params
from utils import (
    validate_input_df,
    channel_weekly_spend,
    has_lastclick,
    lastclick_by_channel,
    total_leads_scalar,
    validation_warnings,
    data_quality_report,
    weekly_outcome_series,
    MIN_WEEKS_FOR_CI,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdCrux",
    page_icon="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHJ4PSI4IiBmaWxsPSIjMEExNjI4Ii8+PGNpcmNsZSBjeD0iMTYiIGN5PSIxNiIgcj0iMTQiIGZpbGw9InVybCgjZmcpIi8+PGRlZnM+PHJhZGlhbEdyYWRpZW50IGlkPSJmZyIgY3g9IjUwJSIgY3k9IjUwJSIgcj0iNTAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMDBFNUJFIiBzdG9wLW9wYWNpdHk9Ii4yNSIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzAwRTVCRSIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD48L2RlZnM+PGxpbmUgeDE9IjYiIHkxPSI2IiB4Mj0iMjYiIHkyPSIyNiIgc3Ryb2tlPSIjMUEzMjUwIiBzdHJva2Utd2lkdGg9IjMuOCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PGxpbmUgeDE9IjI2IiB5MT0iNiIgeDI9IjYiIHkyPSIyNiIgc3Ryb2tlPSIjMUEzMjUwIiBzdHJva2Utd2lkdGg9IjMuOCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PGNpcmNsZSBjeD0iMTYiIGN5PSIxNiIgcj0iNS41IiBmaWxsPSIjMDBFNUJFIi8+PGNpcmNsZSBjeD0iMTYiIGN5PSIxNiIgcj0iOSIgZmlsbD0iIzAwRTVCRSIgZmlsbC1vcGFjaXR5PSIuMTgiLz48L3N2Zz4=",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar logo — official AdCrux wordmark (transparent variant) ────────────
st.logo("data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTY1LjQiIGhlaWdodD0iNTIiIHZpZXdCb3g9IjAgMCAxNjUuNCA1MiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxzdHlsZT5AZm9udC1mYWNle2ZvbnQtZmFtaWx5OidQQic7c3JjOnVybCgnZGF0YTpmb250L3R0ZjtiYXNlNjQsQUFFQUFBQU1BSUFBQXdCQVIxQlBVMFIyVEhVQUFBVm9BQUFBSUVkVFZVSWZTQ2RyQUFBRmlBQUFBREJQVXk4eVdzeGdZd0FBQXdnQUFBQmdZMjFoY0FGa0FaY0FBQU5vQUFBQVhHZHNlV1kzQ3JoRUFBQUF6QUFBQVpSb1pXRmtHbEVrY1FBQUFwQUFBQUEyYUdobFlRdjhBVjhBQUFMa0FBQUFKR2h0ZEhnUkVBREpBQUFDeUFBQUFCeHNiMk5oQVQ4Qm5BQUFBb0FBQUFBUWJXRjRjQUFkQVFjQUFBSmdBQUFBSUc1aGJXVWY5emJPQUFBRHhBQUFBWUp3YjNOMC83Z0FNZ0FBQlVnQUFBQWdBQUlBRUFBQUF0SUN2Z0FIQUFvQUFDVWhCeU1UTXhNakN3SUI4Lzc2S3JQK3h2NjFWbGRXZkh3Q3Z2MUNBUUFCQWY3L0FBQUJBQ0gvK3dMUEFzWUFHd0FBRWpZMk16SVdGeU1tSmlNaUJoVVVGak15TmpjekJnWWpJaVltTlNGYW9XWjlzaDY4RlUweFQySmlUekZORmJ3ZXNuMW1vVm9CeUtOYmhISXNMbTVjWEc0dUxIS0RXNkpvQUFJQUhQLzRBbWtDNUFBU0FCNEFBQkkyTmpNeUZoY1JNeEVqTlFZR0l5SW1KalVrSmlNaUJoVVVGak15TmpVY1FYQkZOMXNhcTZzWVdUdEZjRUVCb2tjek0wZEhNek5IQVc2Q1JpNG5BUVA5SEZBb01FZURWajlLU1VCQVMwcEFBQUVBUGdBQUFaZ0NOQUFNQUFBQU5qTVZJeUlHRlJFakVUTVZBUWRhTnk5QVFLdXJBZjgxdFRkRi92MENMbDBBQUFFQU9mLzZBbVVDTGdBVUFBQUJFU00xQmdZaklpWW1OUkV6RVJRV016STJOUkVDWmFzYVdUWkFZamFxT2pFeU9nSXUvZEpNSlMwNWJFa0JSdjdST0Q0K09BRXZBQUFCQUFVQUFBSklBaTRBQ3dBQUlTY0hJeE1ETXhjM013TVRBWWhyV3JtNHZjQnJXcm03d0p1YkFSMEJFWnFhL3VmKzZ3QUFBUUFBQUFjQWtBQU1BSFFBQmdBQkFBQUFBQUFBQUFBQUFBQUFBQU1BQWdBQUFBQUFHZ0JGQUhVQWpRQ3dBTW9BQVFBQUFBUUJCbmd1L29aZkR6ejFBQU1ENkFBQUFBRFlwS25JQUFBQUFOc1dOc245enYyaUNhMEVWQUFCQUFjQUFnQUFBQUFBQUFIMEFBQUM0UUFRQXZvQUlRS25BQndCckFBK0FxSUFPUUpNQUFVQUFRQUFCQnIrb2dCa0NaMzl6dmtZQ2EwQUFRQUFBQUFBQUFBQUFBQUFBQUFBQUFjQUJBTnBBcndBQlFBQUFvb0NXQUFBQUVzQ2lnSllBQUFCWGdBeUFVNEFBQUFBQ0FBQUFBQUFBQUFBQUFBQkFBQUFBQUFBQUFBQUFBQUFTVlJHVHdDZ0FFRUFlQVFhL3FJQVpBUnZBbk1BQUFBQkFBQUFBQUl1QXNFQUFBQWdBQVFBQUFBQ0FBQUFBd0FBQUJRQUF3QUJBQUFBRkFBRUFFZ0FBQUFPQUFnQUFnQUdBRUVBUXdCa0FISUFkUUI0Ly84QUFBQkJBRU1BWkFCeUFIVUFlUC8vLzhEL3YvK2YvNUwva1ArT0FBRUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBY0FXZ0FEQUFFRUNRQUFBS0lBQUFBREFBRUVDUUFCQUE0QW9nQURBQUVFQ1FBQ0FBZ0FzQUFEQUFFRUNRQURBRFlBdUFBREFBRUVDUUFFQUJnQTdnQURBQUVFQ1FBRkFBb0JCZ0FEQUFFRUNRQUdBQmdCRUFCREFHOEFjQUI1QUhJQWFRQm5BR2dBZEFBZ0FESUFNQUF5QURBQUlBQlVBR2dBWlFBZ0FGQUFid0J3QUhBQWFRQnVBSE1BSUFCUUFISUFid0JxQUdVQVl3QjBBQ0FBUVFCMUFIUUFhQUJ2QUhJQWN3QWdBQ2dBYUFCMEFIUUFjQUJ6QURvQUx3QXZBR2NBYVFCMEFHZ0FkUUJpQUM0QVl3QnZBRzBBTHdCcEFIUUFaZ0J2QUhVQWJnQmtBSElBZVFBdkFGQUFid0J3QUhBQWFRQnVBSE1BS1FCUUFHOEFjQUJ3QUdrQWJnQnpBRUlBYndCc0FHUUFTUUJVQUVZQVR3QTdBQ0FBVUFCdkFIQUFjQUJwQUc0QWN3QWdBRUlBYndCc0FHUUFPd0FnQURRQUxnQXdBREFBTkFCaUFEZ0FVQUJ2QUhBQWNBQnBBRzRBY3dBZ0FFSUFid0JzQUdRQU5BQXVBREFBTUFBMEFGQUFid0J3QUhBQWFRQnVBSE1BTFFCQ0FHOEFiQUJrQUFBQUF3QUFBQUFBQVArMUFESUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFCQUFBQUNnQWNBQjRBQVVSR1RGUUFDQUFFQUFBQUFQLy9BQUFBQUFBQUFBRUFBQUFLQUN3QUxnQURSRVpNVkFBVVpHVjJNZ0FlWkdWMllRQWVBQVFBQUFBQS8vOEFBQUFBQUFBQUFBQUEnKSBmb3JtYXQoJ3RydWV0eXBlJyk7Zm9udC13ZWlnaHQ6ODAwO308L3N0eWxlPgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJnX3NiIiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMDBFNUJFIiBzdG9wLW9wYWNpdHk9Ii4yNSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwMEU1QkUiIHN0b3Atb3BhY2l0eT0iMCIvPgogICAgPC9yYWRpYWxHcmFkaWVudD4KICA8L2RlZnM+CiAgPHRleHQgeD0iMCIgeT0iNDAiIGZvbnQtZmFtaWx5PSJQQixETSBTYW5zLC1hcHBsZS1zeXN0ZW0sc2Fucy1zZXJpZiIgZm9udC13ZWlnaHQ9IjgwMCIgZm9udC1zaXplPSI0MiIgbGV0dGVyLXNwYWNpbmc9Ii0xLjAiIGZpbGw9IiMzRDUyNzAiPkFkPC90ZXh0PgogIDx0ZXh0IHg9IjU1LjQiIHk9IjQwIiBmb250LWZhbWlseT0iUEIsRE0gU2FucywtYXBwbGUtc3lzdGVtLHNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSI4MDAiIGZvbnQtc2l6ZT0iNDIiIGxldHRlci1zcGFjaW5nPSItMS4wIiBmaWxsPSIjRjFGNUY5Ij5DcnU8L3RleHQ+CiAgPGNpcmNsZSBjeD0iMTQzLjAiIGN5PSIyOC4wIiByPSIyMC4wIiBmaWxsPSJ1cmwoI2dfc2IpIi8+CiAgPGxpbmUgeDE9IjEyOS4wIiB5MT0iMTQuMCIgeDI9IjE1Ny4wIiB5Mj0iNDIuMCIgc3Ryb2tlPSIjMUEzMjUwIiBzdHJva2Utd2lkdGg9IjYuNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CiAgPGxpbmUgeDE9IjE1Ny4wIiB5MT0iMTQuMCIgeDI9IjEyOS4wIiB5Mj0iNDIuMCIgc3Ryb2tlPSIjMUEzMjUwIiBzdHJva2Utd2lkdGg9IjYuNCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+CiAgPGNpcmNsZSBjeD0iMTQzLjAiIGN5PSIyOC4wIiByPSIxMC4wIiBmaWxsPSIjMDBFNUJFIiBmaWxsLW9wYWNpdHk9Ii4xOCIvPgogIDxjaXJjbGUgY3g9IjE0My4wIiBjeT0iMjguMCIgcj0iNi4wIiBmaWxsPSIjMDBFNUJFIi8+Cjwvc3ZnPg==")




# ─────────────────────────────────────────────────────────────────────────────
#  Design system
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#00E5BE",  # teal    — AdCrux primary
    "#8B5CF6",  # violet
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#38BDF8",  # blue
    "#EC4899",  # pink
    "#A3E635",  # lime
    "#06B6D4",  # cyan
]

BIAS_COLORS = {
    "Undervalued": "#10B981",
    "Overvalued":  "#EF4444",
    "Fair":        "#6366F1",
    "No last-click data": "#94A3B8",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ───────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #080810 !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ─────────────────────────────── */
.stApp { background-color: #080810 !important; }
.block-container { padding-top: 1.5rem !important; }

/* ── Top hero bar ───────────────────────────────── */
.hero-bar {
    background: linear-gradient(135deg, #0A1628 0%, #0E1A2E 100%);
    border: 1px solid #1C1C2E;
    padding: 1.25rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}
.hero-bar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,229,190,.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-bar h1 { color:#EEEEF5; font-size:1.5rem; font-weight:800; margin:0 0 .25rem; letter-spacing:-.02em; }
.hero-bar p  { color:#8B9DB0; font-size:.85rem; margin:0; }
.hero-bar strong { color:#00E5BE; }

/* ── Section headers ────────────────────────────── */
.sec-header {
    font-size:.75rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:#00E5BE;
    border-bottom:1px solid #1C1C2E;
    padding-bottom:.4rem; margin:1.8rem 0 1rem;
}

/* ── KPI cards ──────────────────────────────────── */
.kpi-card {
    background:#0E0E1A;
    border:1px solid #1C1C2E;
    border-radius:12px;
    padding:1rem 1.2rem;
    text-align:center;
    transition: border-color .15s, box-shadow .15s;
}
.kpi-card:hover {
    border-color: #252540;
    box-shadow: 0 4px 20px rgba(0,229,190,.06);
}
.kpi-card .label {
    font-size:.68rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:#8B9DB0; margin-bottom:.35rem;
}
.kpi-card .value {
    font-size:1.6rem; font-weight:700;
    color:#EEEEF5; line-height:1.1;
    font-family:'DM Mono', monospace;
}
.kpi-card .sub {
    font-size:.72rem; color:#8B9DB0; margin-top:.3rem;
}

/* ── Attribution Gap cards ──────────────────────── */
.gap-card {
    border-radius:12px; padding:1rem 1.4rem;
    margin-bottom:.6rem; border-left:4px solid;
}
.gap-undervalued { background:#0A1F14; border-color:#10B981; }
.gap-overvalued  { background:#1A0D0D; border-color:#EF4444; }
.gap-fair        { background:#0D0D1E; border-color:#00E5BE; }

.gap-card .ch-name  { font-weight:700; font-size:1rem; color:#EEEEF5; }
.gap-card .gap-badge {
    display:inline-block; padding:.15rem .65rem; border-radius:999px;
    font-size:.68rem; font-weight:700; letter-spacing:.05em; margin-left:.5rem;
}
.badge-under { background:rgba(16,185,129,.15); color:#10B981; border:1px solid rgba(16,185,129,.3); }
.badge-over  { background:rgba(239,68,68,.15);  color:#EF4444; border:1px solid rgba(239,68,68,.3); }
.badge-fair  { background:rgba(0,229,190,.12);  color:#00E5BE; border:1px solid rgba(0,229,190,.25); }
.gap-card .rec { font-size:.8rem; color:#8B9DB0; margin-top:.4rem; }

/* ── Insight callout ────────────────────────────── */
.insight {
    background:#0D0D1E; border-left:3px solid #00E5BE;
    border-radius:0 10px 10px 0; padding:.8rem 1.1rem;
    font-size:.875rem; color:#EEEEF5; margin-bottom:.5rem;
}
.insight.warn    { background:#141005; border-color:#F59E0B; }
.insight.danger  { background:#120808; border-color:#EF4444; }

/* ── Sat pills ──────────────────────────────────── */
.pill-under { background:rgba(245,158,11,.12); color:#F59E0B; border:1px solid rgba(245,158,11,.25); padding:.2rem .8rem; border-radius:999px; font-size:.72rem; font-weight:700; }
.pill-opt   { background:rgba(0,229,190,.12);  color:#00E5BE; border:1px solid rgba(0,229,190,.25); padding:.2rem .8rem; border-radius:999px; font-size:.72rem; font-weight:700; }
.pill-sat   { background:rgba(239,68,68,.12);  color:#EF4444; border:1px solid rgba(239,68,68,.25); padding:.2rem .8rem; border-radius:999px; font-size:.72rem; font-weight:700; }

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0A0A14 !important;
    border-right: 1px solid #1C1C2E !important;
}
[data-testid="stSidebar"] * { color:#8B9DB0 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:#EEEEF5 !important; font-size:1rem !important; }
[data-testid="stSidebar"] .stButton button {
    background:#00E5BE !important; color:#080810 !important;
    border:none !important; border-radius:8px !important;
    font-weight:700 !important; font-size:.85rem !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background:#00B896 !important;
}
[data-testid="stSidebar"] label { color:#EEEEF5 !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { color:#00E5BE !important; }

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1C1C2E !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight:600; font-size:.875rem;
    color:#8B9DB0 !important;
    background: transparent !important;
    border: none !important;
    padding: .75rem 1.25rem !important;
}
.stTabs [aria-selected="true"] {
    color:#00E5BE !important;
    border-bottom: 2px solid #00E5BE !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color:#EEEEF5 !important; }

/* ── Tables / DataFrames ────────────────────────── */
[data-testid="stDataFrame"] {
    font-size:.82rem !important;
    border: 1px solid #1C1C2E !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: #0E0E1A !important;
    color: #8B9DB0 !important;
    font-size: .7rem !important;
    font-weight: 700 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td { color: #EEEEF5 !important; }

/* ── Inputs & widgets ───────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: #0E0E1A !important;
    border: 1px solid #1C1C2E !important;
    color: #EEEEF5 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #00E5BE !important;
    box-shadow: 0 0 0 2px rgba(0,229,190,.15) !important;
}

/* ── Alerts / info boxes ────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid #1C1C2E !important;
}
.stAlert [data-baseweb="notification"] {
    background: #0E0E1A !important;
    border: none !important;
}

/* ── Expander ───────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0E0E1A !important;
    border: 1px solid #1C1C2E !important;
    border-radius: 10px !important;
}

/* ── Plotly charts — transparent bg ────────────── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── File uploader ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0E0E1A !important;
    border: 1px dashed #252540 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00E5BE !important;
}

/* ── Toggle / checkbox ──────────────────────────── */
[data-testid="stToggle"] [data-checked="true"] {
    background: #00E5BE !important;
}

/* ── Metric ─────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0E0E1A !important;
    border: 1px solid #1C1C2E !important;
    border-radius: 10px !important;
    padding: .75rem 1rem !important;
}
[data-testid="stMetric"] label { color: #8B9DB0 !important; font-size:.7rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #EEEEF5 !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Download button ────────────────────────────── */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid #1C1C2E !important;
    color: #8B9DB0 !important;
    border-radius: 8px !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #00E5BE !important;
    color: #00E5BE !important;
}

/* ── Spinner ─────────────────────────────────────── */
[data-testid="stSpinner"] { color: #00E5BE !important; }

/* ── Progress bar ───────────────────────────────── */
[data-testid="stProgressBar"] > div { background: #00E5BE !important; }

/* ── Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080810; }
::-webkit-scrollbar-thumb { background: #252540; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #3A3A55; }

/* ── Selection ──────────────────────────────────── */
::selection { background: rgba(0,229,190,.18); color: #00E5BE; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Auth + multi-tenant sidebar
# ─────────────────────────────────────────────────────────────────────────────
import db, auth as _auth

# Determine if Supabase is configured; fall back to single-password mode if not
_SUPABASE_MODE = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY"))

if _SUPABASE_MODE:
    # ── Supabase multi-tenant auth ────────────────────────────────────────────
    with st.sidebar:
        sb = _auth.require_auth()          # stops here if not logged in

    # Load account — create one automatically if missing (e.g. sign_up race condition)
    if "account" not in st.session_state:
        st.session_state["account"] = db.get_account(sb)

    if st.session_state.get("account") is None:
        # Account wasn't created during sign_up — create it now
        user = _auth.get_current_user()
        _acct_name = user.email.split("@")[0] if user else "My Account"
        try:
            _acct = sb.table("accounts").insert({
                "name": _acct_name, "plan": "free"
            }).execute()
            _acct_id = _acct.data[0]["id"]
            sb.table("account_members").insert({
                "account_id": _acct_id,
                "user_id":    user.id,
                "role":       "owner",
            }).execute()
            st.session_state["account"] = db.get_account(sb)
        except Exception as e:
            st.error(f"Could not create account: {e}")
            st.stop()

    account = st.session_state["account"]

    with st.sidebar:
        if account:
            st.caption(f"**{account['name']}** · {account['plan'].upper()}")
        st.markdown("---")

        # ── Brand selector ────────────────────────────────────────────────────
        st.markdown("### Brand / Client")
        brands = db.get_brands(sb, account["id"]) if account else []

        if not brands:
            st.info("No brands yet.")
            new_brand_name = st.text_input("Brand name", key="new_brand_input")
            if st.button("Create brand", use_container_width=True, type="primary"):
                if new_brand_name.strip():
                    brand = db.create_brand(sb, account["id"], new_brand_name.strip())
                    st.session_state["active_brand"] = brand
                    st.session_state.pop("active_dataset", None)
                    st.rerun()
            st.stop()

        brand_names = [b["name"] for b in brands]
        active_brand = st.session_state.get("active_brand", brands[0])
        # Ensure active_brand is in the current brand list
        if active_brand not in brands:
            active_brand = brands[0]

        selected_idx = brand_names.index(active_brand["name"]) if active_brand["name"] in brand_names else 0
        selected_name = st.selectbox("Select brand", brand_names, index=selected_idx, key="brand_select")
        if selected_name != active_brand["name"]:
            active_brand = next(b for b in brands if b["name"] == selected_name)
            st.session_state["active_brand"] = active_brand
            st.session_state.pop("active_dataset", None)
            st.session_state.pop("_csv_df", None)
            st.rerun()

        with st.expander("+ Add brand"):
            new_b = st.text_input("Name", key="add_brand_name")
            if st.button("Add", key="add_brand_btn"):
                if new_b.strip():
                    db.create_brand(sb, account["id"], new_b.strip())
                    st.rerun()

        st.markdown("---")

        # ── Dataset selector ──────────────────────────────────────────────────
        st.markdown("### Dataset")
        datasets = db.get_datasets(sb, active_brand["id"])

        _ds_options = ["⬆ Upload new dataset"] + [
            f"{d['name']}  ({d['n_weeks']}w · {d['n_channels']}ch · {d['week_end']})"
            for d in datasets
        ]
        _ds_selection = st.selectbox("Select dataset", _ds_options, key="dataset_select")

        if _ds_selection == "⬆ Upload new dataset":
            uploaded = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                help="Required: week, channel, spend, leads",
            )
            if uploaded:
                _ds_name = st.text_input(
                    "Dataset name",
                    value=uploaded.name.replace(".csv", "").replace("_", " "),
                    key="new_ds_name",
                )
                if st.button("Save dataset", use_container_width=True, type="primary"):
                    with st.spinner("Uploading…"):
                        try:
                            _df_up = validate_input_df(pd.read_csv(uploaded))
                            _user  = _auth.get_current_user()
                            _saved = db.upload_dataset(
                                sb, active_brand["id"], _ds_name,
                                _df_up, _user.id if _user else None
                            )
                            st.session_state["active_dataset"] = _saved
                            st.session_state["_csv_df"]  = _df_up
                            st.session_state["_csv_key"] = _saved["id"]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
            active_dataset = st.session_state.get("active_dataset")
        else:
            _ds_idx = _ds_options.index(_ds_selection) - 1  # offset for "Upload new"
            active_dataset = datasets[_ds_idx]
            if st.session_state.get("active_dataset", {}).get("id") != active_dataset["id"]:
                st.session_state["active_dataset"] = active_dataset
                st.session_state.pop("_csv_df", None)
                st.session_state.pop("_csv_key", None)
                st.session_state.pop("_run_cache", None)

        # Load dataset into session
        if active_dataset and "_csv_df" not in st.session_state:
            with st.spinner("Loading dataset…"):
                try:
                    st.session_state["_csv_df"]  = db.load_dataset(sb, active_dataset)
                    st.session_state["_csv_key"] = active_dataset["id"]
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
                    st.stop()

        uploaded = None   # signal: data comes from session_state, not uploader

        st.markdown("---")

        # ── Model settings ────────────────────────────────────────────────────
        st.markdown("### Model")
        auto_mode = st.toggle("Auto-tune parameters", value=True)
        if not auto_mode:
            theta       = st.slider("Adstock θ",            0.0, 0.9,  0.5, 0.05)
            alpha       = st.slider("Saturation α",          0.8, 3.0,  1.5, 0.1)
            gamma_scale = st.slider("Half-sat γ",            0.2, 2.0,  1.0, 0.1)
            ridge_alpha = st.slider("Regularisation λ",      0.1, 50.0, 5.0, 0.1)
        else:
            theta = alpha = ridge_alpha = gamma_scale = None
            st.caption("Best params selected via rolling backtest MAPE.")

        st.markdown("### Optimizer")
        optimizer_mode = st.selectbox("Mode", ["Advanced (curves + mROI)", "Simple (fast)"])
        st.caption("💡 Weekly budget — same units as your CSV.")
        _default_budget = 10_000.0
        total_budget    = st.number_input("Weekly budget ($)", min_value=0.0,
                                           value=_default_budget, step=500.0,
                                           key="optimizer_budget_raw")
        min_pct = st.slider("Min % per channel", 0.0,  0.30, 0.05, 0.01)
        max_pct = st.slider("Max % per channel", 0.10, 1.00, 1.00, 0.01)
        step    = st.slider("Allocation step %", 0.01, 0.20, 0.02, 0.01)

        _auth.render_user_menu(sb)

        # ── Run counter ───────────────────────────────────────────────────────
        _sidebar_account = st.session_state.get("account")
        if _sidebar_account:
            from plans import check_run_limit as _crl
            _rl = _crl(_sidebar_account)
            if not _rl["unlimited"] and _rl["limit"] > 0:
                _pct = _rl["used"] / max(_rl["limit"], 1)
                _color = "#10B981" if _pct < 0.7 else "#F59E0B" if _pct < 1.0 else "#EF4444"
                st.markdown(
                    f'<div style="padding:.4rem 0;font-size:.75rem;color:#8B9DB0">'
                    f'<span style="color:{_color};font-weight:700">{_rl["used"]}</span>'
                    f' / {_rl["limit"]} runs this month</div>',
                    unsafe_allow_html=True,
                )

else:
    # ── Single-password fallback (no Supabase configured) ────────────────────
    APP_PASSWORD = os.getenv("APP_PASSWORD", "adcrux")
    if "auth" not in st.session_state:
        st.session_state["auth"] = False

    with st.sidebar:
        st.markdown("---")
        if not st.session_state["auth"]:
            pw = st.text_input("Password", type="password", placeholder="Enter password")
            if st.button("Sign in", use_container_width=True, type="primary"):
                if pw == APP_PASSWORD:
                    st.session_state["auth"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.stop()
        else:
            st.caption("✅ Signed in")
            if st.button("Sign out", use_container_width=True):
                st.session_state["auth"] = False
                st.rerun()
            st.markdown("---")

    with st.sidebar:
        st.markdown("### Data")
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="Required columns: week, channel, spend, leads",
        )
        st.markdown("### Model")
        auto_mode = st.toggle("Auto-tune parameters", value=True)
        if not auto_mode:
            theta       = st.slider("Adstock θ",            0.0, 0.9,  0.5, 0.05,
                                     help="0 = instantaneous effect · 0.7 = strong carryover (e.g. YouTube, TV)")
            alpha       = st.slider("Saturation α",          0.8, 3.0,  1.5, 0.1,
                                     help="Controls the steepness of the diminishing-returns curve")
            gamma_scale = st.slider("Half-sat multiplier γ", 0.2, 2.0,  1.0, 0.1,
                                     help="γ × mean(adstocked spend) = half-saturation point per channel. "
                                          "0.4 = early saturation · 1.0 = neutral · 1.5 = late saturation")
            ridge_alpha = st.slider("Regularisation λ",      0.1, 50.0, 5.0, 0.1,
                                     help="Higher = more shrinkage. Helps when channels are correlated")
        else:
            theta = alpha = ridge_alpha = gamma_scale = None
            st.caption("Auto-tune tests parameter combinations and keeps the one with the best rolling backtest MAPE.")

        st.markdown("### Optimizer")
        optimizer_mode = st.selectbox("Mode", ["Advanced (curves + mROI)", "Simple (fast)"])
        st.caption("💡 Enter your **weekly** budget — same units as your CSV spend.")
        _default_budget = 10_000.0
        total_budget   = st.number_input("Weekly budget ($)", min_value=0.0,
                                          value=_default_budget, step=500.0,
                                          key="optimizer_budget_raw")
        min_pct = st.slider("Min % per channel", 0.0,  0.30, 0.05, 0.01)
        max_pct = st.slider("Max % per channel", 0.10, 1.00, 1.00, 0.01)
        step    = st.slider("Allocation step %", 0.01, 0.20, 0.02, 0.01)


st.markdown("""
<div class="hero-bar">
  <svg width="170" height="56" style="margin-bottom:.4rem;display:block" viewBox="0 0 165.4 52" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>@font-face{font-family:'PB';src:url('data:font/ttf;base64,AAEAAAAMAIAAAwBAR1BPU0R2THUAAAVoAAAAIEdTVUIfSCdrAAAFiAAAADBPUy8yWsxgYwAAAwgAAABgY21hcAFkAZcAAANoAAAAXGdseWY3CrhEAAAAzAAAAZRoZWFkGlEkcQAAApAAAAA2aGhlYQv8AV8AAALkAAAAJGhtdHgREADJAAACyAAAABxsb2NhAT8BnAAAAoAAAAAQbWF4cAAdAQcAAAJgAAAAIG5hbWUf9zbOAAADxAAAAYJwb3N0/7gAMgAABUgAAAAgAAIAEAAAAtICvgAHAAoAACUhByMTMxMjCwIB8/76KrP+xv61VldWfHwCvv1CAQABAf7/AAABACH/+wLPAsYAGwAAEjY2MzIWFyMmJiMiBhUUFjMyNjczBgYjIiYmNSFaoWZ9sh68FU0xT2JiTzFNFbwesn1moVoByKNbhHIsLm5cXG4uLHKDW6JoAAIAHP/4AmkC5AASAB4AABI2NjMyFhcRMxEjNQYGIyImJjUkJiMiBhUUFjMyNjUcQXBFN1saq6sYWTtFcEEBokczM0dHMzNHAW6CRi4nAQP9HFAoMEeDVj9KSUBAS0pAAAEAPgAAAZgCNAAMAAAANjMVIyIGFREjETMVAQdaNy9AQKurAf81tTdF/v0CLl0AAAEAOf/6AmUCLgAUAAABESM1BgYjIiYmNREzERQWMzI2NRECZasaWTZAYjaqOjEyOgIu/dJMJS05bEkBRv7ROD4+OAEvAAABAAUAAAJIAi4ACwAAIScHIxMDMxc3MwMTAYhrWrm4vcBrWrm7wJubAR0BEZqa/uf+6wAAAQAAAAcAkAAMAHQABgABAAAAAAAAAAAAAAAAAAMAAgAAAAAAGgBFAHUAjQCwAMoAAQAAAAQBBngu/oZfDzz1AAMD6AAAAADYpKnIAAAAANsWNsn9zv2iCa0EVAABAAcAAgAAAAAAAAH0AAAC4QAQAvoAIQKnABwBrAA+AqIAOQJMAAUAAQAABBr+ogBkCZ39zvkYCa0AAQAAAAAAAAAAAAAAAAAAAAcABANpArwABQAAAooCWAAAAEsCigJYAAABXgAyAU4AAAAACAAAAAAAAAAAAAABAAAAAAAAAAAAAAAASVRGTwCgAEEAeAQa/qIAZARvAnMAAAABAAAAAAIuAsEAAAAgAAQAAAACAAAAAwAAABQAAwABAAAAFAAEAEgAAAAOAAgAAgAGAEEAQwBkAHIAdQB4//8AAABBAEMAZAByAHUAeP///8D/v/+f/5L/kP+OAAEAAAAAAAAAAAAAAAAAAAAAAAcAWgADAAEECQAAAKIAAAADAAEECQABAA4AogADAAEECQACAAgAsAADAAEECQADADYAuAADAAEECQAEABgA7gADAAEECQAFAAoBBgADAAEECQAGABgBEABDAG8AcAB5AHIAaQBnAGgAdAAgADIAMAAyADAAIABUAGgAZQAgAFAAbwBwAHAAaQBuAHMAIABQAHIAbwBqAGUAYwB0ACAAQQB1AHQAaABvAHIAcwAgACgAaAB0AHQAcABzADoALwAvAGcAaQB0AGgAdQBiAC4AYwBvAG0ALwBpAHQAZgBvAHUAbgBkAHIAeQAvAFAAbwBwAHAAaQBuAHMAKQBQAG8AcABwAGkAbgBzAEIAbwBsAGQASQBUAEYATwA7ACAAUABvAHAAcABpAG4AcwAgAEIAbwBsAGQAOwAgADQALgAwADAANABiADgAUABvAHAAcABpAG4AcwAgAEIAbwBsAGQANAAuADAAMAA0AFAAbwBwAHAAaQBuAHMALQBCAG8AbABkAAAAAwAAAAAAAP+1ADIAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAACgAcAB4AAURGTFQACAAEAAAAAP//AAAAAAAAAAEAAAAKACwALgADREZMVAAUZGV2MgAeZGV2YQAeAAQAAAAA//8AAAAAAAAAAAAA') format('truetype');font-weight:800;}</style>
    <radialGradient id="g_hb" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#00E5BE" stop-opacity=".25"/>
      <stop offset="100%" stop-color="#00E5BE" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <text x="0" y="40" font-family="PB,DM Sans,-apple-system,sans-serif" font-weight="800" font-size="42" letter-spacing="-1.0" fill="#3D5270">Ad</text>
  <text x="55.4" y="40" font-family="PB,DM Sans,-apple-system,sans-serif" font-weight="800" font-size="42" letter-spacing="-1.0" fill="#F1F5F9">Cru</text>
  <circle cx="143.0" cy="28.0" r="20.0" fill="url(#g_hb)"/>
  <line x1="129.0" y1="14.0" x2="157.0" y2="42.0" stroke="#1A3250" stroke-width="6.4" stroke-linecap="round"/>
  <line x1="157.0" y1="14.0" x2="129.0" y2="42.0" stroke="#1A3250" stroke-width="6.4" stroke-linecap="round"/>
  <circle cx="143.0" cy="28.0" r="10.0" fill="#00E5BE" fill-opacity=".18"/>
  <circle cx="143.0" cy="28.0" r="6.0" fill="#00E5BE"/>
</svg>
  <p>
    Google says Google generated your leads. Meta says Meta generated your leads.
    &nbsp;·&nbsp; <strong>AdCrux shows you what's actually true.</strong>
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────────────────────
(tab_overview, tab_attribution, tab_saturation,
 tab_optimizer, tab_scenarios, tab_diagnostics) = st.tabs([
    "🏠 Overview",
    "🔍 Attribution Gap",
    "📈 Saturation",
    "⚙️ Optimizer",
    "🔀 Scenarios",
    "🔬 Diagnostics",
])


# ─────────────────────────────────────────────────────────────────────────────
#  Load & validate — session_state is the source of truth in both modes
# ─────────────────────────────────────────────────────────────────────────────

# In Supabase mode: df comes from session_state (loaded from Storage above)
# In password mode: df comes from the file uploader → also cached in session_state
if not _SUPABASE_MODE and uploaded is not None:
    _file_key = f"{uploaded.name}__{uploaded.size}"
    if st.session_state.get("_csv_key") != _file_key:
        try:
            _df_new = validate_input_df(pd.read_csv(uploaded))
            st.session_state["_csv_df"]  = _df_new
            st.session_state["_csv_key"] = _file_key
        except Exception as exc:
            st.error(f"❌ Data error: {exc}")
            st.stop()

# No data available yet — show empty state
if "_csv_df" not in st.session_state:
    with tab_overview:
        st.markdown('<p class="sec-header">How it works</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, icon, title, body in [
            (c1, "📁", "Upload your data",
             "CSV with spend and leads by channel by week. Optionally include last-click leads from each platform."),
            (c2, "🧠", "Model runs automatically",
             "AdCrux fits a Marketing Mix Model that separates the true incremental contribution of each channel."),
            (c3, "💡", "See the real picture",
             "Compare what platforms claim vs what the model finds. Reallocate budget based on actual incremental impact."),
        ]:
            with col:
                st.markdown(
                    f'<div class="kpi-card" style="text-align:left;padding:1.4rem">'
                    f'<div style="font-size:1.8rem;margin-bottom:.5rem">{icon}</div>'
                    f'<div style="font-weight:700;color:#EEEEF5;margin-bottom:.4rem">{title}</div>'
                    f'<div style="font-size:.85rem;color:#8B9DB0">{body}</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<p class="sec-header">CSV Format</p>', unsafe_allow_html=True)
        st.markdown("""
| Column | Required | Description |
|---|---|---|
| `week` | ✅ | Monday date in YYYY-MM-DD |
| `channel` | ✅ | Channel name |
| `spend` | ✅ | Total spend this week |
| `leads` | ✅ | Total leads this week |
| `lastclick_leads` | ⭐ Optional | Platform-reported leads — unlocks Attribution Gap |
| `promo` / `holiday` | ⭐ Optional | `1`/`0` flags for exogenous controls |
        """)
    st.stop()

# Data is ready
df = st.session_state["_csv_df"]


_val_warnings = validation_warnings(df)
if _val_warnings:
    for _w in _val_warnings:
        st.warning(_w)

channels_list = sorted(df["channel"].unique().tolist())
n_channels    = len(channels_list)
n_weeks       = df["week"].nunique()
color_map     = {ch: PALETTE[i % len(PALETTE)] for i, ch in enumerate(channels_list)}
lc_available  = has_lastclick(df)


from plans import can, check_run_limit, feature_level, get_upgrade_msg, run_counter_label, PLANS
from insights import build_context, get_insights, render_insights_panel

# Get account from session_state (set during sidebar in Supabase mode)
_account = st.session_state.get("account") if _SUPABASE_MODE else {"plan": "professional", "runs_this_month": 0}

# ─────────────────────────────────────────────────────────────────────────────
#  Run limit check — before running any expensive computation
# ─────────────────────────────────────────────────────────────────────────────
_run_limit = check_run_limit(_account)
_dataset_key = st.session_state.get("_csv_key", "")
_already_ran = st.session_state.get(f"_ran_{_dataset_key}", False)

if not _run_limit["ok"] and not _already_ran:
    msg = get_upgrade_msg("run_limit")
    st.error(f"🚫 **{msg['title']}** — {msg['body']}")
    if _SUPABASE_MODE:
        st.info(f"You've used **{_run_limit['used']} / {_run_limit['limit']}** runs this month. "
                f"Runs reset on the 1st of each month.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  Auto-tune / manual params — gated by plan
# ─────────────────────────────────────────────────────────────────────────────
_can_autotune = can(_account, "auto_tune")

if auto_mode and _can_autotune:
    st.markdown(
        '<small style="color:#8B9DB0">🔄 Auto-tuning model parameters '
        '(first run ~20-30s · cached after)</small>',
        unsafe_allow_html=True)
    best, leaderboard = auto_tune_params(
        df,
        min_train_weeks=16, test_weeks=4,
        thetas=(0.2, 0.4, 0.6),
        alphas=(1.0, 1.3, 1.8, 2.5),
        ridge_alphas=(1.0, 5.0),
        gamma_scales=(0.4, 0.7, 1.3),
    )
    if best is None or best.get("backtest_mape_mean", 999) >= 900:
        theta, alpha, ridge_alpha, gamma_scale = 0.5, 1.5, 5.0, 1.0
    else:
        theta       = float(best["theta"])
        alpha       = float(best["alpha"])
        ridge_alpha = float(best["ridge_alpha"])
        gamma_scale = float(best.get("gamma_scale", 1.0))
elif auto_mode and not _can_autotune:
    # Starter: show info banner, fall back to defaults
    st.info("ℹ️ Auto-tune is available on **Professional** and above. "
            "Using recommended default parameters.", icon="🔒")
    theta, alpha, ridge_alpha, gamma_scale = 0.5, 1.5, 5.0, 1.0
    best = None
    leaderboard = None

hp  = HyperParams(theta=theta, alpha=alpha, ridge_alpha=ridge_alpha, gamma_scale=gamma_scale)
# auto_tune_channels=False: global hp params tuned via backtest MAPE (tuning.py)
# Per-channel tuning requires 80+ weeks — global params are safer for MVP
mmm = MMMLeadGen(hp=hp, auto_tune_channels=False)
fit = mmm.fit(df)

# Mark this dataset+session as "run" and increment monthly counter
if _SUPABASE_MODE and not _already_ran and _account:
    try:
        db.increment_run_count(sb, _account["id"])
        # Refresh account to get updated run count
        _refreshed = db.refresh_account(sb, _account["id"])
        if _refreshed:
            st.session_state["account"] = {**_account, **_refreshed}
            _account = st.session_state["account"]
    except Exception:
        pass  # Don't block the app if counter fails
    st.session_state[f"_ran_{_dataset_key}"] = True


# ─────────────────────────────────────────────────────────────────────────────
#  Shared computed objects
# ─────────────────────────────────────────────────────────────────────────────
backtest_df  = mmm.rolling_backtest(df, min_train_weeks=16, test_weeks=4)
conf         = compute_confidence_score(
    df=df, backtest_df=backtest_df, hp=hp,
    negative_coef_channels=fit.get('negative_coef_channels', []))
tracking     = detect_tracking_breaks(df)
contrib      = mmm.contributions_by_channel(df)
avp          = mmm.actual_vs_predicted(df)
attr_gap     = mmm.attribution_gap(df)
pivot_spend  = channel_weekly_spend(df)

spend_totals  = df.groupby("channel")["spend"].sum()
total_spend   = float(spend_totals.sum())
total_leads   = total_leads_scalar(df)
blended_cpl   = total_spend / total_leads if total_leads > 0 else 0.0

# Data quality report (used in Overview and Diagnostics)
dqr = data_quality_report(df)

# Bootstrap confidence intervals — Professional+ only
_can_bootstrap = can(_account, "bootstrap_ci")
with st.spinner("Computing confidence intervals…"):
    boot_ci = (
        mmm.bootstrap_mmm_shares(df, n_bootstrap=150)
        if (_can_bootstrap and n_weeks >= MIN_WEEKS_FOR_CI)
        else pd.DataFrame()
    )


# ─────────────────────────────────────────────────────────────────────────────────
#  MODULE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────────
with tab_overview:

    # ── KPI cards ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    _avg_bt_mape = backtest_df["mape"].mean() if not backtest_df.empty else None
    _mape_sub    = ("✅ Strong" if _avg_bt_mape < 0.10 else
                    "🟡 Acceptable" if _avg_bt_mape < 0.20 else
                    "⚠️ Review") if _avg_bt_mape is not None else "no backtest"
    kpi_data = [
        (k1, "Total Spend",      f"${total_spend:,.0f}",                          f"{n_weeks} weeks"),
        (k2, "Total Leads",      f"{total_leads:,.0f}",                            f"{n_channels} channels"),
        (k3, "Blended CPL",      f"${blended_cpl:,.2f}",                          "cost per lead"),
        (k4, "Backtest MAPE",    f"{_avg_bt_mape:.1%}" if _avg_bt_mape else "—",  _mape_sub),
        (k5, "Confidence",       f"{conf['score']}/100",                           conf['label']),
    ]
    for col, label, value, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Data quality strip
    st.markdown("")
    _dqr_color = {"GOOD": "#10B981", "FAIR": "#F59E0B", "POOR": "#EF4444"}[dqr["overall"]]
    _dqr_items = " &nbsp;·&nbsp; ".join(dqr["strengths"] + dqr["warnings"] + dqr["issues"])
    st.markdown(
        f'<div style="border-left:3px solid {_dqr_color};background:#0A0A18;'
        f'padding:.6rem 1rem;border-radius:4px;font-size:.78rem;color:#8B9DB0;line-height:1.6">'
        f'<strong style="color:{_dqr_color}">Data quality: {dqr["overall"]}</strong>'
        f'&nbsp;&nbsp;{_dqr_items}</div>',
        unsafe_allow_html=True)
    if dqr["issues"]:
        st.error("🔴 Data quality issues detected — model reliability is reduced. See Diagnostics tab.")

    if fit.get("exo_cols"):
        st.info(
            f"🎛️ **Exogenous controls active:** {', '.join(fit['exo_cols'])}. "
            f"These have been removed from channel contributions — ensuring promos/seasonality "
            f"don't inflate any channel's estimated impact."
        )

    st.markdown("")

    # ─────────────────────────────────────────────────────────────────────────
    # Overview layout differs by plan:
    #   Pro/Agency  → AI Insights (replaces Key Insights) + compact At a Glance
    #   Free/Starter → Key Insights (rule-based) + full At a Glance teaser
    # ─────────────────────────────────────────────────────────────────────────
    _is_pro = can(_account, "attribution_gap")   # Pro+ feature as proxy

    if _is_pro:
        # ── AI Insights Engine (Pro+) ─────────────────────────────────────────
        _insights_key = st.session_state.get("_csv_key", "default")
        _insights_ctx = build_context(
            total_spend=total_spend, total_leads=total_leads,
            blended_cpl=blended_cpl, n_weeks=n_weeks, conf=conf,
            backtest_df=backtest_df, attr_gap=attr_gap, contrib=contrib,
            pivot_spend=pivot_spend, fit=fit, tracking=tracking, dqr=dqr,
        )
        _insights_cached = f"_insights_{_insights_key}" in st.session_state
        if not _insights_cached:
            _ins_ph = st.empty()
            with _ins_ph:
                render_insights_panel([], is_loading=True, cache_key=_insights_key)
            with st.spinner(""):
                _insights = get_insights(_insights_ctx, cache_key=_insights_key)
            _ins_ph.empty()
            st.rerun()
        else:
            _insights = get_insights(_insights_ctx, cache_key=_insights_key)
            render_insights_panel(_insights, cache_key=_insights_key)
            if st.button("🔄 Regenerate insights", key="regen_insights_btn",
                         help="Generate a fresh set of insights from the same model data"):
                from insights import invalidate_insights
                invalidate_insights(_insights_key)
                st.rerun()

        # ── Attribution At a Glance — compact version for Pro (has full tab) ──
        if lc_available:
            st.markdown("")
            st.markdown('<p class="sec-header">Attribution Gap — Top Signals</p>',
                        unsafe_allow_html=True)
            valid_gaps  = attr_gap.dropna(subset=["gap_points"])
            undervalued = valid_gaps[valid_gaps["gap_points"] >  0.05].sort_values("gap_points", ascending=False)
            overvalued  = valid_gaps[valid_gaps["gap_points"] < -0.05].sort_values("gap_points", ascending=True)
            if not undervalued.empty or not overvalued.empty:
                ha, hb = st.columns(2)
                with ha:
                    if not undervalued.empty:
                        top = undervalued.iloc[0]
                        st.markdown(
                            f'<div class="gap-card gap-undervalued">' +
                            f'<div class="ch-name">{top["channel"]}<span class="gap-badge badge-under">UNDERVALUED</span></div>' +
                            f'<div class="rec"><strong>+{top["gap_points"]*100:.1f} pp</strong> undercredited · ' +
                            f'Platform: {top["lastclick_share"]*100:.1f}% → Model: {top["mmm_share"]*100:.1f}%</div></div>',
                            unsafe_allow_html=True)
                with hb:
                    if not overvalued.empty:
                        bot = overvalued.iloc[0]
                        st.markdown(
                            f'<div class="gap-card gap-overvalued">' +
                            f'<div class="ch-name">{bot["channel"]}<span class="gap-badge badge-over">OVERVALUED</span></div>' +
                            f'<div class="rec"><strong>{bot["gap_points"]*100:.1f} pp</strong> overcredited · ' +
                            f'Platform: {bot["lastclick_share"]*100:.1f}% → Model: {bot["mmm_share"]*100:.1f}%</div></div>',
                            unsafe_allow_html=True)
            st.caption("👉 Go to **Attribution Gap** tab for the full breakdown.")

    else:
        # ── Key Insights — rule-based (Free / Starter) ────────────────────────
        st.markdown('<p class="sec-header">Key Insights</p>', unsafe_allow_html=True)
        best_ch  = chan_contrib_df.sort_values("est_roas", ascending=False).iloc[0]["channel"]
        worst_ch = chan_contrib_df.sort_values("est_roas").iloc[0]["channel"]
        _ki = [
            ("info",   f"🏆 <b>{best_ch}</b> has the highest estimated ROAS in your mix."),
            ("warn",   f"⚠️ <b>{worst_ch}</b> has the lowest estimated ROAS — review allocation."),
            ("info",   f"📊 Blended CPL across the portfolio: <b>${blended_cpl:,.2f}</b>."),
        ]
        if conf["score"] < 55:
            _ki.append(("danger", f"🔴 Model confidence is LOW ({conf['score']}/100) — check Diagnostics."))
        elif conf["score"] < 75:
            _ki.append(("warn",   f"🟡 Model confidence is MEDIUM ({conf['score']}/100) — directional only."))
        else:
            _ki.append(("info",   f"✅ Model confidence is HIGH ({conf['score']}/100) — results are reliable."))
        for kind, text in _ki:
            css = "warn" if kind == "warn" else ("danger" if kind == "danger" else "")
            st.markdown(f'<div class="insight {css}">{text}</div>', unsafe_allow_html=True)

        # ── Attribution At a Glance — full teaser for Starter ────────────────
        st.markdown("")
        if lc_available:
            st.markdown('<p class="sec-header">⚡ The Attribution Problem — At a Glance</p>',
                        unsafe_allow_html=True)
            valid_gaps  = attr_gap.dropna(subset=["gap_points"])
            undervalued = valid_gaps[valid_gaps["gap_points"] >  0.05].sort_values("gap_points", ascending=False)
            overvalued  = valid_gaps[valid_gaps["gap_points"] < -0.05].sort_values("gap_points", ascending=True)
            if not undervalued.empty or not overvalued.empty:
                ha, hb = st.columns(2)
                with ha:
                    if not undervalued.empty:
                        top = undervalued.iloc[0]
                        _ch2  = top["channel"]
                        _rec2 = top["recommendation"]
                        st.markdown(
                            f'<div class="gap-card gap-undervalued">' +
                            f'<div class="ch-name">{_ch2}<span class="gap-badge badge-under">UNDERVALUED</span></div>' +
                            f'<div style="margin:.6rem 0;font-size:.85rem;color:#8B9DB0">Platforms say: <strong>{top["lastclick_share"]*100:.1f}%</strong> · Model says: <strong>{top["mmm_share"]*100:.1f}%</strong></div>' +
                            f'<div style="font-size:1.4rem;font-weight:800;color:#10B981">+{top["gap_points"]*100:.1f} pp undercredited</div>' +
                            f'<div class="rec">{_rec2}</div></div>',
                            unsafe_allow_html=True)
                with hb:
                    if not overvalued.empty:
                        bot = overvalued.iloc[0]
                        _ch3  = bot["channel"]
                        _rec3 = bot["recommendation"]
                        st.markdown(
                            f'<div class="gap-card gap-overvalued">' +
                            f'<div class="ch-name">{_ch3}<span class="gap-badge badge-over">OVERVALUED</span></div>' +
                            f'<div style="margin:.6rem 0;font-size:.85rem;color:#8B9DB0">Platforms say: <strong>{bot["lastclick_share"]*100:.1f}%</strong> · Model says: <strong>{bot["mmm_share"]*100:.1f}%</strong></div>' +
                            f'<div style="font-size:1.4rem;font-weight:800;color:#EF4444">{bot["gap_points"]*100:.1f} pp overcredited</div>' +
                            f'<div class="rec">{_rec3}</div></div>',
                            unsafe_allow_html=True)
            st.caption("👉 Upgrade to **Professional** to unlock the full Attribution Gap analysis.")
        else:
            st.markdown('<p class="sec-header">Attribution</p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="insight warn">
                ⭐ <strong>Unlock the Attribution Gap feature</strong> — add a
                <code>lastclick_leads</code> column to your CSV to compare what each
                platform claims vs what the model actually finds.
            </div>""", unsafe_allow_html=True)

    # ── Spend over time ───────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Spend by Channel Over Time</p>', unsafe_allow_html=True)

    pivot_long = pivot_spend.reset_index().melt(id_vars="week", var_name="channel", value_name="spend")
    fig_area   = px.area(
        pivot_long, x="week", y="spend", color="channel",
        color_discrete_map=color_map,
        labels={"spend": "Weekly Spend ($)", "week": ""},
        template="plotly_dark",
    )
    fig_area.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
        margin=dict(t=30, b=10, l=0, r=0), hovermode="x unified",
    )
    st.plotly_chart(fig_area, use_container_width=True)

    # ── Spend share + performance table ──────────────────────────────────────
    col_donut, col_table = st.columns([1, 2])

    chan_contrib_df = (
        contrib[contrib["component"] != "baseline"]
        .rename(columns={"component": "channel"})
        .merge(spend_totals.reset_index().rename(columns={"spend": "total_spend"}), on="channel", how="left")
    )
    chan_contrib_df["est_roas"]    = chan_contrib_df["total_contribution"] / chan_contrib_df["total_spend"].replace(0, np.nan)
    chan_contrib_df["spend_share"] = chan_contrib_df["total_spend"] / max(total_spend, 1e-9)
    # Propagate unreliable flag from model to all downstream views
    _unreliable_set = set(fit.get("negative_coef_channels", []))
    chan_contrib_df["is_unreliable"] = chan_contrib_df["channel"].isin(_unreliable_set)

    with col_donut:
        st.markdown('<p class="sec-header">Spend Share</p>', unsafe_allow_html=True)
        fig_d = px.pie(
            chan_contrib_df, names="channel", values="total_spend",
            color="channel", color_discrete_map=color_map, hole=0.55,
            template="plotly_dark",
        )
        fig_d.update_traces(textposition="outside", textinfo="percent+label")
        fig_d.update_layout(showlegend=False, margin=dict(t=10, b=10, l=0, r=0
    ))
        st.plotly_chart(fig_d, use_container_width=True)

    with col_table:
        st.markdown('<p class="sec-header">Channel Performance</p>', unsafe_allow_html=True)
        tbl = chan_contrib_df[["channel", "total_spend", "total_contribution", "est_roas", "spend_share", "is_unreliable"]].copy()
        tbl["channel"] = tbl.apply(
            lambda r: f"⚠️ {r['channel']}" if r["is_unreliable"] else r["channel"], axis=1)
        tbl = tbl.drop(columns=["is_unreliable"])
        tbl.columns = ["Channel", "Spend ($)", "MMM Contribution", "Est. ROAS", "Spend Share"]
        tbl["Spend ($)"]        = tbl["Spend ($)"].map("${:,.0f}".format)
        tbl["MMM Contribution"] = tbl["MMM Contribution"].map("{:,.1f}".format)
        tbl["Est. ROAS"]        = tbl["Est. ROAS"].map("{:.2f}x".format)
        tbl["Spend Share"]      = tbl["Spend Share"].map("{:.1%}".format)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        if _unreliable_set:
            st.caption(f"⚠️ Channels marked with ⚠️ have constrained model coefficients — ROAS estimates are directional only.")

    # ── Key insights ──────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 — ATTRIBUTION GAP
# ─────────────────────────────────────────────────────────────────────────────
with tab_attribution:

    if not can(_account, "attribution_gap"):
        _msg = get_upgrade_msg("attribution_gap")
        st.markdown(f'<p class="sec-header">🔒 {_msg["title"]}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight">{_msg["body"]}<br><br>'
            f'<strong>Available on Professional ($399/mo) and Agency ($799/mo).</strong></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown('<p class="sec-header">The Platform Bias Problem</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
        <b>Why this matters:</b> Every ad platform measures attribution using its own model —
        and every platform has an incentive to take credit for as many conversions as possible.
        Google attributes success using last-click search. Meta uses a 7-day click / 1-day view window.
        Neither accounts for the full customer journey or for channels that operate higher in the funnel.
        <br><br>
        Marketing Mix Modeling measures the <b>incremental contribution</b> of each channel
        by analysing the statistical relationship between spend and outcomes across time —
        without relying on any platform's self-reported numbers.
    </div>
    """, unsafe_allow_html=True)

    # ── Model uncertainty disclaimer ──────────────────────────────────────────
    _ci_available = not boot_ci.empty
    if _ci_available:
        _max_width = boot_ci["ci_width"].max()
        _ci_quality = ("narrow — estimates are robust" if _max_width < 0.05
                       else "moderate — directional insights are reliable"
                       if _max_width < 0.12 else "wide — treat exact figures as directional only")
        st.markdown(
            f'''<div style="background:#F1F5F9;border-left:4px solid #6366F1;padding:.75rem 1rem;
            border-radius:4px;font-size:.82rem;color:#8B9DB0;margin-bottom:.5rem">
            📊 <strong>Model uncertainty:</strong> Confidence intervals are <strong>{_ci_quality}</strong>.
            The shaded bands on the chart below show the P10–P90 range from {150} bootstrap samples —
            how much the estimates would shift if different weeks had been observed.
            Channels with narrow bands have robust attribution. Wide bands mean the model
            needs more data or the channel's spend pattern is irregular.
            </div>''', unsafe_allow_html=True)
    else:
        st.markdown(
            f'''<div style="background:#FFF7ED;border-left:4px solid #F59E0B;padding:.75rem 1rem;
            border-radius:4px;font-size:.82rem;color:#8B9DB0;margin-bottom:.5rem">
            ⚠️ <strong>Confidence intervals unavailable</strong> — {MIN_WEEKS_FOR_CI}+ weeks of data
            required (you have {n_weeks}). Attribution shares shown are point estimates only.
            Treat as directional guidance, not precise measurements.
            </div>''', unsafe_allow_html=True)

    if not lc_available:
        st.warning(
            "⭐ Add a `lastclick_leads` column to your CSV to unlock the full Attribution Gap comparison. "
            "Without it, you can still see the MMM contribution breakdown below."
        )

    # ── The gap chart ─────────────────────────────────────────────────────────
    if lc_available:
        st.markdown('<p class="sec-header">Platform Claims vs Model Reality</p>', unsafe_allow_html=True)

        gap_plot = attr_gap.dropna(subset=["gap_points"]).copy()
        gap_plot["lc_pct"]  = gap_plot["lastclick_share"]  * 100
        gap_plot["mmm_pct"] = gap_plot["mmm_share"] * 100
        gap_plot["gap_pct"] = gap_plot["gap_points"] * 100

        bar_df = pd.concat([
            gap_plot[["channel", "lc_pct"]].rename(columns={"lc_pct": "Share (%)"}).assign(Source="Platform (last-click)"),
            gap_plot[["channel", "mmm_pct"]].rename(columns={"mmm_pct": "Share (%)"}).assign(Source="Model (MMM)"),
        ])

        fig_gap = px.bar(
            bar_df, x="channel", y="Share (%)", color="Source",
            barmode="group",
            color_discrete_map={"Platform (last-click)": "#94A3B8", "Model (MMM)": "#6366F1"},
            template="plotly_dark",
            labels={"Share (%)": "% of Total Leads", "channel": ""},
        )

        # Add bootstrap CI error bars on MMM bars
        if _ci_available and not boot_ci.empty:
            ci_map = boot_ci.set_index("channel")
            for ch in gap_plot["channel"]:
                if ch in ci_map.index:
                    p10 = float(ci_map.loc[ch, "ci_p10"]) * 100
                    p50 = float(ci_map.loc[ch, "ci_p50"]) * 100
                    p90 = float(ci_map.loc[ch, "ci_p90"]) * 100
                    mmm_val = float(gap_plot.loc[gap_plot["channel"]==ch, "mmm_pct"].values[0])
                    fig_gap.add_trace(go.Scatter(
                        x=[ch, ch], y=[p10, p90],
                        mode="lines+markers",
                        line=dict(color="#312E81", width=2.5),
                        marker=dict(symbol="line-ew", size=10, color="#312E81",
                                    line=dict(width=2.5, color="#312E81")),
                        showlegend=False,
                        hovertemplate=f"{ch} MMM P10–P90: {p10:.1f}%–{p90:.1f}%<extra></extra>",
                    ))

        fig_gap.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
            margin=dict(t=30, b=10, l=0, r=0),
        )
        st.plotly_chart(fig_gap, use_container_width=True)
        if _ci_available:
            st.caption(
                "Dark vertical bars on MMM columns = P10–P90 confidence interval from bootstrap resampling. "
                "Narrow bars = robust estimate. Wide bars = more uncertainty, treat directionally."
            )

        # Gap waterfall
        st.markdown('<p class="sec-header">Attribution Gap by Channel  (+ = Undervalued  ·  − = Overvalued)</p>',
                    unsafe_allow_html=True)

        gap_plot_sorted = gap_plot.sort_values("gap_pct")
        fig_wf = px.bar(
            gap_plot_sorted,
            x="gap_pct", y="channel", orientation="h",
            color="gap_pct",
            color_continuous_scale=[[0, "#EF4444"], [0.5, "#E2E8F0"], [1, "#10B981"]],
            template="plotly_dark",
            labels={"gap_pct": "Gap (percentage points)", "channel": ""},
            text=gap_plot_sorted["gap_pct"].map("{:+.1f}pp".format),
        )
        fig_wf.update_traces(textposition="outside")
        fig_wf.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=0, r=0
    ),
            xaxis_zeroline=True,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        st.caption(
            "**How to read this:** A positive gap means the model finds the channel generates more leads "
            "than the platform claims — it's being undercredited. A negative gap means the platform "
            "is taking more credit than the model can statistically justify."
        )

        # ── Per-channel cards with recommendations ─────────────────────────────
        st.markdown('<p class="sec-header">Channel-by-Channel Verdict</p>', unsafe_allow_html=True)

        for _, row in attr_gap.iterrows():
            if row["lastclick_share"] is None:
                continue
            lc_p  = row["lastclick_share"] * 100
            mmm_p = row["mmm_share"] * 100
            gap_p = row["gap_points"] * 100 if row["gap_points"] is not None else 0

            css_card = {
                "Undervalued": "gap-undervalued",
                "Overvalued":  "gap-overvalued",
            }.get(row["bias_label"], "gap-fair")

            css_badge = {
                "Undervalued": "badge-under",
                "Overvalued":  "badge-over",
            }.get(row["bias_label"], "badge-fair")

            # Bootstrap CI for this channel
            _ci_row = (boot_ci[boot_ci["channel"] == row["channel"]].iloc[0]
                       if _ci_available and not boot_ci.empty
                       and (boot_ci["channel"] == row["channel"]).any() else None)
            _ci_str = (f"Model range (P10–P90): <strong>{_ci_row['ci_p10']*100:.1f}%–{_ci_row['ci_p90']*100:.1f}%</strong>"
                       if _ci_row is not None else "")
            _unreliable_tag = " ⚠️ <em style='font-size:.75rem;color:#F59E0B'>constrained coef</em>" if row.get("is_unreliable") else ""

            st.markdown(f"""
            <div class="gap-card {css_card}">
                <div class="ch-name">
                    {row['channel']}{_unreliable_tag}
                    <span class="gap-badge {css_badge}">{row['bias_label'].upper()}</span>
                </div>
                <div style="display:flex;gap:2rem;margin:.5rem 0;font-size:.85rem;color:#8B9DB0;flex-wrap:wrap">
                    <span>Platform says: <strong>{lc_p:.1f}%</strong></span>
                    <span>Model says: <strong>{mmm_p:.1f}%</strong></span>
                    <span>Gap: <strong>{gap_p:+.1f} pp</strong></span>
                    <span>MMM ROAS: <strong>{row['mmm_roas']:.2f}x</strong></span>
                    {"<span>" + _ci_str + "</span>" if _ci_str else ""}
                </div>
                <div class="rec">{row['recommendation']}</div>
            </div>""", unsafe_allow_html=True)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Actual vs Predicted Leads</p>', unsafe_allow_html=True)

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=avp.index, y=avp["baseline"],
        name="Organic Baseline",
        fill="tozeroy", fillcolor="rgba(148,163,184,.15)",
        line=dict(color="#94A3B8", width=1.5, dash="dot"), mode="lines",
    ))
    fig_avp.add_trace(go.Scatter(
        x=avp.index, y=avp["predicted"], name="Model Predicted",
        line=dict(color="#6366F1", width=2, dash="dash"), mode="lines",
    ))
    fig_avp.add_trace(go.Scatter(
        x=avp.index, y=avp["actual"], name="Actual Leads",
        line=dict(color="#10B981", width=2.5), mode="lines+markers",
        marker=dict(size=4),
    ))
    fig_avp.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
        hovermode="x unified", margin=dict(t=30, b=10, l=0, r=0),
        yaxis_title="Leads", xaxis_title="",
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    st.caption(
        "**Green line** = actual leads · **Purple dashed** = model prediction · "
        "**Grey area** = organic baseline (leads you'd have generated with zero paid media)"
    )

    # ── Contribution breakdown ────────────────────────────────────────────────
    st.markdown('<p class="sec-header">MMM Contribution by Channel</p>', unsafe_allow_html=True)
    roas_df = chan_contrib_df.sort_values("est_roas").copy()
    roas_df["label"] = roas_df.apply(
        lambda r: f"⚠️ {r['channel']}" if r.get("is_unreliable", False) else r["channel"], axis=1)
    fig_roas = px.bar(
        roas_df, x="est_roas", y="label", orientation="h",
        color="channel", color_discrete_map=color_map,
        labels={"est_roas": "Estimated ROAS", "label": ""},
        template="plotly_dark",
        text=roas_df["est_roas"].map("{:.2f}x".format),
    )
    fig_roas.update_traces(textposition="outside")
    fig_roas.update_layout(showlegend=False, margin=dict(t=10, b=10, l=0, r=0
    ))
    st.plotly_chart(fig_roas, use_container_width=True)
    if _unreliable_set:
        st.caption("⚠️ = channel with constrained coefficient. ROAS estimate is directional only.")


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 — SATURATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_saturation:

    st.markdown('<p class="sec-header">Response Curve Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight">
        Every channel has a point of diminishing returns. Below that point, each extra dollar
        generates strong incremental leads. Beyond it, spend is increasingly wasted.
        These curves show you where each channel sits — and where to scale vs pull back.
        <br><br>
        <em>Curve shape is governed by the adstock decay (θ) and Hill saturation (α) parameters
        optimised for your data via walk-forward backtest. See Diagnostics → Model Parameters
        for the values used.</em>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns([1, 3])
    with sc1:
        sel_ch     = st.selectbox("Channel", channels_list)
        curve_mode = st.radio("Curve mode", ["Advanced", "Simple"], index=0)
        if curve_mode == "Advanced":
            st.caption(
                "**Advanced (recommended):** varies this channel's spend while holding all "
                "others at historical average. Shows real-world diminishing returns in context."
            )
        else:
            st.caption(
                "**Simple:** abstract curve — channel viewed in isolation, no cross-channel context. "
                "Useful for understanding the functional form; less useful for budget decisions."
            )

    ch_avg = float(pivot_spend[sel_ch].mean()) if sel_ch in pivot_spend else 1.0
    if ch_avg <= 0:
        ch_avg = 1.0
    pts = np.linspace(ch_avg * 0.2, ch_avg * 2.5, 40)

    if curve_mode == "Advanced":
        cdf   = mmm.response_curve_advanced(df, sel_ch, pts)
        xcol  = "avg_weekly_spend"
        ycol  = "incremental_total_leads"
    else:
        cdf   = mmm.response_curve(sel_ch, pts)
        xcol  = "spend"
        ycol  = "incremental_vs_min"
        if "predicted_leads" in cdf.columns:
            cdf["mroi_like"] = cdf["predicted_leads"].diff() / cdf["spend"].diff()

    # Saturation status
    if "mroi_like" in cdf.columns:
        mroi_now = float(cdf.iloc[(cdf[xcol] - ch_avg).abs().argsort()[:1]]["mroi_like"].values[0])
        mroi_max = cdf["mroi_like"].dropna().max()
        sat_pct  = max(0.0, min(1.0, 1.0 - (mroi_now / mroi_max if mroi_max > 0 else 0.5)))

        if sat_pct < 0.40:
            s_pill, s_label, s_text = "pill-under", "⬆ Underinvested", \
                "Significant headroom to scale. Consider allocating more budget here."
        elif sat_pct < 0.75:
            s_pill, s_label, s_text = "pill-opt", "✓ Optimal", \
                "Spend is in the efficient zone. Maintain and monitor."
        else:
            s_pill, s_label, s_text = "pill-sat", "⬇ Saturated", \
                "Diminishing returns are significant. Reallocate to underinvested channels."
    else:
        s_pill, s_label, s_text, sat_pct = "pill-opt", "—", "", 0.5

    with sc2:
        st.markdown(f"**Saturation status:** <span class='{s_pill}'>{s_label}</span>",
                    unsafe_allow_html=True)
        if s_text:
            st.caption(s_text)

    fig_curve = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=["Response Curve (Incremental Leads)", "Marginal Efficiency (mROI)"],
    )
    fig_curve.add_trace(go.Scatter(
        x=cdf[xcol], y=cdf[ycol], name="Incremental Leads",
        line=dict(color=color_map.get(sel_ch, PALETTE[0]), width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,.08)",
    ), row=1, col=1)
    fig_curve.add_vline(
        x=ch_avg, line=dict(color="#EF4444", width=2, dash="dash"),
        annotation_text="Current avg spend", annotation_position="top right",
        row=1, col=1,
    )
    if "mroi_like" in cdf.columns:
        fig_curve.add_trace(go.Scatter(
            x=cdf[xcol], y=cdf["mroi_like"], name="mROI",
            line=dict(color="#F59E0B", width=2),
        ), row=2, col=1)
        fig_curve.add_hline(y=0, line=dict(color="#CBD5E1", width=1, dash="dot"), row=2, col=1)

    fig_curve.update_layout(
        template="plotly_dark", showlegend=False, hovermode="x unified",
        margin=dict(t=40, b=10, l=0, r=0
    ),
    )
    fig_curve.update_xaxes(title_text="Avg Weekly Spend ($)", row=2, col=1)
    st.plotly_chart(fig_curve, use_container_width=True)

    # ── All-channel saturation grid ───────────────────────────────────────────
    st.markdown('<p class="sec-header">Saturation Status — All Channels</p>', unsafe_allow_html=True)

    sat_cols = st.columns(min(n_channels, 5))
    for i, ch in enumerate(channels_list):
        ch_sp = float(pivot_spend[ch].mean()) if ch in pivot_spend else 0.0
        p2 = np.linspace(max(ch_sp * 0.1, 1), ch_sp * 2.5, 20)
        try:
            c2  = mmm.response_curve_advanced(df, ch, p2)
            mc  = float(c2.iloc[(c2["avg_weekly_spend"] - ch_sp).abs().argsort()[:1]]["mroi_like"].values[0])
            mm  = c2["mroi_like"].dropna().max()
            sp2 = max(0.0, min(1.0, 1.0 - (mc / mm if mm > 0 else 0.5)))
        except Exception:
            sp2 = 0.5

        p_css = "pill-under" if sp2 < 0.4 else ("pill-sat" if sp2 > 0.75 else "pill-opt")
        p_lbl = "Underinvested" if sp2 < 0.4 else ("Saturated" if sp2 > 0.75 else "Optimal")

        with sat_cols[i % len(sat_cols)]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="label">{ch}</div>
                <div class="value" style="font-size:1.1rem">{sp2:.0%}</div>
                <div class="sub"><span class="{p_css}">{p_lbl}</span></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4 — OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
with tab_optimizer:

    # ── Auto-set weekly budget default from historical data ───────────────────
    hist_weekly_total = float(pivot_spend.mean().sum())
    # Show the historical avg as a reference alongside the user input
    st.markdown('<p class="sec-header">Model-Driven Budget Recommendation</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight">
        The optimizer uses the MMM response curves to find a high-performing budget
        allocation — not what the platforms recommend, and not equal-weight distribution.
        <br><br>
        💡 <strong>Budget is weekly</strong> — the same unit as your CSV spend data.
        Your historical average is <strong>${hist_weekly_total:,.0f}/week</strong>.
        Adjust the weekly budget in the sidebar to explore different scenarios.
        <br><br>
        <small style="color:#8B9DB0">
        <strong>Methodology note:</strong> The allocation is found via a greedy iterative algorithm
        (bucket-fill) — a practical approximation, not a mathematical global optimum. The
        projected lead gain uses <code>project_leads()</code>, which runs the full model with all
        channels at the recommended allocation simultaneously. Results should be treated as a
        directional recommendation, not a precise prediction.
        </small>
    </div>
    """, unsafe_allow_html=True)

    # ── Conceptual explainer: Attribution Gap ≠ Saturation ───────────────────
    with st.expander("❓ Why might a channel be Overvalued AND Underinvested at the same time?"):
        st.markdown("""
        These two metrics measure **completely different things** and are not contradictory.

        | | **Underinvested** | **Saturated** |
        |---|---|---|
        | **Overvalued** | ⚠️ Platform overcredits it, but the model finds room to grow. The channel works — you're just not seeing the full picture in your reporting. | 🔴 Platform overcredits it AND you're past the efficient range. Cut budget and fix your reporting. |
        | **Undervalued** | ✅ Platform ignores it AND there's room to scale. Strong signal to increase investment. | 🟡 Platform ignores it but you've already hit the ceiling. Maintain spend but update your reporting. |

        **Attribution Gap** answers: *"Is the platform's reported contribution an accurate reflection of reality?"*
        → It's about **credit** — is this channel getting the right share of attribution?

        **Saturation** answers: *"If I spend more on this channel, will I get proportional returns?"*
        → It's about **marginal efficiency** — is there room to scale?

        A channel can be overcredited (Overvalued) by last-click **and** still have spend headroom (Underinvested)
        because the MMM model finds a real statistical relationship between that spend and leads —
        even if the platform was exaggerating the effect in its own reporting.
        """)

    # ── Budget range check ────────────────────────────────────────────────────
    _budget_ratio = float(total_budget) / max(hist_weekly_total, 1.0)
    if _budget_ratio < 0.5:
        st.warning(
            f"⚠️ **Budget is {_budget_ratio:.0%} of your historical average** "
            f"(${hist_weekly_total:,.0f}/wk). At this level the optimizer is constrained "
            f"to a much smaller spend range than the model was trained on — recommendations "
            f"will be directional but less reliable. Consider testing closer to your historical average."
        )
    elif _budget_ratio > 2.0:
        st.warning(
            f"⚠️ **Budget is {_budget_ratio:.1f}× your historical average** "
            f"(${hist_weekly_total:,.0f}/wk). The model extrapolates beyond observed spend levels, "
            f"which increases uncertainty. Response curves assume the same saturation shape holds "
            f"at higher spend — verify with a controlled experiment before committing."
        )

    with st.spinner("Running optimizer…"):
        try:
            weekly_budget = float(total_budget)

            if optimizer_mode.startswith("Advanced"):
                opt = optimize_budget_advanced(
                    model=mmm, df=df,
                    total_budget=weekly_budget,
                    min_pct=float(min_pct), max_pct=float(max_pct),
                    step=float(step), points=40,
                )
            else:
                opt = optimize_budget_mvp(
                    model=mmm,
                    df_media_weekly=df[["week", "channel", "spend"]],
                    total_budget=weekly_budget,
                    min_pct=float(min_pct), max_pct=float(max_pct),
                    step=float(step),
                )

            # All columns now use readable names — no more wk_rec_col mapping
            rec_weekly    = opt["Recommended ($/wk)"].sum()
            cur_weekly    = opt["Current ($/wk)"].sum()
            total_gain    = opt.attrs.get("total_lead_gain", 0.0)
            rec_leads_tot = opt.attrs.get("recommended_leads_total", 0.0)
            cur_leads_tot = opt.attrs.get("current_leads_total", 0.0)
            n_opt_weeks   = opt.attrs.get("n_weeks", df["week"].nunique())

            gain_color = "#10B981" if total_gain >= 0 else "#EF4444"

            if total_gain < 0:
                st.markdown('''<div class="insight warn">
                ⚠️ <strong>The optimizer returned a negative projected lead gain.</strong>
                This can happen for several reasons: the weekly budget entered is below
                your historical average (try matching it above); the min/max allocation
                constraints force a channel mix that the model considers less efficient
                than your current one; or the model has high uncertainty on some channels
                (check the Confidence Score and CI width in Attribution Gap).
                The optimizer direction is still valid — it reflects the model's best
                estimate of how to distribute <em>this</em> budget level.
                </div>''', unsafe_allow_html=True)

            # ── KPI cards ──────────────────────────────────────────────────
            o1, o2, o3, o4 = st.columns(4)
            kpi_data_opt = [
                (o1, "Recommended Weekly",   f"${rec_weekly:,.0f}",               "new allocation"),
                (o2, "Current Weekly",       f"${cur_weekly:,.0f}",               "historical average"),
                (o3, "Budget Change",        f"${rec_weekly - cur_weekly:+,.0f}",
                                             f"{(rec_weekly-cur_weekly)/max(cur_weekly,1):.1%} vs current"),
                (o4, "Est. Lead Gain",       f"{total_gain:+,.0f}",
                                             f"over {n_opt_weeks} weeks vs current mix"),
            ]
            for col, label, val, sub in kpi_data_opt:
                with col:
                    val_style = f"color:{gain_color}" if label == "Est. Lead Gain" else ""
                    st.markdown(f"""<div class="kpi-card">
                        <div class="label">{label}</div>
                        <div class="value" style="{val_style}">{val}</div>
                        <div class="sub">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.caption(
                "Budget figures are **weekly**. Multiply ×4.3 for monthly, ×52 for annual. "
                "Lead gain is computed via the full model with all channels applied simultaneously."
            )

            # Confidence-aware disclaimer
            _opt_ci_note = ""
            if _ci_available and not boot_ci.empty:
                _max_ci = boot_ci["ci_width"].max()
                if _max_ci > 0.10:
                    _opt_ci_note = (f"⚠️ Attribution estimates have wide confidence intervals (up to "
                                    f"{_max_ci*100:.0f} pp). Budget recommendations are directional — "
                                    f"the direction of reallocation is reliable, but exact dollar amounts "
                                    f"carry model uncertainty.")
                else:
                    _opt_ci_note = (f"✅ Attribution estimates are robust (max CI width {_max_ci*100:.1f} pp). "
                                    f"Budget recommendations reflect stable model estimates.")
            else:
                _opt_ci_note = (f"📊 Confidence intervals require {MIN_WEEKS_FOR_CI}+ weeks. "
                                f"With {n_weeks} weeks, treat recommendations as directional guidance.")
            st.markdown(
                f'''<div style="background:#F8FAFC;border-left:4px solid #6366F1;padding:.65rem 1rem;
                border-radius:4px;font-size:.80rem;color:#8B9DB0;margin:.5rem 0">{_opt_ci_note}</div>''',
                unsafe_allow_html=True)

            # ── Projected leads comparison ─────────────────────────────────
            st.markdown("")
            leads_cmp = pd.DataFrame({
                "Scenario": ["Current allocation", "Recommended allocation"],
                "Projected Leads": [cur_leads_tot, rec_leads_tot],
            })
            fig_leads = px.bar(
                leads_cmp, x="Scenario", y="Projected Leads", text="Projected Leads",
                color="Scenario",
                color_discrete_map={"Current allocation": "#CBD5E1",
                                    "Recommended allocation": "#6366F1"},
                template="plotly_dark",
                labels={"Projected Leads": f"Total leads over {n_opt_weeks} weeks"},
            )
            fig_leads.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
            fig_leads.update_layout(showlegend=False, margin=dict(t=30,b=10,l=0,r=0
    ),
                                    yaxis=dict(range=[0, max(rec_leads_tot, cur_leads_tot) * 1.18]))
            st.plotly_chart(fig_leads, use_container_width=True)

            # ── Current vs Recommended spend by channel ────────────────────
            st.markdown('<p class="sec-header">Spend by Channel — Current vs Recommended</p>',
                        unsafe_allow_html=True)
            cmp = pd.DataFrame({
                "Channel":     opt["Channel"],
                "Current":     opt["Current ($/wk)"],
                "Recommended": opt["Recommended ($/wk)"],
            }).melt(id_vars="Channel", var_name="Allocation", value_name="Weekly Spend ($)")
            fig_opt = px.bar(
                cmp, x="Channel", y="Weekly Spend ($)", color="Allocation", barmode="group",
                color_discrete_map={"Current": "#CBD5E1", "Recommended": "#6366F1"},
                template="plotly_dark", labels={"Channel": ""},
            )
            fig_opt.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
                margin=dict(t=30, b=10, l=0, r=0),
            )
            st.plotly_chart(fig_opt, use_container_width=True)

            # ── Budget shift waterfall ─────────────────────────────────────
            st.markdown('<p class="sec-header">Budget Reallocation by Channel</p>',
                        unsafe_allow_html=True)
            opt_sorted = opt.sort_values("Change ($/wk)")
            bar_colors = ["#10B981" if v >= 0 else "#EF4444" for v in opt_sorted["Change ($/wk)"]]
            fig_shift = px.bar(
                opt_sorted, x="Change ($/wk)", y="Channel", orientation="h",
                template="plotly_dark",
                labels={"Change ($/wk)": "Weekly Budget Change ($)", "Channel": ""},
                text=opt_sorted["Change ($/wk)"].map("${:+,.0f}".format),
            )
            fig_shift.update_traces(marker_color=bar_colors, textposition="outside")
            fig_shift.update_layout(showlegend=False, margin=dict(t=10, b=10, l=0, r=0
    ))
            st.plotly_chart(fig_shift, use_container_width=True)

            # ── mROI comparison chart ──────────────────────────────────────
            st.markdown('<p class="sec-header">Marginal ROI — Current vs Recommended Spend Level</p>',
                        unsafe_allow_html=True)
            st.caption(
                "mROI = incremental leads generated per additional $1 spent at that spend level. "
                "The optimizer moves budget from low-mROI channels to high-mROI channels."
            )
            mroi_cmp = pd.DataFrame({
                "Channel":     list(opt["Channel"]) * 2,
                "Spend Level": ["Current"] * len(opt) + ["Recommended"] * len(opt),
                "mROI":        list(opt["mROI at Current"]) + list(opt["mROI at Recommended"]),
            })
            fig_mroi = px.bar(
                mroi_cmp, x="Channel", y="mROI", color="Spend Level", barmode="group",
                color_discrete_map={"Current": "#CBD5E1", "Recommended": "#6366F1"},
                template="plotly_dark",
                labels={"mROI": "Leads per $ (marginal)", "Channel": ""},
            )
            fig_mroi.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
                margin=dict(t=30, b=10, l=0, r=0),
            )
            st.plotly_chart(fig_mroi, use_container_width=True)

            # ── Full recommendation table (clean) ──────────────────────────
            with st.expander("Full recommendation table"):
                disp = opt[["Channel", "Current ($/wk)", "Recommended ($/wk)",
                             "Change ($/wk)", "Change (%)",
                             "mROI at Current", "mROI at Recommended",
                             "Recommended Share"]].copy()
                disp["Current ($/wk)"]     = disp["Current ($/wk)"].map("${:,.0f}".format)
                disp["Recommended ($/wk)"] = disp["Recommended ($/wk)"].map("${:,.0f}".format)
                disp["Change ($/wk)"]      = disp["Change ($/wk)"].map("${:+,.0f}".format)
                disp["Change (%)"]         = disp["Change (%)"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
                disp["mROI at Current"]    = disp["mROI at Current"].map("{:.4f}".format)
                disp["mROI at Recommended"]= disp["mROI at Recommended"].map("{:.4f}".format)
                disp["Recommended Share"]  = disp["Recommended Share"].map("{:.1%}".format)
                st.dataframe(disp, use_container_width=True, hide_index=True)

            st.download_button("⬇ Download Recommendation CSV",
                               opt.to_csv(index=False).encode(),
                               "budget_recommendation.csv", "text/csv")

        except Exception as exc:
            st.error("Optimizer failed.")
            st.exception(exc)
            opt = None


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5 — SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
with tab_scenarios:

    st.markdown('<p class="sec-header">What-If Budget Simulator</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight">
        Adjust the weekly budget per channel and instantly see the projected
        incremental impact. Use this to model budget cuts, reallocation strategies,
        or the effect of shifting investment toward undervalued channels.
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "📊 **Directional planning tool.** "
        "The total lead projection uses the full MMM model with all channels applied simultaneously — "
        "this is the most reliable output. Per-channel lead estimates use standalone response curves "
        "(one channel varied at a time) and should be read as directional only, not precise forecasts. "
        "All projections assume market conditions similar to your historical data.",
        icon="ℹ️",
    )

    baseline_avg        = pivot_spend.mean(axis=0).reindex(channels_list).fillna(0.0)
    total_baseline_wk   = float(baseline_avg.sum())

    preset = st.selectbox("Start from a preset", [
        "Current allocation",
        "+10% Total budget",
        "-15% Total budget",
        "Shift budget to most undervalued channel",
        "Efficiency cut: -15%, focus on best ROAS",
    ])

    p_alloc = {ch: float(baseline_avg[ch]) for ch in channels_list}

    if preset == "+10% Total budget":
        p_alloc = {ch: v * 1.10 for ch, v in p_alloc.items()}
    elif preset == "-15% Total budget":
        p_alloc = {ch: v * 0.85 for ch, v in p_alloc.items()}
    elif preset == "Shift budget to most undervalued channel":
        if lc_available and not attr_gap.empty:
            undv = attr_gap[attr_gap["bias_label"] == "Undervalued"]
            if not undv.empty:
                top_undv = undv.sort_values("gap_points", ascending=False).iloc[0]["channel"]
                freed = total_baseline_wk * 0.10
                p_alloc = {ch: v * 0.90 for ch, v in p_alloc.items()}
                p_alloc[top_undv] = p_alloc.get(top_undv, 0) + freed
    elif preset == "Efficiency cut: -15%, focus on best ROAS":
        roas_r = chan_contrib_df.set_index("channel")["est_roas"].to_dict()
        med    = chan_contrib_df["est_roas"].median()
        p_alloc = {
            ch: float(baseline_avg[ch]) * (0.95 if roas_r.get(ch, 1) >= med else 0.78)
            for ch in channels_list
        }

    st.markdown("**Weekly budget per channel:**")
    slider_alloc = {}
    slider_cols  = st.columns(min(n_channels, 3))
    for i, ch in enumerate(channels_list):
        dv  = float(round(p_alloc[ch], 2))
        mx  = float(max(float(baseline_avg[ch]) * 3, dv * 1.5, 500))
        with slider_cols[i % len(slider_cols)]:
            slider_alloc[ch] = st.slider(ch, 0.0, mx, dv,
                                         max(1.0, round(mx / 100, 2)), key=f"sc_{ch}")

    # ── Compute scenario ────────────────────────────────────────────────────
    # Total leads: project_leads() runs the full model with ALL channels at the
    # new allocation simultaneously — same methodology as the Optimizer.
    # Per-channel breakdown: standalone curves (one channel at a time, others fixed)
    # — useful for showing direction, but note they don't sum to the full-model total.
    sc_total_leads   = mmm.project_leads(df, slider_alloc)
    sc_baseline_leads = mmm.project_leads(df, {ch: float(baseline_avg[ch]) for ch in channels_list})
    sc_net_gain      = sc_total_leads - sc_baseline_leads

    sc_rows = []
    for ch in channels_list:
        pt = np.array([slider_alloc[ch]])
        try:
            cdf2 = mmm.response_curve_advanced(df, ch, pt)
            inc  = float(cdf2["incremental_total_leads"].iloc[0])
        except Exception:
            inc = 0.0
        sc_rows.append({
            "channel":                ch,
            "baseline_weekly ($)":    float(baseline_avg[ch]),
            "scenario_weekly ($)":    float(slider_alloc[ch]),
            "delta ($)":              float(slider_alloc[ch]) - float(baseline_avg[ch]),
            "delta (%)":              (float(slider_alloc[ch]) - float(baseline_avg[ch])) / max(float(baseline_avg[ch]), 1e-9),
            "standalone leads (dir.)": inc,
        })

    sc_df       = pd.DataFrame(sc_rows)
    sc_total_wk = sc_df["scenario_weekly ($)"].sum()
    sc_delta    = sc_df["delta ($)"].sum()

    s1, s2, s3, s4 = st.columns(4)
    _gain_color = "#10B981" if sc_net_gain >= 0 else "#EF4444"
    for col, label, val, sub in [
        (s1, "Baseline Weekly",    f"${total_baseline_wk:,.0f}", "current mix"),
        (s2, "Scenario Weekly",    f"${sc_total_wk:,.0f}",        f"{sc_delta:+,.0f} delta"),
        (s3, "Budget Delta",       f"${sc_delta:+,.0f}",          f"{sc_delta/total_baseline_wk:+.1%}" if total_baseline_wk > 0 else ""),
        (s4, "Est. Lead Change",   f"{sc_net_gain:+,.0f}",        "full-model projection"),
    ]:
        with col:
            _val_style = f"color:{_gain_color}" if label == "Est. Lead Change" else ""
            st.markdown(f"""
            <div class="kpi-card">
                <div class="label">{label}</div>
                <div class="value" style="{_val_style}">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        '<div style="background:#F1F5F9;border-left:4px solid #6366F1;padding:.55rem 1rem;'
        'border-radius:4px;font-size:.79rem;color:#8B9DB0;margin-bottom:.5rem">'
        '📊 <strong>Est. Lead Change</strong> uses the same full-model projection as the Optimizer '
        '(all channels applied simultaneously). The per-channel breakdown in the table below '
        'uses standalone curves — directional only.</div>',
        unsafe_allow_html=True)

    st.markdown('<p class="sec-header">Baseline vs Scenario</p>', unsafe_allow_html=True)

    sc_chart = sc_df[["channel", "baseline_weekly ($)", "scenario_weekly ($)"]].melt(
        id_vars="channel", var_name="Type", value_name="Weekly Spend"
    )
    sc_chart["Type"] = sc_chart["Type"].map({
        "baseline_weekly ($)": "Baseline", "scenario_weekly ($)": "Scenario"
    })
    fig_sc = px.bar(
        sc_chart, x="channel", y="Weekly Spend", color="Type", barmode="group",
        color_discrete_map={"Baseline": "#CBD5E1", "Scenario": "#6366F1"},
        template="plotly_dark",
    )
    fig_sc.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
        margin=dict(t=30, b=10, l=0, r=0),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("Scenario detail table"):
        fmt = {
            "baseline_weekly ($)":      "${:,.2f}",
            "scenario_weekly ($)":      "${:,.2f}",
            "delta ($)":                "${:+,.2f}",
            "delta (%)":                "{:+.1%}",
            "standalone leads (dir.)":  "{:,.1f}",
        }
        disp = sc_df.copy()
        for c, f in fmt.items():
            disp[c] = disp[c].map(f.format)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(
            "Standalone leads are per-channel directional estimates (one channel varied, "
            "others fixed). They do not sum to Est. Lead Change, which is a full-model projection."
        )

    st.download_button("⬇ Download Scenario CSV",
                       sc_df.to_csv(index=False).encode(), "scenario.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 6 — DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_diagnostics:

    _diag_level = feature_level(_account, "diagnostics")

    if not _diag_level:
        _msg = get_upgrade_msg("diagnostics")
        st.markdown(f'<p class="sec-header">🔒 {_msg["title"]}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight">{_msg["body"]}<br><br>'
            f'<strong>Available on Starter ($149/mo) and above.</strong></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    if _diag_level == "basic":
        # Starter: show only R² and MAPE, then block the rest
        st.markdown('<p class="sec-header">Model Accuracy</p>', unsafe_allow_html=True)
        _c1, _c2 = st.columns(2)
        with _c1:
            st.metric("Training R²", f"{fit['r2']:.3f}")
        with _c2:
            _mape_val = backtest_df["mape"].mean() if not backtest_df.empty else None
            st.metric("Backtest MAPE", f"{_mape_val:.1%}" if _mape_val else "—")
        st.markdown("---")
        _msg = get_upgrade_msg("confidence_score_full")
        st.markdown(
            f'<div class="insight">🔒 <strong>{_msg["title"]}</strong> — {_msg["body"]}<br><br>'
            f'Upgrade to <strong>Professional</strong> for the full diagnostics suite.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Full diagnostics — Professional+ ─────────────────────────────────────
    # ── Confidence score ──────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Model Confidence Score</p>', unsafe_allow_html=True)

    badge_css = {"HIGH": "badge-under", "MEDIUM": "badge-fair", "LOW": "badge-over"}[conf["label"]]
    st.markdown(f"""
    <div class="kpi-card" style="text-align:left;padding:1.5rem 2rem">
        <span class="gap-badge {badge_css}" style="font-size:.8rem">{conf['label']}</span>
        <span style="font-size:2.4rem;font-weight:800;color:#EEEEF5;margin-left:.75rem">
            {conf['score']}
            <span style="font-size:1rem;font-weight:400;color:#8B9DB0">/ 100</span>
        </span>
        <div style="font-size:.82rem;color:#8B9DB0;margin-top:.6rem">
            Composite score: data volume · backtest accuracy · coefficient stability · collinearity
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    bk = conf["breakdown"]

    # Build labels and values — include neg_penalty only if it applies
    _bk_labels = ["Data Volume", "Backtest Accuracy", "Coef. Stability", "Collinearity Penalty"]
    _bk_values = [bk["data_score"], bk["acc_score"], bk["stab_points"], bk["col_penalty"]]
    _bk_colors = {
        "Data Volume": "#6366F1", "Backtest Accuracy": "#06B6D4",
        "Coef. Stability": "#10B981", "Collinearity Penalty": "#EF4444",
    }
    if bk.get("neg_penalty", 0) != 0:
        _bk_labels.append("Negative Coef Penalty")
        _bk_values.append(bk["neg_penalty"])   # already negative in breakdown dict
        _bk_colors["Negative Coef Penalty"] = "#F59E0B"

    fig_sc2 = px.bar(
        x=_bk_labels, y=_bk_values, color=_bk_labels,
        color_discrete_map=_bk_colors,
        text=[str(v) for v in _bk_values],
        template="plotly_dark", labels={"x": "", "y": "Points"},
    )
    fig_sc2.update_traces(textposition="outside")
    fig_sc2.update_layout(showlegend=False, margin=dict(t=20, b=10, l=0, r=0
    ))
    st.plotly_chart(fig_sc2, use_container_width=True)

    with st.expander("Score breakdown details"):
        for r in conf["reasons"]:
            st.write("· " + r)

    # ── Rolling backtest ──────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Rolling Backtest</p>', unsafe_allow_html=True)

    with st.expander(f"In-sample fit: R\u00b2 = {fit['r2']:.3f}  \u00b7  Training MAPE = {fit['mape']:.1%}  (training data only, not predictive)"):
        st.caption(
            "R\u00b2 measures variance explained on training data. A high value is necessary but not sufficient "
            "\u2014 models with many seasonality dummies can overfit. "
            "Use Backtest MAPE as the primary accuracy signal.")

    if backtest_df.empty:
        st.warning("Not enough weekly data for rolling backtest (minimum ~20 weeks recommended).")
    else:
        avg_mape = backtest_df["mape"].mean()
        d1, d2, d3 = st.columns(3)
        for col, label, val, sub in [
            (d1, "Avg Backtest MAPE", f"{avg_mape:.1%}",
             "✅ Strong" if avg_mape < 0.10 else "🟡 Acceptable" if avg_mape < 0.20 else "⚠️ Review"),
            (d2, "Backtest Windows",  str(len(backtest_df)), "4-week test windows"),
            (d3, "Params (auto)" if auto_mode else "Params (manual)",
             f"θ={theta:.2f} α={alpha:.2f} γ={hp.gamma_scale:.1f} λ={ridge_alpha:.1f}",
             "adstock · sat · half-sat · ridge"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="label">{label}</div>
                    <div class="value">{val}</div>
                    <div class="sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        fig_bt = px.line(
            backtest_df, x="test_start", y="mape", markers=True,
            template="plotly_dark", color_discrete_sequence=["#6366F1"],
            labels={"mape": "MAPE", "test_start": "Test Window Start"},
        )
        # Naïve baseline: always predict the training mean
        from utils import weekly_outcome_series as _wos
        _y_all   = _wos(df).sort_index()
        _y_mean  = float(_y_all.mean())
        _naive   = float(np.mean(np.abs(_y_all - _y_mean) / np.maximum(_y_all, 1e-6)))
        _cv      = float(_y_all.std() / max(_y_all.mean(), 1e-6))

        fig_bt.add_hline(y=0.10, line_dash="dash", line_color="#10B981",
                         annotation_text="10% — strong")
        fig_bt.add_hline(y=0.20, line_dash="dash", line_color="#F59E0B",
                         annotation_text="20% — acceptable")
        if _cv >= 0.15:
            # Naïve baseline only meaningful when leads have high variance
            fig_bt.add_hline(y=_naive, line_dash="dot", line_color="#94A3B8",
                             annotation_text=f"Naïve mean baseline {_naive:.0%}")
        fig_bt.update_yaxes(tickformat=".1%")
        fig_bt.update_layout(margin=dict(t=20, b=10, l=0, r=0
    ))
        st.plotly_chart(fig_bt, use_container_width=True)

        if _cv < 0.15:
            # Low-variance leads: backtest vs naive isn't a meaningful signal
            _qual = "✅ Strong" if avg_mape < 0.10 else "🟡 Acceptable" if avg_mape < 0.20 else "⚠️ High"
            st.caption(
                f"{_qual} backtest MAPE of {avg_mape:.0%}. "
                f"Your leads signal has low variance (CV={_cv:.0%}) — the model's primary quality "
                f"signals are in-sample fit and CI width, not naive baseline comparison. "
                f"See Confidence Score breakdown."
            )
        elif avg_mape > _naive:
            st.markdown(
                f'''<div class="insight warn">⚠️ <strong>Backtest MAPE ({avg_mape:.0%}) exceeds naïve baseline ({_naive:.0%}).</strong>
                Common causes: fewer than 52 weeks of data, high spend concentration in one channel,
                or structural changes mid-period. Directional insights may still be valid —
                check the Attribution Gap stability and CI width in the Bootstrap section above.
                </div>''', unsafe_allow_html=True)
        else:
            st.caption(
                f"✅ Model beats naïve baseline ({_naive:.0%}) by "
                f"{(_naive - avg_mape)*100:.0f}pp — generalises beyond mean prediction."
            )

    _neg_chs = fit.get("negative_coef_channels", [])
    if _neg_chs:
        st.markdown(f'''<div class="insight warn">\u26a0\ufe0f <strong>Channels with constrained coefficients: {", ".join(_neg_chs)}</strong><br>The model could not find a positive statistical contribution for these channels given the current data. High collinearity is the most common cause. Consider more data or reviewing channel groupings.</div>''', unsafe_allow_html=True)

    if auto_mode and "leaderboard" in dir():
        # Winner metadata: explain why this combination won
        _runner_up_mape = float(leaderboard.iloc[1]["backtest_mape_mean"]) if len(leaderboard) > 1 else None
        _winner_mape    = float(best["backtest_mape_mean"])
        _gap_str        = (f" — **{(_runner_up_mape - _winner_mape):.2%} better** than the runner-up "
                           f"(θ={leaderboard.iloc[1]['theta']:.1f} · "
                           f"α={leaderboard.iloc[1]['alpha']:.1f} · "
                           f"λ={leaderboard.iloc[1]['ridge_alpha']:.0f} · "
                           f"γ={leaderboard.iloc[1]['gamma_scale']:.1f}, "
                           f"MAPE {_runner_up_mape:.2%})") if _runner_up_mape else ""
        st.markdown(
            f"**Winning combination:** "
            f"θ = {best['theta']:.2f} · α = {best['alpha']:.2f} · "
            f"λ = {best['ridge_alpha']:.1f} · γ-scale = {best['gamma_scale']:.2f} "
            f"→ Backtest MAPE **{_winner_mape:.2%}**{_gap_str}."
        )
        st.caption(
            "Selection criterion: lowest mean walk-forward backtest MAPE across rolling windows. "
            "The winner balances adstock carry-over (θ), saturation steepness (α), regularisation "
            "strength (λ), and half-saturation scale (γ). A lower MAPE means the model's "
            "out-of-sample predictions were closer to actual lead counts — not just a better fit "
            "on training data."
        )
        with st.expander("Full auto-tune leaderboard (top 15)"):
            lb = leaderboard.head(15).copy()
            lb["backtest_mape_mean"] = lb["backtest_mape_mean"].map("{:.2%}".format)
            lb.insert(0, "Rank", range(1, len(lb) + 1))
            st.dataframe(lb, use_container_width=True, hide_index=True)

    # ── Tracking diagnostics ──────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Tracking Diagnostics</p>', unsafe_allow_html=True)

    if tracking["n_alerts"] == 0:
        st.success("✅ No structural tracking breaks detected.")
    else:
        st.warning(f"⚠️ {tracking['n_alerts']} potential tracking break(s) detected.")
        a_df = pd.DataFrame(tracking["alerts"])
        a_df["leads_change"] = a_df["leads_change"].map("{:+.1%}".format)
        a_df["spend_change"] = a_df["spend_change"].map("{:+.1%}".format)
        st.dataframe(a_df, use_container_width=True, hide_index=True)

    st.caption(
        f"Weeks where leads moved >{tracking['thresholds_used']['leads_change_threshold']:.0%} "
        f"while spend was stable (<{tracking['thresholds_used']['spend_change_threshold']:.0%} change). "
        "These may indicate pixel failures, CRM imports, or external demand shocks."
    )

    # ── Model parameters ─────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Model Parameters</p>', unsafe_allow_html=True)
    st.caption(
        "AdCrux uses two key transforms per channel: **adstock decay (θ)** models how ad "
        "impressions carry over into future weeks — higher θ means longer memory. "
        "**Hill saturation (α)** controls how quickly returns diminish — higher α means steeper "
        "drop-off."
    )
    st.info(
        "**Active mode: Global shared parameters** — θ, α, γ-scale and λ are a single set "
        "optimised across all channels via walk-forward backtest MAPE. This is the safest "
        "approach for datasets under 80 weeks: it avoids overfitting and ensures parameters "
        "generalise to unseen data. "
        "Per-channel tuning (a different parameter set per channel) becomes statistically "
        "reliable only with 80+ weeks of data per channel, and is not active in this version.",
        icon="ℹ️",
    )
    _ch_params = fit.get("channel_params", {})
    if _ch_params:
        # Show global params + per-channel half-saturation point (gamma = gamma_scale × mean adstock)
        _params_rows = []
        for ch, p in _ch_params.items():
            _spend_sh  = float(spend_totals.get(ch, 0) / max(total_spend, 1)) * 100
            _half_sat  = mmm.channel_gammas_.get(ch, None)
            _params_rows.append({
                "Channel":       ch,
                "Adstock θ":     f"{p['theta']:.2f}",
                "Hill α":        f"{p['alpha']:.2f}",
                "γ-scale":       f"{hp.gamma_scale:.2f}",
                "Half-sat pt":   f"${_half_sat:,.0f}/wk" if _half_sat else "—",
                "Spend Share":   f"{_spend_sh:.1f}%",
                "Memory":        "Long (>2 wks)" if p['theta'] >= 0.5 else "Short (≤2 wks)",
                "Saturation":    "Steep" if p['alpha'] >= 1.8 else "Gradual",
            })
        st.dataframe(pd.DataFrame(_params_rows), use_container_width=True, hide_index=True)
        st.caption(
            "θ (adstock) · α (Hill steepness) · γ-scale (half-saturation multiplier) — "
            "all shared across channels, tuned via walk-forward backtest MAPE. "
            "Half-sat pt = γ-scale × mean(adstocked spend): the weekly spend at which "
            "a channel reaches 50% of its maximum response. "
            f"λ = {hp.ridge_alpha:.1f} (ridge regularisation)."
        )

    # ── Full data quality report ───────────────────────────────────────────────
    st.markdown('<p class="sec-header">Data Quality Assessment</p>', unsafe_allow_html=True)

    _dqr_badge = {"GOOD": ("✅ GOOD", "#10B981"), "FAIR": ("🟡 FAIR", "#F59E0B"), "POOR": ("🔴 POOR", "#EF4444")}
    _badge_txt, _badge_col = _dqr_badge[dqr["overall"]]
    st.markdown(
        f'''<span style="background:{_badge_col};color:white;padding:.25rem .75rem;
        border-radius:12px;font-size:.8rem;font-weight:700">{_badge_txt}</span>
        &nbsp;&nbsp;<span style="font-size:.85rem;color:#8B9DB0">
        {dqr["n_weeks"]} weeks · {dqr["n_channels"]} channels · 
        ${dqr["total_spend"]:,.0f} total spend · 
        max collinearity |r|={dqr["max_corr"]:.2f}</span>''',
        unsafe_allow_html=True)
    st.markdown("")

    _dq1, _dq2 = st.columns(2)
    with _dq1:
        st.markdown("**Strengths**")
        for s in dqr["strengths"]:
            st.markdown(s)
    with _dq2:
        st.markdown("**Watch points**")
        for w in dqr["warnings"] + dqr["issues"]:
            st.markdown(w)
        if not dqr["warnings"] and not dqr["issues"]:
            st.markdown("None — data looks clean ✅")

    # Spend coverage heatmap (which channels active in which weeks)
    with st.expander("Channel coverage by week"):
        _cov_pivot = (
            df[df["spend"] > 0]
            .groupby(["week","channel"])["spend"]
            .sum().unstack(fill_value=0)
        )
        _cov_active = (_cov_pivot > 0).astype(int)
        fig_cov = go.Figure(go.Heatmap(
            z=_cov_active.T.values,
            x=[str(w.date()) for w in _cov_active.index],
            y=_cov_active.columns.tolist(),
            colorscale=[[0,"#F1F5F9"],[1,"#6366F1"]],
            showscale=False,
            hovertemplate="%{y} · %{x}<extra></extra>",
        ))
        fig_cov.update_layout(
            template="plotly_dark",
            margin=dict(t=10, b=10, l=0, r=0
    ), height=180,
            xaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig_cov, use_container_width=True)
        st.caption("Purple = channel active that week · White = zero spend")

    # ── Collinearity heatmap ──────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Spend Collinearity — Heuristic</p>', unsafe_allow_html=True)

    col_res = compute_collinearity(df)
    if not col_res["corr_matrix"].empty:
        cm = col_res["corr_matrix"]
        fig_heat = go.Figure(go.Heatmap(
            z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(cm.values, 2), texttemplate="%{text}",
            colorbar=dict(title="r"),
        ))
        fig_heat.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=0, r=0
    ))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption(
            "Correlation of week-over-week % spend changes. This is a **heuristic** — "
            "not a definitive collinearity test. Channels that are always ramped up and "
            "down together (on/off at the same time) may appear less correlated here than "
            "they truly are, because the % change calculation can mask that pattern. "
            "High |r| > 0.80 is a reliable warning sign. Low |r| is encouraging but not a "
            "guarantee that the model can fully separate channel contributions."
        )

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("Raw data preview"):
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)
