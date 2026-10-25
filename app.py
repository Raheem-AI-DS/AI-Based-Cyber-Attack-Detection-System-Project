"""
AI-Based Cyber Attack Detection and Hacking Risk Prediction System
===================================================================
app.py  -  Interactive dashboard (Streamlit)

Run:
    pip install "streamlit>=1.50" plotly pandas numpy psutil reportlab
    streamlit run app.py

Model contract (to be implemented in Model.py / model.py in the next step):
    predict_attack(features: dict) -> dict
        {
          "attack_type":   str,                 # one of CLASSES below
          "confidence":    float,               # 0..1
          "probabilities": {class_name: float}, # sums to ~1
          "risk_score":    float,               # 0..100 hacking-risk score
        }
Until the model exists the app runs in DEMO mode with a rule-based engine,
so the whole UI, live monitor and PDF report can be tested right now.
"""

from __future__ import annotations

import io
import ipaddress
import json
import platform
import random
import socket
import time
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="NIDS | AI Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
CLASSES = ["Normal", "DDoS Attack", "Port Scan", "SQL Injection", "XSS Attack", "Brute Force"]

ATTACK_COLORS = {
    "Normal": "#22c55e",
    "DDoS Attack": "#2563eb",
    "Port Scan": "#f97316",
    "SQL Injection": "#ef4444",
    "XSS Attack": "#06b6d4",
    "Brute Force": "#8b5cf6",
    "Others": "#94a3b8",
}

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]

RECOMMENDATIONS = {
    "Normal": [
        "No malicious pattern detected. Continue routine monitoring.",
        "Keep OS, drivers and applications patched.",
        "Review firewall and MFA posture regularly.",
    ],
    "DDoS Attack": [
        "Enable rate limiting and SYN-cookie protection on edge devices.",
        "Engage upstream provider / CDN scrubbing for volumetric traffic.",
        "Blackhole or geo-block the top offending source ranges.",
    ],
    "Port Scan": [
        "Close every unused port and disable unneeded services.",
        "Enable IDS port-scan signatures and auto-block repeat scanners.",
        "Restrict management ports (SSH/RDP/SMB) to VPN or allow-listed IPs.",
    ],
    "SQL Injection": [
        "Use parameterised queries / prepared statements everywhere.",
        "Deploy a WAF with SQLi rule sets and validate all user input.",
        "Run database accounts with least privilege and audit recent queries.",
    ],
    "XSS Attack": [
        "Encode output and apply a strict Content-Security-Policy.",
        "Sanitise user-supplied HTML and set HttpOnly/SameSite cookies.",
        "Scan application inputs for stored and reflected injection points.",
    ],
    "Brute Force": [
        "Enforce account lockout, progressive delays and CAPTCHA.",
        "Require multi-factor authentication on all remote logins.",
        "Block repeat offenders and rotate any weak or reused credentials.",
    ],
}

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
:root{
  --bg:#0a0e1a; --panel:#0f1424; --panel2:#121a2e; --border:#1c2540;
  --text:#e5e9f5; --muted:#8892ad; --accent:#2f5bea; --green:#22c55e;
  --red:#ef4444; --orange:#f59e0b; --purple:#8b5cf6;
}
html, body, [data-testid="stAppViewContainer"], .stApp {background:var(--bg); color:var(--text);}
[data-testid="stHeader"]{background:transparent; height:0;}
#MainMenu, footer {visibility:hidden;}
.block-container{padding-top:1.4rem; padding-bottom:2rem; max-width:1500px;}

/* Sidebar */
[data-testid="stSidebar"]{background:#0b101d; border-right:1px solid var(--border);}
[data-testid="stSidebar"] .block-container{padding-top:1rem;}
.brand{display:flex; align-items:center; gap:.7rem; padding:.2rem .4rem 1rem .4rem;}
.brand .logo{width:34px; height:34px; border-radius:9px; display:flex; align-items:center;
  justify-content:center; background:linear-gradient(135deg,#3b2fa8,#2f5bea); font-size:1.1rem;}
.brand .name{font-weight:800; letter-spacing:.08em; font-size:1.15rem;}
.brand .sub{font-size:.62rem; color:var(--muted); letter-spacing:.12em; margin-top:-2px;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.25rem;}
[data-testid="stSidebar"] [role="radiogroup"] label{
  background:transparent; border-radius:8px; padding:.55rem .8rem; width:100%;
  color:var(--muted); transition:all .15s ease; border:1px solid transparent;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#131b30; color:var(--text);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:var(--accent); color:#fff; box-shadow:0 4px 14px rgba(47,91,234,.35);}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:.92rem; font-weight:500;}

.sys-status{position:fixed; bottom:18px; left:18px; font-size:.75rem; color:var(--muted); line-height:1.5;}
.dot{display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green);
  box-shadow:0 0 8px var(--green); margin-right:6px;}

/* Header */
.page-title{font-size:1.65rem; font-weight:700; margin:0; letter-spacing:.01em;}
.page-sub{color:var(--muted); font-size:.85rem; margin-top:.15rem;}
.chip{display:inline-block; padding:.35rem .8rem; border-radius:8px; background:var(--panel);
  border:1px solid var(--border); font-size:.8rem; color:var(--text);}
.chip.blue{background:var(--accent); border-color:var(--accent);}

/* Cards */
.card{background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:1rem 1.15rem;}
.card-title{font-weight:600; font-size:1rem; margin-bottom:.4rem;}
.kpi{background:linear-gradient(180deg,var(--panel2),var(--panel)); border:1px solid var(--border);
  border-radius:12px; padding:.9rem 1rem .6rem 1rem; height:100%;}
.kpi .label{color:var(--muted); font-size:.78rem;}
.kpi .value{font-size:1.75rem; font-weight:700; line-height:1.25; margin:.1rem 0 .15rem 0;}
.kpi .delta{font-size:.75rem; font-weight:600; float:right; margin-top:-1.7rem;}
.kpi .vs{font-size:.62rem; color:var(--muted); float:right; clear:right; margin-top:-.2rem;}
.up{color:var(--green);} .down{color:var(--red);}

.kpi .sub{font-size:.72rem; color:var(--muted); margin-bottom:.25rem;}
.act-Allowed{color:#34d399; font-weight:600;}
/* Alert table */
table.alerts{width:100%; border-collapse:collapse; font-size:.8rem;}
table.alerts th{color:var(--muted); font-weight:500; text-align:left; padding:.55rem .5rem;
  border-bottom:1px solid var(--border);}
table.alerts td{padding:.62rem .5rem; border-bottom:1px solid #151d33; color:var(--text);}
.badge{padding:.15rem .55rem; border-radius:5px; font-size:.7rem; font-weight:600; border:1px solid;}
.sev-Critical{color:#ff5b5b; background:rgba(239,68,68,.14); border-color:rgba(239,68,68,.45);}
.sev-High{color:#ff8a5b; background:rgba(249,115,22,.14); border-color:rgba(249,115,22,.45);}
.sev-Medium{color:#f5b73b; background:rgba(245,158,11,.14); border-color:rgba(245,158,11,.45);}
.sev-Low{color:#34d399; background:rgba(34,197,94,.14); border-color:rgba(34,197,94,.45);}
.act-Blocked{color:#ff5b5b; font-weight:600;} .act-Logged{color:#4c8dff; font-weight:600;}
.act-Monitored{color:#f5b73b; font-weight:600;}

/* Progress bars */
.bar-row{margin:.55rem 0; background:#0d1322; border:1px solid var(--border); border-radius:8px; padding:.55rem .7rem;}
.bar-row .top{display:flex; justify-content:space-between; font-size:.8rem; margin-bottom:.35rem;}
.bar-bg{height:6px; border-radius:4px; background:#1a2340; overflow:hidden;}
.bar-fg{height:100%; border-radius:4px;}

/* Result banner */
.result{border-radius:14px; padding:1.1rem 1.3rem; border:1px solid var(--border);
  background:linear-gradient(135deg,#101a33,#0f1424);}
.result .big{font-size:1.8rem; font-weight:800;}
.result .small{color:var(--muted); font-size:.85rem;}

/* Widgets */
.stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"]>button{
  background:var(--accent); color:#fff; border:0; border-radius:8px; font-weight:600; padding:.55rem 1.1rem;}
.stButton>button:hover, .stDownloadButton>button:hover, [data-testid="stFormSubmitButton"]>button:hover{
  background:#3d69f5; color:#fff; box-shadow:0 4px 14px rgba(47,91,234,.4);}
div[data-baseweb="select"]>div, .stNumberInput input, .stTextInput input{
  background:#0d1322 !important; border-color:var(--border) !important; color:var(--text) !important;}
[data-testid="stForm"]{background:var(--panel); border:1px solid var(--border); border-radius:12px;}
label, .stMarkdown p{color:var(--text);}
hr{border-color:var(--border);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Prediction engine (real model if available, otherwise demo engine)
# --------------------------------------------------------------------------
def _import_model_module():
    """Import Model.py (or model.py) from the app folder."""
    for mod_name in ("Model", "model"):
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "predict_attack"):
                return mod
        except Exception:
            continue
    return None


_MODEL_MODULE = _import_model_module()


@st.cache_resource(show_spinner="Preparing the AI model - the first start on a new server can take 1-3 minutes ...")
def _warm_up_model():
    """Load the saved model once per server process. If it was built with another scikit-learn version
    (typical after deploying to a cloud service) Model.py retrains automatically instead of failing."""
    if _MODEL_MODULE is None:
        return "demo", "Model.py was not found next to app.py"
    try:
        _MODEL_MODULE._load()
        return "ml", _MODEL_MODULE.model_status().get("detail", "")
    except Exception as exc:
        return "demo", f"{type(exc).__name__}: {exc}"


_ENGINE_STATE, _ENGINE_NOTE = _warm_up_model()
REAL_PREDICT = _MODEL_MODULE.predict_attack if (_MODEL_MODULE is not None and _ENGINE_STATE == "ml") else None
MODEL_MODE = "ML Model" if REAL_PREDICT else "Demo Engine"


def is_hosted() -> bool:
    """True when the app is opened through a public/remote address (e.g. Streamlit Community Cloud)."""
    try:
        host = str(st.context.headers.get("Host", "")).lower()
    except Exception:
        return False
    if not host:
        return False
    return not host.startswith(("localhost", "127.", "0.0.0.0", "[::1]", "192.168.", "10."))


def posture_risk(f: dict) -> float:
    """Security-posture component of hacking risk (0..40)."""
    r = 0.0
    r += min(f.get("open_ports", 0), 40) / 40 * 12
    r += 8 if f.get("outdated_software") else 0
    r += 0 if f.get("firewall_enabled", True) else 9
    r += 0 if f.get("mfa_enabled", False) else 4
    r += min(f.get("days_since_patch", 0), 180) / 180 * 7
    return r


def risk_level(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Medium"
    if score < 75:
        return "High"
    return "Critical"


def _demo_predict(f: dict) -> dict:
    """Transparent rule-based stand-in so the UI works before Model.py exists."""
    s = {c: 0.0 for c in CLASSES}
    s["Normal"] = 3.0
    pps, syn = f["packets_per_sec"], f["syn_error_rate"]
    s["DDoS Attack"] += min(pps / 4000, 4) + syn * 3.5 + (1.2 if f["conn_count"] > 400 else 0)
    s["Port Scan"] += min(f["unique_dst_ports"] / 25, 4) + (1.5 if f["src_bytes"] < 200 and f["packet_count"] > 30 else 0)
    s["Brute Force"] += min(f["failed_logins"] / 4, 4) + min(f["login_attempts"] / 25, 2)
    s["SQL Injection"] += (3.0 if f["payload_sql_tokens"] > 2 else 0) + min(f["payload_sql_tokens"] / 6, 1.5)
    s["XSS Attack"] += (3.0 if f["payload_script_tokens"] > 2 else 0) + min(f["payload_script_tokens"] / 6, 1.5)
    s["Normal"] -= 0.6 * max(s[c] for c in CLASSES if c != "Normal")
    s["Normal"] = max(s["Normal"], 0.1)
    v = np.array([s[c] for c in CLASSES])
    p = np.exp(v - v.max())
    p /= p.sum()
    probs = {c: float(x) for c, x in zip(CLASSES, p)}
    top = max(probs, key=probs.get)
    attack_prob = 1.0 - probs["Normal"]
    risk = min(100.0, attack_prob * 60 + posture_risk(f))
    return {"attack_type": top, "confidence": probs[top], "probabilities": probs, "risk_score": risk}


def run_prediction(features: dict) -> dict:
    if REAL_PREDICT:
        try:
            out = REAL_PREDICT(features)
        except Exception as exc:  # fall back, but tell the user (once per session)
            if not st.session_state.get("_model_warned"):
                st.session_state["_model_warned"] = True
                st.warning(f"Model error ({exc}). Falling back to the demo engine.")
            out = _demo_predict(features)
    else:
        out = _demo_predict(features)
    out["risk_level"] = risk_level(out["risk_score"])
    out["timestamp"] = datetime.now()
    out["features"] = dict(features)
    return out


# --------------------------------------------------------------------------
# Session state & persistent analysis history
# (every Manual Analysis / Live Monitor result is stored and drives the dashboard)
# --------------------------------------------------------------------------
HISTORY_PATH = Path(__file__).resolve().parent / "analysis_history.jsonl"
MAX_HISTORY = 2000


def _json_default(o):
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def load_history() -> list:
    """Reload earlier analyses so the dashboard survives an app restart."""
    out = []
    if is_hosted():  # a hosted app is shared by many visitors: keep history per browser session only
        return out
    try:
        if HISTORY_PATH.exists():
            for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-MAX_HISTORY:]:
                try:
                    d = json.loads(line)
                    d["timestamp"] = datetime.fromisoformat(d["timestamp"])
                    out.append(d)
                except Exception:
                    continue
    except Exception:
        pass
    return out


def save_result(res: dict):
    if is_hosted():
        return
    try:
        row = dict(res)
        row["timestamp"] = res["timestamp"].isoformat()
        with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=_json_default) + "\n")
    except Exception:
        pass  # history persistence is best-effort


def init_state():
    ss = st.session_state
    if "history" not in ss:
        ss.history = load_history()
    ss.setdefault("last_result", ss.history[-1] if ss.history else None)
    ss.setdefault("manual_result", None)
    ss.setdefault("live_snapshot", None)
    ss.setdefault("live_log", [])
    ss.setdefault("_prev_io", None)
    ss.setdefault("settings", {
        "firewall_enabled": True, "mfa_enabled": False, "days_since_patch": 21,
        "outdated_software": False, "alert_threshold": 0.60, "analyst": "Admin",
    })


init_state()


def record_result(res: dict, source: str, src_ip: str, dst_ip: str) -> dict:
    """Store a finished analysis: it immediately shows up on Dashboard, Alerts and Reports."""
    cfg = st.session_state.settings
    sev = res["risk_level"]
    res["source"], res["src_ip"], res["dst_ip"] = source, src_ip, dst_ip
    res["flagged"] = bool(res["attack_type"] != "Normal" and res["confidence"] >= cfg["alert_threshold"])
    if res["flagged"]:
        res["action"] = "Blocked" if sev in ("High", "Critical") else ("Monitored" if sev == "Medium" else "Logged")
    else:
        res["action"] = "Allowed"
    st.session_state.history.append(res)
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]
    st.session_state.last_result = res
    save_result(res)
    return res


HISTORY_COLS = ["_i", "Time", "Source", "Type", "Confidence", "Risk", "Severity", "Source IP",
                "Destination IP", "Action", "Flagged", "ThreatProb"]


def history_df() -> pd.DataFrame:
    rows = []
    for i, h in enumerate(st.session_state.history):
        rows.append({
            "_i": i, "Time": h["timestamp"],
            "Source": "Manual" if str(h.get("source", "")).startswith("Manual") else "Live",
            "Type": h["attack_type"], "Confidence": round(h["confidence"] * 100, 1),
            "Risk": round(float(h["risk_score"]), 1), "Severity": h["risk_level"],
            "Source IP": h.get("src_ip", "-"), "Destination IP": h.get("dst_ip", "-"),
            "Action": h.get("action", "Logged"),
            "Flagged": bool(h.get("flagged", h["attack_type"] != "Normal")),
            "ThreatProb": (1 - h["probabilities"].get("Normal", 0.0)) * 100,
        })
    df = pd.DataFrame(rows, columns=HISTORY_COLS)
    df["Time"] = pd.to_datetime(df["Time"])
    df["Risk"] = df["Risk"].astype(float)
    df["ThreatProb"] = df["ThreatProb"].astype(float)
    df["Flagged"] = df["Flagged"].astype(bool)
    return df


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#aab3cc", size=11), margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor="#18213a", zeroline=False), yaxis=dict(gridcolor="#18213a", zeroline=False),
)


def plot_layout(**over) -> dict:
    """PLOT_LAYOUT merged with per-chart overrides (avoids 'multiple values for keyword argument')."""
    base = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PLOT_LAYOUT.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


def header(title: str, subtitle: str = "", right_html: str = ""):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"<div class='page-title'>{title}</div><div class='page-sub'>{subtitle}</div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='text-align:right'>{right_html}</div>", unsafe_allow_html=True)
    st.write("")


def sparkline(values, color: str, w=190, h=34) -> str:
    v = np.array(list(values), dtype=float)
    if v.size == 0:
        v = np.array([0.0, 0.0])
    elif v.size == 1:
        v = np.array([v[0], v[0]])
    rng = (v.max() - v.min()) or 1
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 3 - (v - v.min()) / rng * (h - 8)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f"<svg width='100%' height='{h}' viewBox='0 0 {w} {h}' preserveAspectRatio='none'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='1.6' points='{pts}'/></svg>")


def kpi_card(label, value, sub, color, series):
    return (f"<div class='kpi'><div class='label'>{label}</div>"
            f"<div class='value'>{value}</div><div class='sub'>{sub}</div>"
            f"{sparkline(series, color)}</div>")


def alerts_table_html(df: pd.DataFrame, n: int = 6) -> str:
    if df is None or len(df) == 0:
        return "<div style='color:#8892ad;padding:.9rem .2rem'>No events yet.</div>"
    rows = ""
    for _, r in df.head(n).iterrows():
        t = pd.to_datetime(r["Time"]).strftime("%b %d, %H:%M:%S")
        dot = ATTACK_COLORS.get(r["Type"], "#94a3b8")
        rows += (f"<tr><td style='color:#8892ad'>{t}</td><td>{r['Source']}</td>"
                 f"<td><span style='color:{dot}'>●</span> {r['Type']}</td><td>{r['Confidence']:.0f}%</td>"
                 f"<td>{escape(str(r['Source IP']))}</td><td>{escape(str(r['Destination IP']))}</td>"
                 f"<td>{r['Risk']:.0f}</td>"
                 f"<td><span class='badge sev-{r['Severity']}'>{r['Severity']}</span></td>"
                 f"<td class='act-{r['Action']}'>{r['Action']}</td></tr>")
    return ("<table class='alerts'><thead><tr><th>Time</th><th>Source</th><th>Type</th><th>Conf.</th>"
            "<th>Source IP</th><th>Destination IP</th><th>Risk</th><th>Severity</th><th>Action</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def progress_row(label: str, pct: float, color: str) -> str:
    return (f"<div class='bar-row'><div class='top'><span>{label}</span><span>{pct:.0f}%</span></div>"
            f"<div class='bar-bg'><div class='bar-fg' style='width:{min(pct, 100):.0f}%;background:{color}'></div></div></div>")


def gauge(score: float, level: str) -> go.Figure:
    color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}[level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=44, color="#e5e9f5"), suffix=""),
        title=dict(text="Hacking Risk Score", font=dict(size=13, color="#8892ad")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#3a4563"), bar=dict(color=color, thickness=0.28),
            bgcolor="#0d1322", borderwidth=0,
            steps=[dict(range=[0, 25], color="#12301f"), dict(range=[25, 50], color="#3a3012"),
                   dict(range=[50, 75], color="#3b2412"), dict(range=[75, 100], color="#3a1517")],
        )))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def prob_bars(probs: dict) -> go.Figure:
    items = sorted(probs.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v * 100 for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color=[ATTACK_COLORS.get(k, "#2f5bea") for k, _ in items],
        text=[f"{v * 100:.1f}%" for _, v in items], textposition="outside"))
    fig.update_layout(**plot_layout(height=260, xaxis=dict(range=[0, 115], title="Probability (%)")))
    return fig


def result_panel(res: dict):
    level = res["risk_level"]
    col = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}[level]
    verdict_col = "#22c55e" if res["attack_type"] == "Normal" else "#ef4444"
    st.markdown(
        f"<div class='result'><div class='small'>PREDICTED TRAFFIC CLASS</div>"
        f"<div class='big' style='color:{verdict_col}'>{res['attack_type'].upper()}</div>"
        f"<div class='small'>Confidence {res['confidence'] * 100:.1f}% &nbsp;•&nbsp; "
        f"Risk level <b style='color:{col}'>{level}</b> &nbsp;•&nbsp; Engine: {MODEL_MODE}</div></div>",
        unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='card-title'>Hacking Risk</div>", unsafe_allow_html=True)
        st.plotly_chart(gauge(res["risk_score"], level), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='card-title'>Attack Probability Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(prob_bars(res["probabilities"]), width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='card'><div class='card-title'>🛠️ Recommended Actions</div>", unsafe_allow_html=True)
    for tip in RECOMMENDATIONS[res["attack_type"]]:
        st.markdown(f"- {tip}")
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------
def build_pdf(res: dict, source: str, analyst: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Cyber Attack Detection Report", author="NIDS")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=colors.HexColor("#0b1b4d"), fontSize=20, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#1d3fae"), spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#667085"))

    lvl_col = {"Low": "#16a34a", "Medium": "#d97706", "High": "#ea580c", "Critical": "#dc2626"}[res["risk_level"]]
    story = [
        Paragraph("AI-Based Cyber Attack Detection &amp; Hacking Risk Report", h1),
        Paragraph(f"Generated {res['timestamp']:%Y-%m-%d %H:%M:%S} &nbsp;|&nbsp; Analyst: {analyst} "
                  f"&nbsp;|&nbsp; Source: {source} &nbsp;|&nbsp; Engine: {MODEL_MODE}", small),
        Spacer(1, 8),
    ]

    summary = [
        ["Predicted class", res["attack_type"]],
        ["Model confidence", f"{res['confidence'] * 100:.1f}%"],
        ["Hacking risk score", f"{res['risk_score']:.1f} / 100"],
        ["Risk level", res["risk_level"]],
        ["Host", f"{socket.gethostname()} ({platform.system()} {platform.release()})"],
    ]
    t = Table(summary, colWidths=[55 * mm, 110 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2ff")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d2fe")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (1, 3), (1, 3), colors.HexColor(lvl_col)),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [Paragraph("1. Executive Summary", h2), t]

    prob_rows = [["Class", "Probability"]] + [
        [k, f"{v * 100:.2f}%"] for k, v in sorted(res["probabilities"].items(), key=lambda x: -x[1])]
    pt = Table(prob_rows, colWidths=[80 * mm, 40 * mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3fae")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d2fe")), ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8faff")]),
    ]))
    story += [Paragraph("2. Attack Probability Distribution", h2), pt]

    feats = res["features"]
    feat_rows = [["Parameter", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in feats.items()]
    ft = Table(feat_rows, colWidths=[80 * mm, 60 * mm], repeatRows=1)
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3fae")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d5dd")), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8faff")]),
    ]))
    story += [Paragraph("3. Analysed Parameters", h2), ft]

    story.append(Paragraph("4. Recommended Mitigation", h2))
    for tip in RECOMMENDATIONS[res["attack_type"]]:
        story.append(Paragraph(f"• {tip}", body))
    story += [Spacer(1, 12), Paragraph(
        "Disclaimer: predictions are probabilistic and should be validated by a qualified security analyst "
        "before any action is taken.", small)]
    doc.build(story)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Live system snapshot (Feature 2)
# --------------------------------------------------------------------------
SERVICE_BY_PORT = {80: "http", 8080: "http", 443: "https", 8443: "https", 22: "ssh", 21: "ftp",
                   53: "dns", 25: "smtp", 587: "smtp", 23: "telnet", 445: "smb"}
MIN_PACKETS = 20   # below this a scan window carries too little traffic to judge


def _is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return False


def collect_system_features(interval: float) -> tuple[dict, dict]:
    """Collect every parameter automatically from the machine that runs the application.

    The values are built to match what the model saw in training: only real network traffic counts
    (loopback and listening sockets are ignored), "destination ports" are the service ports this machine
    connects OUT to (never the random ports of visiting clients), and ratios are only used when there is
    enough traffic for them to mean something.
    """
    if psutil is None:
        raise RuntimeError("psutil is not installed. Run: pip install psutil")
    ss = st.session_state
    io_now = psutil.net_io_counters()
    now = time.time()
    prev = ss._prev_io
    ss._prev_io = (now, io_now)
    if prev is None:
        time.sleep(0.6)
        io_now2 = psutil.net_io_counters()
        dt, sent, recv = 0.6, io_now2.bytes_sent - io_now.bytes_sent, io_now2.bytes_recv - io_now.bytes_recv
        pk = (io_now2.packets_sent - io_now.packets_sent) + (io_now2.packets_recv - io_now.packets_recv)
        errs = (io_now2.errin + io_now2.errout + io_now2.dropin + io_now2.dropout) - \
               (io_now.errin + io_now.errout + io_now.dropin + io_now.dropout)
        io_now = io_now2
        ss._prev_io = (time.time(), io_now2)
    else:
        dt = max(now - prev[0], 0.1)
        p = prev[1]
        sent, recv = io_now.bytes_sent - p.bytes_sent, io_now.bytes_recv - p.bytes_recv
        pk = (io_now.packets_sent - p.packets_sent) + (io_now.packets_recv - p.packets_recv)
        errs = (io_now.errin + io_now.errout + io_now.dropin + io_now.dropout) - \
               (p.errin + p.errout + p.dropin + p.dropout)
    sent, recv, pk, errs = max(sent, 0), max(recv, 0), max(pk, 0), max(errs, 0)

    conns = []
    try:
        conns = psutil.net_connections(kind="tcp")
    except Exception:
        pass  # needs admin rights on some systems / restricted on cloud containers
    listening = {c.laddr.port for c in conns if c.status == "LISTEN" and c.laddr}
    listen_open = {c.laddr.port for c in conns if c.status == "LISTEN" and c.laddr and not _is_loopback(c.laddr.ip)}
    ext = [c for c in conns if c.raddr and c.status != "LISTEN" and not _is_loopback(c.raddr.ip)]
    outbound = [c for c in ext if c.laddr and c.laddr.port not in listening]   # connections this machine opened
    states = {"ESTABLISHED": 0, "SYN_RECV": 0, "SYN_SENT": 0, "TIME_WAIT": 0}
    for c in ext:
        if c.status in states:
            states[c.status] += 1
    remote_ips = {c.raddr.ip for c in ext}
    out_ports = [c.raddr.port for c in outbound]
    local_ports = sorted(c.laddr.port for c in outbound if c.laddr)
    n_ext = len(ext)
    syn_rate = (states["SYN_RECV"] + states["SYN_SENT"]) / n_ext if n_ext >= 5 else 0.0
    dst_port = max(set(out_ports), key=out_ports.count) if out_ports else 443
    src_port = local_ports[len(local_ports) // 2] if local_ports else 49152
    err_rate = min(errs / pk, 1.0) if pk >= 50 else 0.0   # a ratio of a handful of packets is just noise
    cfg = ss.settings

    feats = {
        "duration": round(dt, 2),
        "protocol_type": "tcp",
        "service": SERVICE_BY_PORT.get(dst_port, "other"),
        "flag": "S0" if syn_rate > 0.3 else "SF",
        "src_bytes": int(sent),
        "dst_bytes": int(recv),
        "packet_count": int(pk),
        "packets_per_sec": round(pk / dt, 1),
        "src_port": int(src_port),
        "dst_port": int(dst_port),
        "unique_dst_ports": max(len(set(out_ports)), 1),
        "unique_dst_ips": max(len(remote_ips), 1),
        "conn_count": max(n_ext, 1),
        "syn_error_rate": round(syn_rate, 3),
        "error_rate": round(err_rate, 4),
        "failed_logins": 0,  # OS auth logs need elevated, OS-specific access
        "login_attempts": 0,
        "payload_sql_tokens": 0,
        "payload_script_tokens": 0,
        "open_ports": len(listen_open),
        "outdated_software": cfg["outdated_software"],
        "firewall_enabled": cfg["firewall_enabled"],
        "mfa_enabled": cfg["mfa_enabled"],
        "days_since_patch": cfg["days_since_patch"],
    }
    sysinfo = {
        "cpu": psutil.cpu_percent(interval=None),
        "mem": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent if platform.system() != "Windows" else psutil.disk_usage("C:\\").percent,
        "net_kbps": (sent + recv) / dt / 1024,
        "procs": len(psutil.pids()),
        "listen": len(listen_open),
        "established": states["ESTABLISHED"],
        "remote_ips": len(remote_ips),
        "packets": int(pk),
    }
    return feats, sysinfo


def apply_idle_guard(res: dict, feats: dict) -> dict:
    """A scan window with almost no traffic cannot be judged reliably - report it as Normal (idle)."""
    if feats["packet_count"] < MIN_PACKETS and feats["syn_error_rate"] < 0.3 and res["attack_type"] != "Normal":
        raw = res["attack_type"]
        res["raw_model_class"] = raw
        res["attack_type"] = "Normal"
        res["confidence"] = float(res["probabilities"].get("Normal", 0.0))
        res["note"] = (f"Only {feats['packet_count']} packets were seen in this scan window, which is too little traffic "
                       f"to judge reliably, so it is reported as Normal (idle). Raw model output was: {raw}.")
    return res


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def goto(page_name: str):
    """on_click callback: jump to another page."""
    st.session_state["nav"] = page_name


NAV = ["📊  Dashboard", "🧪  Manual Analysis", "📡  Live Monitor", "🚨  Alerts", "📄  Reports", "⚙️  Settings"]

with st.sidebar:
    st.markdown("<div class='brand'><div class='logo'>🛡️</div><div><div class='name'>NIDS</div>"
                "<div class='sub'>AI CYBER DEFENSE</div></div></div>", unsafe_allow_html=True)
    page = st.radio("Navigation", NAV, key="nav", label_visibility="collapsed")
    st.markdown(
        f"<div class='sys-status'><span class='dot'></span><b>System Status</b><br>"
        f"<span style='color:#22c55e'>All systems operational</span><br>"
        f"Engine: {MODEL_MODE}<br>Version 1.0.0</div>", unsafe_allow_html=True)



# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

WINDOWS = {"Last hour": timedelta(hours=1), "Last 24 hours": timedelta(hours=24),
           "Last 7 days": timedelta(days=7), "All time": None}
LEVEL_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}


def _empty_fig(msg: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**plot_layout(height=height, xaxis=dict(visible=False), yaxis=dict(visible=False),
                                    annotations=[dict(text=msg, showarrow=False, font=dict(color="#8892ad", size=13))]))
    return fig


def page_dashboard():
    header("Network Intrusion Detection System",
           "Live overview built from every Manual Analysis and Live Monitor result")

    df_all = history_df()
    f1, f2, f3 = st.columns([1.2, 1.2, 2.6])
    window = f1.selectbox("Time window", list(WINDOWS), index=3, key="dash_window")
    source = f2.selectbox("Source", ["All", "Manual", "Live"], key="dash_source")
    only_threats = f3.toggle("Show threats only in the events table", value=False, key="dash_threats")

    df = df_all
    if WINDOWS[window] is not None and not df.empty:
        df = df[df["Time"] >= datetime.now() - WINDOWS[window]]
    if source != "All":
        df = df[df["Source"] == source]
    n = len(df)

    if df_all.empty:
        st.info("No analyses yet - the dashboard fills automatically as you use the app. "
                "Run one to get started:")
        b1, b2 = st.columns(2)
        b1.button("🧪  Run a manual analysis", on_click=goto, args=(NAV[1],), width="stretch")
        b2.button("📡  Start live monitoring", on_click=goto, args=(NAV[2],), width="stretch")
        st.write("")

    # ---- KPI cards ---------------------------------------------------------
    threats = int(df["Flagged"].sum())
    hi_mask = df["Flagged"] & df["Severity"].isin(["High", "Critical"])
    avg_risk = float(df["Risk"].mean()) if n else 0.0
    latest_risk = float(df["Risk"].iloc[-1]) if n else 0.0
    n_man, n_live = int((df["Source"] == "Manual").sum()), int((df["Source"] == "Live").sum())
    cards = [
        ("Total Analyses", f"{n:,}", f"{n_man} manual · {n_live} live", "#3b82f6", np.arange(1, n + 1)),
        ("Threats Detected", f"{threats:,}", f"{(threats / n * 100 if n else 0):.0f}% of analyses",
         "#ef4444", df["Flagged"].astype(int).cumsum()),
        ("High / Critical", f"{int(hi_mask.sum()):,}", "flagged, severity High+", "#f59e0b",
         hi_mask.astype(int).cumsum()),
        ("Average Risk", f"{avg_risk:.1f}", f"level: {risk_level(avg_risk)}", "#a855f7", df["Risk"].expanding().mean()),
        ("Latest Risk", f"{latest_risk:.1f}", (df["Type"].iloc[-1] if n else "no data yet"), "#22c55e", df["Risk"]),
    ]
    for c, args in zip(st.columns(5), cards):
        c.markdown(kpi_card(*args), unsafe_allow_html=True)
    st.write("")

    # ---- timeline + donut --------------------------------------------------
    left, right = st.columns([2.1, 1])
    with left:
        st.markdown("<div class='card'><div class='card-title'>Risk & Threat Timeline</div>", unsafe_allow_html=True)
        if n:
            fig = go.Figure()
            for y0, y1, col in [(0, 25, "#22c55e"), (25, 50, "#f59e0b"), (50, 75, "#f97316"), (75, 100, "#ef4444")]:
                fig.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.05, line_width=0)
            fig.add_trace(go.Scatter(
                x=df["Time"], y=df["Risk"], name="Hacking risk", mode="lines+markers", text=df["Type"],
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=10, color=[ATTACK_COLORS.get(t, "#94a3b8") for t in df["Type"]],
                            line=dict(width=1, color="#0f1424")),
                hovertemplate="%{text}<br>Risk %{y:.1f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=df["Time"], y=df["ThreatProb"], name="Threat probability %", mode="lines",
                                     line=dict(color="#f59e0b", width=1.6, dash="dot")))
            fig.update_layout(**plot_layout(margin=dict(l=10, r=10, t=30, b=10), height=320,
                                            legend=dict(orientation="h", y=1.12, x=0),
                                            yaxis=dict(range=[0, 100], title="Score")))
        else:
            fig = _empty_fig("No analyses in this selection", 320)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='card-title'>Detected Traffic Classes</div>", unsafe_allow_html=True)
        if n:
            counts = df["Type"].value_counts()
            fig = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values), hole=0.68, sort=False,
                marker=dict(colors=[ATTACK_COLORS.get(k, "#94a3b8") for k in counts.index],
                            line=dict(color="#0f1424", width=3)),
                textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
            fig.update_layout(**plot_layout(margin=dict(l=0, r=0, t=0, b=0), height=320, legend=dict(font=dict(size=10)),
                                            annotations=[dict(text=f"<b style='font-size:22px'>{n:,}</b><br>Total",
                                                              showarrow=False, font=dict(color="#e5e9f5"))]))
        else:
            fig = _empty_fig("No data", 320)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    a, b = st.columns([2.1, 1])
    with a:
        events = df[df["Flagged"]] if only_threats else df
        events = events.sort_values("Time", ascending=False)
        st.markdown(f"<div class='card'><div class='card-title'>Recent {'Threats' if only_threats else 'Events'}"
                    f" <span style='color:#8892ad;font-weight:400;font-size:.8rem'>(latest 8 of {len(events)})</span></div>"
                    f"{alerts_table_html(events, 8)}</div>", unsafe_allow_html=True)
    with b:
        if n:
            last = st.session_state.history[int(df["_i"].iloc[-1])]
            col = ATTACK_COLORS.get(last["attack_type"], "#94a3b8")
            lvl = last["risk_level"]
            st.markdown(
                f"<div class='card'><div class='card-title'>Latest Analysis</div>"
                f"<div style='color:#8892ad;font-size:.75rem'>{last.get('source', '')} · "
                f"{last['timestamp']:%b %d, %H:%M:%S}</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{col};margin:.2rem 0'>{last['attack_type'].upper()}</div>"
                f"<div style='font-size:.8rem;color:#8892ad'>Confidence {last['confidence'] * 100:.1f}% · "
                f"Risk level <b style='color:{LEVEL_COLORS[lvl]}'>{lvl}</b></div>"
                f"{progress_row('Hacking risk', last['risk_score'], LEVEL_COLORS[lvl])}"
                f"<div style='font-size:.8rem'>💡 {RECOMMENDATIONS[last['attack_type']][0]}</div></div>",
                unsafe_allow_html=True)
            st.button("📄  Open full report", on_click=goto, args=(NAV[4],), width="stretch", key="dash_open_report")
            st.write("")
        st.markdown("<div class='card'><div class='card-title'>System Status</div>", unsafe_allow_html=True)
        if psutil:
            disk_path = "C:\\" if platform.system() == "Windows" else "/"
            vals = {"CPU Usage": (psutil.cpu_percent(interval=0.2), "#3b82f6"),
                    "Memory Usage": (psutil.virtual_memory().percent, "#a855f7"),
                    "Disk Usage": (psutil.disk_usage(disk_path).percent, "#f59e0b")}
        else:
            vals = {"CPU Usage": (0, "#3b82f6"), "Memory Usage": (0, "#a855f7"), "Disk Usage": (0, "#f59e0b")}
        st.markdown("".join(progress_row(k, v, c) for k, (v, c) in vals.items()), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_manual():
    header("Manual Analysis", "Enter packet, network and security-posture details to predict attack & hacking risk")
    cfg = st.session_state.settings

    with st.form("manual_form"):
        st.markdown("##### 📦 Packet Information")
        c1, c2, c3, c4 = st.columns(4)
        protocol = c1.selectbox("Protocol", ["tcp", "udp", "icmp"])
        service = c2.selectbox("Service", ["http", "https", "ssh", "ftp", "dns", "smtp", "telnet", "smb", "other"])
        flag = c3.selectbox("Connection Flag", ["SF", "S0", "REJ", "RSTO", "SH", "OTH"])
        duration = c4.number_input("Duration (s)", 0.0, 100000.0, 2.0, step=0.5)
        c1, c2, c3, c4 = st.columns(4)
        src_bytes = c1.number_input("Source Bytes", 0, 10**9, 1200, step=100)
        dst_bytes = c2.number_input("Destination Bytes", 0, 10**9, 5400, step=100)
        packet_count = c3.number_input("Packet Count", 0, 10**7, 40, step=10)
        pps = c4.number_input("Packets / Second", 0.0, 1000000.0, 35.0, step=10.0)

        st.markdown("##### 🌐 Network Information")
        c1, c2, c3, c4 = st.columns(4)
        src_ip = c1.text_input("Source IP", "192.168.1.105")
        dst_ip = c2.text_input("Destination IP", "10.0.0.5")
        src_port = c3.number_input("Source Port", 0, 65535, 49152)
        dst_port = c4.number_input("Destination Port", 0, 65535, 443)
        c1, c2, c3, c4 = st.columns(4)
        conn_count = c1.number_input("Concurrent Connections", 0, 10**6, 12)
        udp = c2.number_input("Unique Destination Ports", 0, 65535, 2)
        udi = c3.number_input("Unique Destination IPs", 0, 65535, 1)
        syn = c4.slider("SYN / Error Rate", 0.0, 1.0, 0.02, 0.01)

        st.markdown("##### 🔐 Authentication & Payload")
        c1, c2, c3, c4 = st.columns(4)
        failed = c1.number_input("Failed Logins", 0, 10**5, 0)
        attempts = c2.number_input("Login Attempts", 0, 10**5, 1)
        sql_tok = c3.number_input("SQL Tokens in Payload", 0, 1000, 0, help="e.g. UNION, SELECT, OR 1=1, --")
        xss_tok = c4.number_input("Script Tokens in Payload", 0, 1000, 0, help="e.g. <script>, onerror=, javascript:")

        st.markdown("##### 🛡️ Security Posture")
        c1, c2, c3, c4, c5 = st.columns(5)
        open_ports = c1.number_input("Open Ports", 0, 65535, 6)
        days_patch = c2.number_input("Days Since Last Patch", 0, 3650, cfg["days_since_patch"])
        outdated = c3.toggle("Outdated Software", cfg["outdated_software"])
        fw = c4.toggle("Firewall Enabled", cfg["firewall_enabled"])
        mfa = c5.toggle("MFA Enabled", cfg["mfa_enabled"])

        submitted = st.form_submit_button("🔍  Analyze & Predict", width="stretch")

    if submitted:
        feats = {
            "duration": duration, "protocol_type": protocol, "service": service, "flag": flag,
            "src_bytes": src_bytes, "dst_bytes": dst_bytes, "packet_count": packet_count,
            "packets_per_sec": pps, "src_port": src_port, "dst_port": dst_port,
            "unique_dst_ports": udp, "unique_dst_ips": udi, "conn_count": conn_count,
            "syn_error_rate": syn, "error_rate": syn, "failed_logins": failed, "login_attempts": attempts,
            "payload_sql_tokens": sql_tok, "payload_script_tokens": xss_tok, "open_ports": open_ports,
            "outdated_software": outdated, "firewall_enabled": fw, "mfa_enabled": mfa,
            "days_since_patch": days_patch,
        }
        with st.spinner("Running detection pipeline..."):
            time.sleep(0.6)
            res = run_prediction(feats)
        st.session_state.manual_result = record_result(res, "Manual input", src_ip, dst_ip)

    res = st.session_state.manual_result
    if res is not None:
        st.write("")
        result_panel(res)
        st.write("")
        pdf = build_pdf(res, "Manual input", st.session_state.settings["analyst"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇️  Download PDF Report", pdf, f"cyber_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                           "application/pdf", width="stretch")
        d2.button("📊  View on Dashboard", on_click=goto, args=(NAV[0],), key="man_to_dash", width="stretch")



def page_live():
    header("Live System Monitor", "Automatically collects network & host parameters from this computer")
    if psutil is None:
        st.error("`psutil` is required for live monitoring. Install it with `pip install psutil`.")
        return
    if is_hosted():
        st.warning("This application is running on a **remote server**. The Live Monitor measures the machine that "
                   "runs the app (the server), **not your own computer**, and a quiet server will mostly look normal. "
                   "To monitor your own PC, run the app locally with `streamlit run app.py`. "
                   "Use **Manual Analysis** for any online demonstration.")

    c1, c2, c3 = st.columns([1, 1, 2])
    auto = c1.toggle("Auto-refresh", value=False, key="live_auto")
    interval = c2.slider("Interval (s)", 2, 30, 5, key="live_interval")
    scan = c3.button("▶  Scan Now", width="stretch")

    if scan or auto:  # only a real scan creates a new record (button clicks elsewhere don't)
        try:
            feats, info = collect_system_features(interval)
        except Exception as exc:
            st.error(f"Could not collect system data: {exc}")
            return
        res = apply_idle_guard(run_prediction(feats), feats)
        record_result(res, f"Live monitor ({socket.gethostname()})", "multiple remote hosts", local_ip())
        st.session_state.live_log.append({
            "t": res["timestamp"], "risk": res["risk_score"], "kbps": info["net_kbps"], "cpu": info["cpu"]})
        st.session_state.live_log = st.session_state.live_log[-120:]
        st.session_state.live_snapshot = (feats, info, res)

    snap = st.session_state.live_snapshot
    if snap is None:
        st.info("Press **Scan Now** or enable **Auto-refresh** to start monitoring this machine. "
                "Every scan is added to the Dashboard automatically.")
        return
    feats, info, res = snap

    cards = st.columns(6)
    items = [("Host", socket.gethostname()), ("Local IP", local_ip()), ("Established", info["established"]),
             ("Listening Ports", info["listen"]), ("Remote IPs", info["remote_ips"]), ("Processes", info["procs"])]
    for c, (k, v) in zip(cards, items):
        c.markdown(f"<div class='kpi'><div class='label'>{k}</div><div class='value' style='font-size:1.25rem'>{v}</div></div>",
                   unsafe_allow_html=True)
    st.write("")

    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("<div class='card'><div class='card-title'>Live Risk & Throughput</div>", unsafe_allow_html=True)
        log = pd.DataFrame(st.session_state.live_log)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log["t"], y=log["risk"], name="Hacking risk", line=dict(color="#ef4444", width=2),
                                 fill="tozeroy", fillcolor="rgba(239,68,68,.10)"))
        fig.add_trace(go.Scatter(x=log["t"], y=log["kbps"], name="Network KB/s", yaxis="y2",
                                 line=dict(color="#3b82f6", width=2)))
        fig.update_layout(**plot_layout(
            margin=dict(l=10, r=10, t=30, b=10), height=280, legend=dict(orientation="h", y=1.15),
            yaxis=dict(range=[0, 100], title="Risk"),
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="KB/s")))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='card-title'>Host Resources</div>", unsafe_allow_html=True)
        st.markdown(progress_row("CPU", info["cpu"], "#3b82f6") + progress_row("Memory", info["mem"], "#a855f7")
                    + progress_row("Disk", info["disk"], "#f59e0b"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if res.get("note"):
        st.info(res["note"])
    result_panel(res)
    with st.expander("🔬 Collected feature vector (sent to the model)"):
        st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "value"}).astype(str), width="stretch")
    d1, d2 = st.columns(2)
    pdf = build_pdf(res, res["source"], st.session_state.settings["analyst"])
    d1.download_button("⬇️  Download PDF Report", pdf, f"live_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                       "application/pdf", key="live_pdf", width="stretch")
    d2.button("📊  View on Dashboard", on_click=goto, args=(NAV[0],), key="live_to_dash", width="stretch")

    if auto:
        time.sleep(interval)
        st.rerun()


def page_alerts():
    header("Alerts", "Threats detected by Manual Analysis and Live Monitor, with automated responses")
    df = history_df()
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.1, 1.3])
    sev = c1.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
    typ = c2.multiselect("Attack type", CLASSES[1:], default=CLASSES[1:])
    src = c3.multiselect("Source", ["Manual", "Live"], default=["Manual", "Live"])
    show_normal = c4.toggle("Include normal events", value=False)
    if not show_normal:
        df = df[df["Flagged"]]
        mask = df["Type"].isin(typ)
    else:
        mask = df["Type"].isin(typ + ["Normal"])
    f = df[mask & df["Severity"].isin(sev) & df["Source"].isin(src)].sort_values("Time", ascending=False)
    st.markdown(f"<div class='card'><div class='card-title'>{len(f)} alerts</div>"
                f"{alerts_table_html(f, 50)}</div>", unsafe_allow_html=True)
    st.write("")
    st.download_button("⬇️  Export CSV", f.drop(columns=["_i"]).to_csv(index=False).encode(), "alerts.csv", "text/csv")


def page_reports():
    header("Reports", "Generate a professional PDF from the latest analysis")
    res = st.session_state.last_result
    if res is None:
        st.info("No analysis yet. Run **Manual Analysis** or **Live Monitor** first.")
        return
    result_panel(res)
    st.write("")
    pdf = build_pdf(res, res.get("source", "Analysis"), st.session_state.settings["analyst"])
    st.download_button("⬇️  Download PDF Report", pdf, f"cyber_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                       "application/pdf", key="rep_pdf")
    hist = st.session_state.history
    if hist:
        st.markdown("##### Analysis history")
        st.dataframe(pd.DataFrame([{
            "Time": h["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), "Source": h.get("source", ""),
            "Class": h["attack_type"], "Confidence %": round(h["confidence"] * 100, 1),
            "Risk": round(h["risk_score"], 1), "Level": h["risk_level"]} for h in reversed(hist)]),
            width="stretch")



def page_settings():
    header("Settings", "Host posture assumptions and alert behaviour")
    cfg = st.session_state.settings
    with st.form("settings"):
        c1, c2 = st.columns(2)
        cfg["analyst"] = c1.text_input("Analyst name (shown in reports)", cfg["analyst"])
        cfg["alert_threshold"] = c2.slider("Alert confidence threshold", 0.3, 0.99, cfg["alert_threshold"], 0.01)
        st.markdown("##### Host security posture (used by Live Monitor — not detectable cross-platform)")
        c1, c2, c3, c4 = st.columns(4)
        cfg["firewall_enabled"] = c1.toggle("Firewall enabled", cfg["firewall_enabled"])
        cfg["mfa_enabled"] = c2.toggle("MFA enabled", cfg["mfa_enabled"])
        cfg["outdated_software"] = c3.toggle("Outdated software", cfg["outdated_software"])
        cfg["days_since_patch"] = c4.number_input("Days since last patch", 0, 3650, cfg["days_since_patch"])
        st.form_submit_button("💾  Save settings")
    st.markdown(f"<div class='card'><b>Prediction engine:</b> {MODEL_MODE}<br>"
                f"<span style='color:#8892ad'>Place <code>Model.py</code> with <code>predict_attack(features)</code> "
                f"next to <code>app.py</code> to switch from the demo engine to the trained ML model."
                f"{'<br>Engine note: ' + escape(str(_ENGINE_NOTE)) if _ENGINE_NOTE else ''}"
                f"{'<br>Hosted deployment: history is kept for this browser session only.' if is_hosted() else ''}</span></div>",
                unsafe_allow_html=True)
    if st.button("🗑️  Clear analysis history"):
        ss = st.session_state
        ss.history, ss.last_result, ss.live_log = [], None, []
        ss.manual_result, ss.live_snapshot = None, None
        try:
            HISTORY_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        st.rerun()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
{
    NAV[0]: page_dashboard, NAV[1]: page_manual, NAV[2]: page_live,
    NAV[3]: page_alerts, NAV[4]: page_reports, NAV[5]: page_settings,
}[page]()
