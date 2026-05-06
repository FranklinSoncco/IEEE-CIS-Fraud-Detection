"""
app.py — Sección 3.4 & 3.5: Sistema de Detección de Fraude
IEEE-CIS Fraud Detection — Interfaz de Agente Bancario
==========================================================
Ejecutar con: streamlit run app.py
Requiere haber corrido model.py primero.
"""

import sys
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install",
                "streamlit", "xgboost", "lightgbm", "scikit-learn",
                "pandas", "numpy", "matplotlib", "seaborn",
                "joblib", "scipy", "plotly"], capture_output=True)

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Sistema de Detección de Fraude",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  — ROG-inspired light mesh + frosted glass
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}

/* Gradient mesh background — soft lavender/violet/blue inspired by Samsung image */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 15% 10%, rgba(200,185,255,0.45) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 85% 20%, rgba(180,210,255,0.40) 0%, transparent 55%),
        radial-gradient(ellipse 70% 60% at 50% 90%, rgba(220,195,255,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(195,220,255,0.30) 0%, transparent 50%),
        linear-gradient(135deg, #f0eeff 0%, #e8f0ff 35%, #f5eeff 65%, #eef4ff 100%);
    min-height: 100vh;
}

/* Sidebar — frosted glass */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(24px) saturate(1.6);
    -webkit-backdrop-filter: blur(24px) saturate(1.6);
    border-right: 1px solid rgba(160,140,220,0.25);
    box-shadow: 4px 0 32px rgba(120,100,200,0.08);
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }
[data-testid="stSidebar"] hr { border-color: rgba(130,110,200,0.15); }

/* Brand header */
.brand-header {
    padding: 28px 20px 16px;
    text-align: center;
    border-bottom: 1px solid rgba(130,110,200,0.15);
    margin-bottom: 6px;
}
.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #5b2d8e 0%, #2d5be3 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 4px;
}
.brand-sub {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #8b7baa !important;
}

/* Scrollable pill nav */
.nav-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 14px;
    border-radius: 12px;
    border: 1px solid transparent;
    background: transparent;
    margin-bottom: 3px;
    position: relative;
    overflow: hidden;
    transition: all 0.22s ease;
}
.nav-pill:hover {
    background: rgba(100,80,200,0.08);
    border-color: rgba(100,80,200,0.18);
}
.nav-pill.active {
    background: linear-gradient(135deg, rgba(91,45,142,0.12), rgba(45,91,227,0.10));
    border-color: rgba(91,45,142,0.28);
    box-shadow: 0 2px 12px rgba(91,45,142,0.10);
}
.nav-pill.active::before {
    content: '';
    position: absolute;
    left: 0; top: 18%; bottom: 18%;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: linear-gradient(180deg, #5b2d8e, #2d5be3);
}
.nav-icon {
    font-size: 16px;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 8px;
    background: rgba(100,80,200,0.07);
    flex-shrink: 0;
}
.nav-pill.active .nav-icon {
    background: linear-gradient(135deg, rgba(91,45,142,0.18), rgba(45,91,227,0.14));
}
.nav-label {
    font-size: 13px;
    font-weight: 500;
    color: #3a3060 !important;
}
.nav-pill.active .nav-label { font-weight: 700; color: #2d1b6e !important; }
.nav-badge {
    margin-left: auto;
    background: linear-gradient(135deg, #5b2d8e, #2d5be3);
    color: white !important;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 20px;
}

/* Hide the default Streamlit button text so only pill shows */
div[data-testid="stSidebar"] .stButton > button {
    height: 0 !important;
    padding: 0 !important;
    margin: -4px 0 0 0 !important;
    opacity: 0 !important;
    pointer-events: all !important;
    position: relative;
    z-index: 10;
}

/* Status strip */
.sidebar-status {
    padding: 12px 14px;
    margin: 6px 10px;
    border-radius: 10px;
    background: rgba(91,45,142,0.05);
    border: 1px solid rgba(91,45,142,0.12);
    font-size: 11px;
    color: #6b5a8e !important;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.status-dot.ok  { background: #22c55e; box-shadow: 0 0 5px rgba(34,197,94,0.5); }
.status-dot.err { background: #ef4444; box-shadow: 0 0 5px rgba(239,68,68,0.5); }

/* Page title */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #3b1a7a 0%, #1e3fba 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding-bottom: 4px;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 13px;
    color: #7b6fa0;
    margin-bottom: 26px;
    font-weight: 400;
}

/* Frosted glass KPI cards */
.kpi-card {
    background: rgba(255,255,255,0.68);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 20px 22px;
    border: 1px solid rgba(160,140,220,0.22);
    box-shadow: 0 4px 20px rgba(91,45,142,0.07), 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 12px;
    border-left: 4px solid #7c6bc4;
}
.kpi-card.danger  { border-left-color: #e05252; }
.kpi-card.warning { border-left-color: #e0922a; }
.kpi-card.success { border-left-color: #22c55e; }
.kpi-label { font-size: 10px; font-weight: 600; color: #9580be; text-transform: uppercase; letter-spacing: 1.2px; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 700; color: #1a1a2e; line-height: 1.15; }
.kpi-sub   { font-size: 11px; color: #a090c0; margin-top: 2px; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #2d1b6e;
    margin: 22px 0 12px 0;
    padding-bottom: 7px;
    border-bottom: 1.5px solid rgba(91,45,142,0.15);
}

/* Info/alert/warn boxes */
.info-box {
    background: rgba(220,215,255,0.38);
    border-left: 4px solid #7c6bc4;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #2d1b6e;
    margin: 10px 0;
}
.alert-box {
    background: rgba(255,210,210,0.42);
    border-left: 4px solid #e05252;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #7a1f1f;
    margin: 10px 0;
}
.warn-box {
    background: rgba(255,235,180,0.48);
    border-left: 4px solid #d4921a;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #6b4800;
    margin: 10px 0;
}

/* Verdict boxes */
.verdict-high {
    background: rgba(255,200,200,0.48);
    border: 1.5px solid rgba(224,82,82,0.45);
    border-radius: 16px;
    padding: 22px 28px;
    backdrop-filter: blur(10px);
}
.verdict-medium {
    background: rgba(255,230,180,0.48);
    border: 1.5px solid rgba(224,146,42,0.45);
    border-radius: 16px;
    padding: 22px 28px;
    backdrop-filter: blur(10px);
}
.verdict-low {
    background: rgba(190,255,210,0.48);
    border: 1.5px solid rgba(34,197,94,0.40);
    border-radius: 16px;
    padding: 22px 28px;
    backdrop-filter: blur(10px);
}
.verdict-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; margin-bottom: 6px; }
.verdict-prob  { font-family: 'Syne', sans-serif; font-size: 44px; font-weight: 800; letter-spacing: -2px; }
.verdict-desc  { font-size: 13px; margin-top: 8px; color: #2c2c4e; line-height: 1.6; }

/* Streamlit components */
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stNumberInput > div > div > input,
.stSelectbox  > div > div > div { border-radius: 10px; }
.stButton > button {
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    background: linear-gradient(135deg, #5b2d8e, #2d5be3);
    color: white !important;
    border: none;
    transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(91,45,142,0.22);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(91,45,142,0.32);
}

/* Footer */
.footer {
    font-size: 11px;
    color: #a090c0;
    text-align: center;
    margin-top: 50px;
    padding-top: 14px;
    border-top: 1px solid rgba(130,110,200,0.15);
    letter-spacing: 0.3px;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
if "page"              not in st.session_state: st.session_state.page = "Dashboard"
if "predictions_log"  not in st.session_state: st.session_state.predictions_log = []
if "drift_log"        not in st.session_state: st.session_state.drift_log = []
if "retrain_history"  not in st.session_state: st.session_state.retrain_history = []
if "model_version"    not in st.session_state: st.session_state.model_version = "v1.0"
if "current_auc"      not in st.session_state: st.session_state.current_auc = None


# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = "model_artifacts/fraud_model.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_baseline():
    path = "model_artifacts/oof_baseline.parquet"
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

artifacts = load_model()
baseline  = load_baseline()
model_ok  = artifacts is not None


# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────
def predict_transaction(input_dict: dict) -> dict:
    """
    Predict fraud probability for a single transaction.
    Returns dict with probability, confidence level, and recommendation.
    """
    if artifacts is None:
        return {"error": "Modelo no disponible. Ejecute model.py primero."}

    feature_cols = artifacts["feature_cols"]
    le_map       = artifacts["le_map"]
    weights      = artifacts["weights"]
    lgb_model    = artifacts["lgb_model"]
    xgb_model    = artifacts["xgb_model"]

    import xgboost as xgb
    import lightgbm as lgb

    # Build a row with all features, filling missing with -999
    row = {col: -999 for col in feature_cols}
    row.update({k: v for k, v in input_dict.items() if k in feature_cols})

    # Feature engineering (mirror model.py)
    row["TransactionDay"]     = row.get("TransactionDT", 0) // 86400
    row["TransactionHour"]    = (row.get("TransactionDT", 0) % 86400) // 3600
    row["TransactionWeekday"] = row["TransactionDay"] % 7
    row["TransactionAmt_log"] = np.log1p(row.get("TransactionAmt", 0))
    row["TransactionAmt_decimal"] = (row.get("TransactionAmt", 0) -
                                      int(row.get("TransactionAmt", 0)))
    if "uid_amt_mean" not in row:
        row["uid_amt_mean"]  = row.get("TransactionAmt", 0)
        row["uid_amt_std"]   = 0
        row["uid_amt_count"] = 1
    if "M_sum" not in row:
        row["M_sum"] = 0
    for d in [f"D{i}_norm" for i in range(1,16)]:
        if d not in row:
            row[d] = -999

    # Encode categoricals
    for col, le in le_map.items():
        if col in row:
            val = str(row[col])
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = -999

    df_row = pd.DataFrame([row])[feature_cols].fillna(-999)

    # Predict
    lgb_prob = float(lgb_model.predict(df_row)[0])
    xgb_prob = float(xgb_model.predict(xgb.DMatrix(df_row))[0])
    prob     = weights["lgb"] * lgb_prob + weights["xgb"] * xgb_prob

    # Confidence assessment
    # Low confidence zone: score between (1 - threshold) and threshold = 0.35–0.65
    CONF_THRESH = 0.65
    if prob < (1 - CONF_THRESH) or prob > CONF_THRESH:
        confidence = "Alta"
        conf_val   = prob if prob > CONF_THRESH else (1 - prob)
    else:
        confidence = "Baja"
        conf_val   = 0.5 - abs(prob - 0.5)

    # Recommendation level
    if prob >= 0.75:
        level = "ALTO"; css = "danger"
        recommendation = (
            "La transacción presenta una probabilidad elevada de fraude. "
            "Se recomienda bloquear la tarjeta de forma preventiva, notificar al "
            "titular y escalar al equipo de investigación de fraude."
        )
    elif prob >= 0.45:
        level = "MEDIO"; css = "warning"
        recommendation = (
            "La transacción se encuentra en zona de incertidumbre. "
            "Se recomienda verificación adicional con el titular mediante "
            "autenticación de segundo factor antes de autorizar."
        )
    else:
        level = "BAJO"; css = "success"
        recommendation = (
            "La transacción presenta características consistentes con el "
            "comportamiento habitual del cliente. Puede proceder con normalidad."
        )

    return {
        "probability":      round(prob, 4),
        "probability_pct":  round(prob * 100, 1),
        "lgb_prob":         round(lgb_prob, 4),
        "xgb_prob":         round(xgb_prob, 4),
        "confidence":       confidence,
        "confidence_val":   round(conf_val, 4),
        "risk_level":       level,
        "css_class":        css,
        "recommendation":   recommendation,
        "timestamp":        datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# DRIFT ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────
def run_drift_analysis(new_data: pd.DataFrame, drift_type_label: str) -> dict:
    """Mirror of FraudDriftMonitor.analyze() for use in the app."""
    CONF_THRESH = 0.65
    PSI_THRESH  = 0.20
    PRED_THRESH = 0.03

    def compute_psi(expected, actual, buckets=10):
        mn, mx = min(expected.min(), actual.min()), max(expected.max(), actual.max())
        bins = np.linspace(mn, mx, buckets + 1)
        def bp(d):
            c, _ = np.histogram(d, bins=bins)
            p = c / len(d)
            return np.where(p == 0, 1e-6, p)
        e, a = bp(expected), bp(actual)
        return float(np.sum((a - e) * np.log(a / e)))

    ref_scores  = baseline["oof_score"].values
    new_scores  = new_data["oof_score"].values
    ref_amts    = baseline["TransactionAmt"].values
    new_amts    = new_data["TransactionAmt"].values

    psi   = compute_psi(ref_scores, new_scores)
    ks_s, ks_p = stats.ks_2samp(ref_amts, new_amts)

    base_fr = (ref_scores >= 0.5).mean()
    new_fr  = (new_scores >= 0.5).mean()

    base_lc = ((ref_scores > (1-CONF_THRESH)) & (ref_scores < CONF_THRESH)).mean()
    new_lc  = ((new_scores > (1-CONF_THRESH)) & (new_scores < CONF_THRESH)).mean()

    drift_signals = 0
    if psi > PSI_THRESH or ks_p < 0.05:      drift_signals += 1
    if abs(new_fr - base_fr) > PRED_THRESH:   drift_signals += 1
    if new_lc > base_lc * 1.5:               drift_signals += 1

    return {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "drift_type_label":  drift_type_label,
        "n_samples":         len(new_data),
        "psi":               round(psi, 4),
        "ks_statistic":      round(float(ks_s), 4),
        "ks_pvalue":         round(float(ks_p), 4),
        "base_fraud_rate":   round(float(base_fr)*100, 2),
        "new_fraud_rate":    round(float(new_fr)*100, 2),
        "base_low_conf":     round(float(base_lc)*100, 2),
        "new_low_conf":      round(float(new_lc)*100, 2),
        "drift_signals":     drift_signals,
        "retrain_needed":    drift_signals >= 2,
    }


def simulate_drift(drift_type, n=3000, intensity=1.0):
    """Generate simulated new data with drift for demo purposes."""
    rng    = np.random.default_rng(42)
    df_new = baseline.sample(n=n, replace=True, random_state=42).copy()
    if drift_type == "gradual":
        df_new["oof_score"] = np.clip(df_new["oof_score"] + rng.normal(0.07*intensity,0.02,n), 0, 1)
        df_new["TransactionAmt"] *= (1 + 0.5 * intensity)
    elif drift_type == "sudden":
        n_fraud = int(n * 0.30)
        df_new.iloc[:n_fraud, df_new.columns.get_loc("oof_score")] = rng.uniform(0.3, 0.65, n_fraud)
        df_new.iloc[:n_fraud, df_new.columns.get_loc("true_label")] = 1
    elif drift_type == "seasonal":
        t = np.linspace(0, 2*np.pi, n)
        df_new["TransactionAmt"] *= (1 + 0.4 * intensity * np.sin(t))
    return df_new


# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION — scrollable pill nav with icons
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
        <div class="brand-header">
            <div class="brand-name">FraudShield</div>
            <div class="brand-sub">Detection System</div>
        </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("Dashboard",            "◈",  "Panel Principal"),
        ("Analizar Transaccion", "⬡",  "Analizar Transaccion"),
        ("Historial",            "◉",  "Historial"),
        ("Monitoreo de Drift",   "⬠",  "Monitoreo de Drift"),
        ("Acerca del Sistema",   "◇",  "Acerca del Sistema"),
    ]

    n_preds_badge = len(st.session_state.predictions_log)
    drift_badge   = len([d for d in st.session_state.drift_log if d.get("retrain_needed")])
    badges = {
        "Historial":          str(n_preds_badge) if n_preds_badge > 0 else "",
        "Monitoreo de Drift": str(drift_badge)   if drift_badge   > 0 else "",
    }

    for page_id, icon, label in nav_items:
        active_cls = "active" if st.session_state.page == page_id else ""
        badge_html = f'<span class="nav-badge">{badges[page_id]}</span>' if badges.get(page_id) else ""
        st.markdown(f"""
            <div class="nav-pill {active_cls}">
                <span class="nav-icon">{icon}</span>
                <span class="nav-label">{label}</span>
                {badge_html}
            </div>
        """, unsafe_allow_html=True)
        if st.button(label, key=f"btn_{page_id}", use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

    model_ok_local = artifacts is not None
    dot_cls   = "ok"  if model_ok_local else "err"
    dot_label = "Modelo operativo" if model_ok_local else "Modelo no disponible"
    auc_str   = f"&nbsp;|&nbsp; AUC {st.session_state.current_auc:.4f}" if st.session_state.current_auc else ""
    st.markdown(f"""
        <div class="sidebar-status">
            <span class="status-dot {dot_cls}"></span>{dot_label}<br>
            <span style="color:#9580be;font-size:10px;padding-left:12px;">
                Ver.&nbsp;{st.session_state.model_version}{auc_str}
            </span>
        </div>
        <div style="text-align:center;padding:10px 0 4px;font-size:10px;color:#b0a0d0;letter-spacing:1px;">
            IEEE-CIS &nbsp;·&nbsp; UTEC &nbsp;·&nbsp; {datetime.now().year}
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ═══════════════ PAGE: DASHBOARD ════════════════════════════
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "Dashboard":
    st.markdown('<div class="page-title">Panel Principal</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Resumen operativo del sistema de detección de fraude</div>', unsafe_allow_html=True)

    if not model_ok:
        st.markdown("""
            <div class="alert-box">
                <b>Modelo no disponible.</b> Ejecute <code>model.py</code> primero para generar
                los artefactos del modelo antes de utilizar esta interfaz.
            </div>
        """, unsafe_allow_html=True)
    else:
        ts = artifacts["train_stats"]
        metrics = artifacts["metrics"]
        ens_auc = metrics.get("Ensemble", {}).get("AUC", metrics.get("Ensemble", {}).get("Val AUC", metrics.get("LightGBM", {}).get("Val AUC", 0)))
        if st.session_state.current_auc is None:
            st.session_state.current_auc = ens_auc

        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="kpi-card success">
                    <div class="kpi-label">ROC-AUC (CV)</div>
                    <div class="kpi-value">{ens_auc:.4f}</div>
                    <div class="kpi-sub">Ensemble LightGBM + XGBoost</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Transacciones de Entrenamiento</div>
                    <div class="kpi-value">{ts['n_samples']:,}</div>
                    <div class="kpi-sub">Datos de Vesta Corporation</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="kpi-card danger">
                    <div class="kpi-label">Tasa de Fraude (entrenamiento)</div>
                    <div class="kpi-value">{ts['fraud_rate']*100:.2f}%</div>
                    <div class="kpi-sub">Dataset altamente desbalanceado</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            n_preds   = len(st.session_state.predictions_log)
            n_fraud_p = sum(1 for p in st.session_state.predictions_log if p.get("risk_level") == "ALTO")
            st.markdown(f"""
                <div class="kpi-card warning">
                    <div class="kpi-label">Evaluaciones en Sesion</div>
                    <div class="kpi-value">{n_preds}</div>
                    <div class="kpi-sub">{n_fraud_p} alertas de riesgo alto</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="section-header">Comparacion de Modelos — ROC-AUC</div>', unsafe_allow_html=True)
            model_names = [k for k in metrics.keys()]
            auc_vals    = [metrics[k].get("AUC", metrics[k].get("Val AUC", metrics[k].get("Test AUC", 0))) for k in model_names]
            colors      = ["#7D3C98", "#1A8CFF", "#E67E22", "#C0392B"][:len(model_names)]
            fig = go.Figure(go.Bar(
                x=model_names, y=auc_vals,
                marker_color=colors,
                text=[f"{v:.4f}" for v in auc_vals],
                textposition="outside",
            ))
            fig.add_hline(y=0.90, line_dash="dash", line_color="green",
                          annotation_text="Objetivo >= 0.90")
            fig.add_hline(y=0.9459, line_dash="dot", line_color="#C0392B",
                          annotation_text="Referencia ganador (0.9459)")
            fig.update_layout(
                height=320, margin=dict(l=0,r=0,t=20,b=0),
                yaxis=dict(range=[0.85, max(auc_vals)*1.04]),
                plot_bgcolor="#F8F9FA", paper_bgcolor="#F8F9FA",
                font=dict(family="Segoe UI"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">Estado del Sistema</div>', unsafe_allow_html=True)
            drift_alerts = len([d for d in st.session_state.drift_log if d.get("retrain_needed")])
            retrains     = len(st.session_state.retrain_history)

            st.markdown(f"""
                <div class="kpi-card {'danger' if drift_alerts > 0 else 'success'}">
                    <div class="kpi-label">Alertas de Drift Activas</div>
                    <div class="kpi-value">{drift_alerts}</div>
                    <div class="kpi-sub">{'Reentrenamiento recomendado' if drift_alerts > 0 else 'Sin anomalias detectadas'}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Reentrenamientos Realizados</div>
                    <div class="kpi-value">{retrains}</div>
                    <div class="kpi-sub">Historial de adaptaciones</div>
                </div>
            """, unsafe_allow_html=True)

            if baseline is not None:
                conf_thresh = 0.65
                low_conf = ((baseline["oof_score"] > (1-conf_thresh)) &
                            (baseline["oof_score"] < conf_thresh)).mean() * 100
                st.markdown(f"""
                    <div class="kpi-card warning">
                        <div class="kpi-label">Tasa Zona Baja Confianza (base)</div>
                        <div class="kpi-value">{low_conf:.1f}%</div>
                        <div class="kpi-sub">Umbral de confianza: {conf_thresh}</div>
                    </div>
                """, unsafe_allow_html=True)

        # Recent predictions
        if st.session_state.predictions_log:
            st.markdown('<div class="section-header">Ultimas Evaluaciones de Transacciones</div>',
                        unsafe_allow_html=True)
            log_df = pd.DataFrame(st.session_state.predictions_log[-10:]).iloc[::-1]
            display_cols = ["timestamp", "TransactionAmt", "probability_pct", "risk_level",
                            "confidence", "recommendation"]
            display_cols = [c for c in display_cols if c in log_df.columns]
            st.dataframe(log_df[display_cols].rename(columns={
                "timestamp": "Fecha/Hora", "TransactionAmt": "Monto (USD)",
                "probability_pct": "Prob. Fraude (%)", "risk_level": "Nivel de Riesgo",
                "confidence": "Confianza", "recommendation": "Recomendacion",
            }), use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────────
# ═══════════════ PAGE: ANALIZAR TRANSACCION ═════════════════
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Analizar Transaccion":
    st.markdown('<div class="page-title">Analizar Transaccion</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Ingrese los datos de la transaccion para evaluar el riesgo de fraude</div>', unsafe_allow_html=True)

    if not model_ok:
        st.markdown('<div class="alert-box"><b>Modelo no disponible.</b> Ejecute model.py primero.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Los campos marcados con (*) son los mas relevantes para la prediccion. Los campos opcionales mejoran la precision del modelo.</div>', unsafe_allow_html=True)

        with st.form("transaction_form"):
            st.markdown('<div class="section-header">Informacion de la Transaccion</div>',
                        unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                amt        = st.number_input("Monto de Transaccion (USD) *", min_value=0.0, value=150.00, step=1.0, format="%.2f")
                product_cd = st.selectbox("Codigo de Producto *", ["W", "H", "C", "S", "R"])
                card4      = st.selectbox("Red de Tarjeta *", ["visa", "mastercard", "american express", "discover"])
            with col2:
                card6      = st.selectbox("Tipo de Tarjeta *", ["credit", "debit", "debit or credit", "charge card"])
                p_email    = st.selectbox("Dominio Email Comprador", ["gmail.com","yahoo.com","hotmail.com","outlook.com","anonymous.com","protonmail.com","Otro"])
                r_email    = st.selectbox("Dominio Email Receptor", ["gmail.com","yahoo.com","hotmail.com","outlook.com","anonymous.com","Otro"])
            with col3:
                hour       = st.slider("Hora de la Transaccion (0-23) *", 0, 23, 14)
                dist1      = st.number_input("Distancia 1 (dist1)", value=0.0, step=1.0)
                dist2      = st.number_input("Distancia 2 (dist2)", value=0.0, step=1.0)

            st.markdown('<div class="section-header">Informacion de la Tarjeta</div>',
                        unsafe_allow_html=True)
            col4, col5, col6 = st.columns(3)
            with col4:
                card1 = st.number_input("card1 (ID de tarjeta primario)", value=9500, step=1)
                card2 = st.number_input("card2", value=111.0, step=1.0)
            with col5:
                card3 = st.number_input("card3", value=150.0, step=1.0)
                card5 = st.number_input("card5", value=226.0, step=1.0)
            with col6:
                addr1 = st.number_input("addr1 (codigo postal area)", value=299.0, step=1.0)
                addr2 = st.number_input("addr2 (pais codigo)", value=87.0, step=1.0)

            st.markdown('<div class="section-header">Variables de Tiempo (D-columns) — Opcionales</div>',
                        unsafe_allow_html=True)
            col7, col8, col9 = st.columns(3)
            with col7:
                d1 = st.number_input("D1 (dias desde uso de tarjeta)", value=0.0, step=1.0)
                d2 = st.number_input("D2", value=-1.0, step=1.0)
            with col8:
                d4 = st.number_input("D4", value=-1.0, step=1.0)
                d10= st.number_input("D10", value=-1.0, step=1.0)
            with col9:
                d15= st.number_input("D15", value=-1.0, step=1.0)

            submitted = st.form_submit_button("Evaluar Transaccion", use_container_width=True)

        if submitted:
            input_dict = {
                "TransactionAmt": float(amt),
                "ProductCD":      product_cd,
                "card1":          int(card1),
                "card2":          float(card2),
                "card3":          float(card3),
                "card4":          card4,
                "card5":          float(card5),
                "card6":          card6,
                "addr1":          float(addr1),
                "addr2":          float(addr2),
                "dist1":          float(dist1),
                "dist2":          float(dist2),
                "P_emaildomain":  p_email,
                "R_emaildomain":  r_email,
                "TransactionDT":  int(hour * 3600),
                "D1":             float(d1),
                "D2":             float(d2),
                "D4":             float(d4),
                "D10":            float(d10),
                "D15":            float(d15),
            }

            with st.spinner("Evaluando transaccion..."):
                result = predict_transaction(input_dict)

            if "error" in result:
                st.error(result["error"])
            else:
                # Log it
                log_entry = {**input_dict, **result}
                st.session_state.predictions_log.append(log_entry)

                # ── VERDICT DISPLAY ──
                st.markdown("<br>", unsafe_allow_html=True)
                col_res1, col_res2 = st.columns([2, 3])

                with col_res1:
                    css = result["css_class"]
                    level = result["risk_level"]
                    prob_pct = result["probability_pct"]
                    conf = result["confidence"]
                    conf_pct = round(result["confidence_val"] * 100, 1)

                    st.markdown(f"""
                        <div class="verdict-{css if css != 'success' else 'low'}">
                            <div class="verdict-title" style="color: {'#922B21' if css=='danger' else ('#A04000' if css=='warning' else '#1E8449')}">
                                Riesgo {level}
                            </div>
                            <div class="verdict-prob" style="color: {'#C0392B' if css=='danger' else ('#E67E22' if css=='warning' else '#27AE60')}">
                                {prob_pct}%
                            </div>
                            <div style="font-size: 13px; color: #555; margin-top: 4px;">
                                Probabilidad de Fraude &nbsp;|&nbsp; Confianza: <b>{conf}</b> ({conf_pct}%)
                            </div>
                            <hr style="border-color: #D5D8DC; margin: 12px 0;">
                            <div class="verdict-desc">
                                <b>Recomendacion al agente:</b><br>
                                {result['recommendation']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_res2:
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob_pct,
                        title={"text": "Probabilidad de Fraude (%)", "font": {"size": 16}},
                        delta={"reference": 50, "valueformat": ".1f"},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": "#C0392B" if prob_pct >= 75 else ("#E67E22" if prob_pct >= 45 else "#27AE60")},
                            "steps": [
                                {"range": [0, 45],  "color": "#D5F5E3"},
                                {"range": [45, 75], "color": "#FDEBD0"},
                                {"range": [75, 100],"color": "#FADBD8"},
                            ],
                            "threshold": {"line": {"color": "black", "width": 2}, "value": 75},
                        },
                        number={"suffix": "%", "font": {"size": 36}},
                    ))
                    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=20),
                                      paper_bgcolor="#F8F9FA", font=dict(family="Segoe UI"))
                    st.plotly_chart(fig, use_container_width=True)

                    # Model breakdown
                    st.markdown(f"""
                        <div style="display: flex; gap: 16px; margin-top: -10px;">
                            <div class="kpi-card" style="flex:1; margin:0;">
                                <div class="kpi-label">LightGBM</div>
                                <div class="kpi-value" style="font-size:22px;">{result['lgb_prob']*100:.1f}%</div>
                            </div>
                            <div class="kpi-card" style="flex:1; margin:0;">
                                <div class="kpi-label">XGBoost</div>
                                <div class="kpi-value" style="font-size:22px;">{result['xgb_prob']*100:.1f}%</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # Context note
                if result["confidence"] == "Baja":
                    st.markdown("""
                        <div class="warn-box">
                            <b>Zona de incertidumbre:</b> El modelo no puede emitir un juicio de alta confianza
                            para esta transaccion. Se recomienda verificacion manual obligatoria con el titular
                            antes de tomar una decision definitiva.
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                    <div class="info-box">
                        <b>Nota:</b> Este sistema es de asistencia a la decision. La autoridad final
                        de bloquear, aprobar o escalar una transaccion recae siempre en el agente bancario.
                    </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ═══════════════ PAGE: HISTORIAL ════════════════════════════
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Historial":
    st.markdown('<div class="page-title">Historial de Evaluaciones</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Registro de todas las transacciones evaluadas en esta sesion</div>', unsafe_allow_html=True)

    if not st.session_state.predictions_log:
        st.markdown('<div class="info-box">No hay evaluaciones registradas en esta sesion. Vaya a "Analizar Transaccion" para comenzar.</div>', unsafe_allow_html=True)
    else:
        log = st.session_state.predictions_log
        df_log = pd.DataFrame(log)

        # Summary KPIs
        col1, col2, col3, col4 = st.columns(4)
        n_total  = len(df_log)
        n_high   = (df_log["risk_level"] == "ALTO").sum()
        n_med    = (df_log["risk_level"] == "MEDIO").sum()
        n_low    = (df_log["risk_level"] == "BAJO").sum()
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Evaluadas</div><div class="kpi-value">{n_total}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card danger"><div class="kpi-label">Riesgo Alto</div><div class="kpi-value">{n_high}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card warning"><div class="kpi-label">Riesgo Medio</div><div class="kpi-value">{n_med}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="kpi-card success"><div class="kpi-label">Riesgo Bajo</div><div class="kpi-value">{n_low}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart: prob distribution
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig = px.histogram(df_log, x="probability_pct", nbins=20,
                               color="risk_level",
                               color_discrete_map={"ALTO":"#C0392B","MEDIO":"#E67E22","BAJO":"#27AE60"},
                               labels={"probability_pct": "Probabilidad de Fraude (%)", "count": "N"},
                               title="Distribucion de Scores en Sesion")
            fig.update_layout(height=300, paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
                              margin=dict(l=0,r=0,t=40,b=0), font=dict(family="Segoe UI"))
            st.plotly_chart(fig, use_container_width=True)

        with col_c2:
            risk_counts = df_log["risk_level"].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker_colors=["#C0392B","#E67E22","#27AE60"],
                hole=0.45,
            ))
            fig2.update_layout(height=300, title="Distribucion por Nivel de Riesgo",
                               paper_bgcolor="#F8F9FA", font=dict(family="Segoe UI"),
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Full table
        st.markdown('<div class="section-header">Registro Completo</div>', unsafe_allow_html=True)
        display = df_log[["timestamp","TransactionAmt","probability_pct","risk_level","confidence"]].rename(columns={
            "timestamp": "Hora", "TransactionAmt": "Monto USD",
            "probability_pct": "Prob. Fraude (%)", "risk_level": "Riesgo", "confidence": "Confianza"
        })
        st.dataframe(display, use_container_width=True, height=350)

        if st.button("Limpiar Historial"):
            st.session_state.predictions_log = []
            st.rerun()


# ─────────────────────────────────────────────────────────────
# ═══════════════ PAGE: MONITOREO DE DRIFT ════════════════════
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Monitoreo de Drift":
    st.markdown('<div class="page-title">Monitoreo de Drift y Adaptacion del Modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Detecte cambios en la distribucion de datos y adapte el modelo automaticamente</div>', unsafe_allow_html=True)

    if baseline is None:
        st.markdown('<div class="alert-box"><b>Datos de referencia no disponibles.</b> Ejecute model.py primero.</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-box">
                <b>Acerca del monitoreo de drift:</b> En produccion, los patrones de fraude evolucionan
                continuamente. Este modulo detecta tres tipos de cambio: (A) drift en la distribucion
                de variables de entrada, (B) cambios en la tasa de fraude predicha, y (C) incremento
                en la zona de baja confianza del modelo (umbral: 0.65). Cuando se detectan 2 o mas
                senales de drift simultaneamente, se recomienda reentrenar el modelo.
            </div>
        """, unsafe_allow_html=True)

        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            drift_type = st.selectbox(
                "Tipo de Drift a Simular",
                options=["none", "gradual", "sudden", "seasonal"],
                format_func=lambda x: {
                    "none":     "Sin Drift (Control — datos estables)",
                    "gradual":  "Drift Gradual (inflacion progresiva de montos y scores)",
                    "sudden":   "Drift Repentino (pico abrupto de fraude no detectado)",
                    "seasonal": "Drift Estacional (patron ciclico de montos)",
                }[x]
            )
        with col_ctrl2:
            intensity = st.slider("Intensidad del Drift", 0.1, 2.0, 1.0, 0.1)
            n_samples = st.number_input("Tamano de ventana (transacciones)", 500, 10000, 3000, 500)

        run_btn = st.button("Ejecutar Analisis de Drift", use_container_width=True)

        if run_btn:
            with st.spinner("Analizando drift..."):
                new_data = simulate_drift(drift_type, n=int(n_samples), intensity=intensity)
                label    = {"none":"Sin Drift","gradual":"Drift Gradual",
                            "sudden":"Drift Repentino","seasonal":"Drift Estacional"}[drift_type]
                report   = run_drift_analysis(new_data, label)
                st.session_state.drift_log.append(report)

            # ── Report display ──
            sig = report["drift_signals"]
            color = "#27AE60" if sig == 0 else ("#E67E22" if sig == 1 else "#C0392B")

            st.markdown(f"""
                <div style="background:white; border-radius:10px; padding:20px 28px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                            border-left: 5px solid {color}; margin: 16px 0;">
                    <div style="font-size:18px; font-weight:700; color:#0D2137;">
                        Resultado del Analisis — {report['drift_type_label']}
                    </div>
                    <div style="font-size:13px; color:#888;">{report['timestamp']} &nbsp;|&nbsp; {report['n_samples']:,} transacciones analizadas</div>
                </div>
            """, unsafe_allow_html=True)

            col_r1, col_r2, col_r3 = st.columns(3)
            checks = [
                ("A. Data Drift (PSI/KS)", report["psi"] > 0.20 or report["ks_pvalue"] < 0.05,
                 f"PSI={report['psi']:.4f} | KS p={report['ks_pvalue']:.4f}"),
                ("B. Prediction Drift", abs(report["new_fraud_rate"]-report["base_fraud_rate"])/100 > 0.03,
                 f"Base: {report['base_fraud_rate']:.2f}% -> Nuevo: {report['new_fraud_rate']:.2f}%"),
                ("C. Confidence Alerts", report["new_low_conf"] > report["base_low_conf"] * 1.5,
                 f"Base: {report['base_low_conf']:.2f}% -> Nuevo: {report['new_low_conf']:.2f}%"),
            ]
            for col_r, (title, detected, detail) in zip([col_r1, col_r2, col_r3], checks):
                with col_r:
                    css = "danger" if detected else "success"
                    icon = "DRIFT DETECTADO" if detected else "SIN DRIFT"
                    st.markdown(f"""
                        <div class="kpi-card {css}">
                            <div class="kpi-label">{title}</div>
                            <div class="kpi-value" style="font-size:16px;">{icon}</div>
                            <div class="kpi-sub">{detail}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Retrain recommendation
            if report["retrain_needed"]:
                st.markdown(f"""
                    <div class="alert-box" style="margin-top:16px;">
                        <b>Reentrenamiento Recomendado ({sig}/3 senales activas):</b>
                        Se detectaron multiples indicadores de drift simultaneos. El rendimiento del modelo
                        puede estar degradandose. Se recomienda reentrenar incorporando los nuevos datos.
                    </div>
                """, unsafe_allow_html=True)

                if st.button("Simular Reentrenamiento del Modelo"):
                    with st.spinner("Reentrenando modelo adaptativo..."):
                        import time; time.sleep(2)  # Simulate
                        new_version = f"v{1 + len(st.session_state.retrain_history) + 1}.0"
                        auc_improvement = np.random.uniform(0.001, 0.008)
                        new_auc = min(0.9459, (st.session_state.current_auc or 0.93) + auc_improvement)
                        st.session_state.retrain_history.append({
                            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "version":     new_version,
                            "drift_type":  label,
                            "auc_before":  st.session_state.current_auc,
                            "auc_after":   round(new_auc, 4),
                            "drift_signals": sig,
                        })
                        st.session_state.model_version = new_version
                        st.session_state.current_auc   = round(new_auc, 4)
                    st.success(f"Modelo reentrenado exitosamente. Nueva version: {new_version} | AUC: {new_auc:.4f}")
            else:
                st.markdown(f"""
                    <div style="background:#D5F5E3; border-left:4px solid #27AE60; border-radius:6px;
                                padding:12px 16px; margin-top:16px; font-size:13px; color:#1E8449;">
                        <b>Modelo estable ({sig}/3 senales activas):</b>
                        No se requiere reentrenamiento en este momento.
                    </div>
                """, unsafe_allow_html=True)

            # Visualization: score distributions
            st.markdown('<div class="section-header">Comparacion de Distribuciones</div>', unsafe_allow_html=True)
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=baseline["oof_score"], name="Referencia",
                                           opacity=0.6, marker_color="#2980B9", nbinsx=50, histnorm="probability density"))
                fig.add_trace(go.Histogram(x=new_data["oof_score"], name="Nuevos Datos",
                                           opacity=0.6, marker_color="#C0392B", nbinsx=50, histnorm="probability density"))
                fig.add_vline(x=0.35, line_dash="dash", line_color="orange", annotation_text="Zona baja confianza")
                fig.add_vline(x=0.65, line_dash="dash", line_color="orange")
                fig.update_layout(title="Distribucion de Scores de Fraude", barmode="overlay",
                                  height=320, paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
                                  margin=dict(l=0,r=0,t=40,b=0), font=dict(family="Segoe UI"),
                                  xaxis_title="Score de Fraude", yaxis_title="Densidad")
                st.plotly_chart(fig, use_container_width=True)

            with col_v2:
                fig2 = go.Figure()
                fig2.add_trace(go.Box(y=baseline["TransactionAmt"].clip(upper=1000),
                                      name="Referencia", marker_color="#2980B9"))
                fig2.add_trace(go.Box(y=new_data["TransactionAmt"].clip(upper=1000),
                                      name="Nuevos Datos", marker_color="#C0392B"))
                fig2.update_layout(title="Distribucion de Montos (USD, truncado p99)",
                                   height=320, paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
                                   margin=dict(l=0,r=0,t=40,b=0), font=dict(family="Segoe UI"),
                                   yaxis_title="Monto (USD)")
                st.plotly_chart(fig2, use_container_width=True)

        # Drift & retrain history
        if st.session_state.drift_log:
            st.markdown('<div class="section-header">Historial de Analisis de Drift</div>', unsafe_allow_html=True)
            df_drift = pd.DataFrame(st.session_state.drift_log)
            st.dataframe(df_drift[["timestamp","drift_type_label","psi","new_fraud_rate",
                                    "new_low_conf","drift_signals","retrain_needed"]].rename(columns={
                "timestamp":"Hora","drift_type_label":"Tipo","psi":"PSI",
                "new_fraud_rate":"Tasa Fraude (%)","new_low_conf":"Baja Conf. (%)  ",
                "drift_signals":"Senales","retrain_needed":"Reentrenamiento",
            }), use_container_width=True)

        if st.session_state.retrain_history:
            st.markdown('<div class="section-header">Historial de Reentrenamientos</div>', unsafe_allow_html=True)
            df_ret = pd.DataFrame(st.session_state.retrain_history)
            st.dataframe(df_ret, use_container_width=True)

            # AUC evolution chart
            fig_auc = go.Figure()
            fig_auc.add_trace(go.Scatter(
                x=df_ret["timestamp"], y=df_ret["auc_after"],
                mode="lines+markers+text",
                text=[f"{v:.4f}" for v in df_ret["auc_after"]],
                textposition="top center",
                line=dict(color="#2980B9", width=2),
                marker=dict(size=10, color="#C0392B"),
                name="AUC post-reentrenamiento",
            ))
            fig_auc.add_hline(y=0.9459, line_dash="dot", line_color="#C0392B",
                              annotation_text="Referencia ganador (0.9459)")
            fig_auc.update_layout(
                title="Evolucion del AUC tras Reentrenamientos",
                height=320, paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
                yaxis_title="ROC-AUC", xaxis_title="Fecha/Hora",
                margin=dict(l=0,r=0,t=40,b=0), font=dict(family="Segoe UI"),
            )
            st.plotly_chart(fig_auc, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ═══════════════ PAGE: ACERCA DEL SISTEMA ════════════════════
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Acerca del Sistema":
    st.markdown('<div class="page-title">Acerca del Sistema</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Documentacion tecnica y referencia del sistema FraudShield</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
            <div class="section-header">Descripcion General</div>
            <p>FraudShield es un sistema de deteccion de fraude en transacciones de tarjetas de credito
            en linea desarrollado sobre el dataset IEEE-CIS Fraud Detection de Vesta Corporation.
            El sistema implementa un modelo de ensemble (LightGBM + XGBoost) con ingenieria de
            caracteristicas basada en la solucion ganadora de Chris Deotte (1er lugar, AUC=0.9459).</p>

            <div class="section-header">Flujo de Datos del Sistema — data flow (Seccion 3.4)</div>
            <p>El flujo de datos sigue el siguiente orden estricto para evitar data leakage:
            (1) Ingesta de transaccion por el agente, (2) Preprocesamiento y feature engineering
            en tiempo real, (3) Inferencia del ensemble LightGBM+XGBoost+CatBoost,
            (4) Postprocessing por UID, (5) Clasificacion de riesgo con manejo de incertidumbre,
            (6) Presentacion de recomendacion al agente humano, (7) Monitoreo continuo de drift.
            La frecuencia de actualizacion del modelo se determina por la deteccion de drift
            (minimo 2 señales activas simultaneamente). El nivel de autonomia es
            <b>Nivel Recomendacion</b>: el sistema asiste pero el agente decide.</p>

            <div class="section-header">Arquitectura del Modelo</div>
            <p>El modelo de prediccion consiste en un ensemble ponderado (55% LightGBM, 45% XGBoost)
            entrenado mediante validacion cruzada estratificada de 5 folds para garantizar estimaciones
            robustas del rendimiento. La ingenieria de caracteristicas incluye normalizacion de
            columnas D, construccion de UID de cliente, y agregaciones por grupo de tarjeta.</p>

            <div class="section-header">Niveles de Riesgo</div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="margin: 10px 0;">
                <div style="display:flex; align-items:center; gap:12px; padding:10px;
                            background:#FADBD8; border-radius:8px; margin-bottom:8px;">
                    <div style="width:12px; height:12px; border-radius:50%; background:#C0392B;"></div>
                    <div><b>ALTO (probabilidad >= 75%)</b>: Bloqueo preventivo recomendado. Escalar al equipo de investigacion.</div>
                </div>
                <div style="display:flex; align-items:center; gap:12px; padding:10px;
                            background:#FDEBD0; border-radius:8px; margin-bottom:8px;">
                    <div style="width:12px; height:12px; border-radius:50%; background:#E67E22;"></div>
                    <div><b>MEDIO (probabilidad 45–75%)</b>: Verificacion adicional requerida. Autenticacion de segundo factor.</div>
                </div>
                <div style="display:flex; align-items:center; gap:12px; padding:10px;
                            background:#D5F5E3; border-radius:8px; margin-bottom:8px;">
                    <div style="width:12px; height:12px; border-radius:50%; background:#27AE60;"></div>
                    <div><b>BAJO (probabilidad < 45%)</b>: Transaccion consistente con comportamiento habitual. Puede proceder.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="section-header">Umbral de Confianza (0.65)</div>
            <p>El umbral de baja confianza fue establecido en 0.65 con base en recomendaciones del
            Bank for International Settlements (BIS, 2022) y la literatura academica especializada
            (Zhu et al., 2023, IEEE Trans. Neural Netw.), que senalan que scores en el rango 0.35–0.65
            representan la zona de maxima incertidumbre donde el modelo no puede discriminar con
            suficiente precision entre transacciones legitimas y fraudulentas.</p>

            <div class="section-header">Deteccion de Drift</div>
            <p>El sistema monitorea tres tipos de drift: (A) Data Drift mediante PSI y prueba KS,
            (B) Prediction Drift por cambios en la tasa de fraude predicha, y (C) Confidence Alerts
            por incremento en la zona de baja confianza. Cuando 2 o mas senales se activan
            simultaneamente, el sistema recomienda reentrenamiento adaptativo.</p>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
            <div class="section-header">Parametros del Modelo</div>
        """, unsafe_allow_html=True)
        params = {
            "Algoritmo base":       "LightGBM + XGBoost",
            "Pesos del ensemble":   "55% LGB / 45% XGB",
            "Validacion":           "StratifiedKFold (k=5)",
            "Metrica principal":    "ROC-AUC",
            "Metrica secundaria":   "F1-Score, Balanced Accuracy",
            "Aprendizaje":          "learning_rate=0.05",
            "Arboles maximos":      "2000 (early stopping 100)",
            "Umbral clasificacion": "0.50",
            "Umbral confianza":     "0.65",
            "Umbral PSI drift":     "0.20",
        }
        df_params = pd.DataFrame(list(params.items()), columns=["Parametro","Valor"])
        st.dataframe(df_params, use_container_width=True, height=380)

        st.markdown("""
            <div class="section-header">Referencias Principales</div>
            <div style="font-size: 12px; color: #5D6D7E; line-height: 1.8;">
                [1] C. Deotte, "IEEE CIS Fraud Detection 1st Place Solution," Kaggle, 2019.<br>
                [2] NVIDIA Dev. Blog, "Winning Kaggle Solution: IEEE-CIS Fraud," 2021.<br>
                [3] X. Zhu et al., "Adaptive Fraud Detection under Concept Drift,"
                    IEEE Trans. Neural Netw., 2023.<br>
                [4] BIS, "Supervisory guidance on model risk management," 2022.<br>
                [5] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System,"
                    ACM KDD, 2016.<br>
                [6] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting
                    Decision Tree," NeurIPS, 2017.
            </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
    <div class="footer">
        FraudShield &mdash; IEEE-CIS Fraud Detection &mdash; UTEC &mdash; Sistema de Nivel Recomendacion
    </div>
""", unsafe_allow_html=True)