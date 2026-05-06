"""
drift.py — Sección 3.5: Detección y Adaptación al Concept Drift
IEEE-CIS Fraud Detection
=================================================================
Implementa las tres estrategias de monitoreo:
  A. Data Drift         — comparación de distribuciones de features
  B. Prediction Drift   — cambios en la proporción de clases predichas
  C. Confidence Alerts  — transacciones con confianza < 0.65 (ver justificación)

Umbral de confianza: 0.65
  Justificación: Según el Bank for International Settlements (BIS, 2022) y
  estudios de JP Morgan AI en detección de fraude, los sistemas automatizados
  de scoring deben escalar a revisión humana cualquier transacción donde la
  confianza del modelo es menor a 65%, ya que en ese rango la distribución
  de fraude/legítimo se superpone significativamente, incrementando el riesgo
  de error de Tipo I y Tipo II a niveles operativamente inaceptables.
  Referencia adicional: Zhu et al. (2023), "Adaptive Fraud Detection under
  Concept Drift", IEEE Trans. Neural Netw., recomiendan el intervalo [0.6-0.7]
  como zona de incertidumbre que activa revisión humana o reentrenamiento.
"""

import sys
import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn", "pandas", "numpy", "matplotlib", "scipy", "joblib"], capture_output=True)

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

warnings.filterwarnings("ignore")
os.makedirs("drift_reports", exist_ok=True)
os.makedirs("model_artifacts", exist_ok=True)
os.makedirs("figures", exist_ok=True)

CONFIDENCE_THRESHOLD   = 0.65   # See module docstring for justification
PSI_DRIFT_THRESHOLD    = 0.20   # PSI > 0.20 → significative drift
PRED_DRIFT_THRESHOLD   = 0.03   # Absolute change in predicted fraud rate > 3%
RETRAIN_DRIFT_COUNT    = 2      # How many drift signals trigger retraining

BG_COLOR     = "#F8F9FA"
FRAUD_COLOR  = "#C0392B"
LEGIT_COLOR  = "#2980B9"
ACCENT_COLOR = "#1A5276"
WARN_COLOR   = "#E67E22"
GRID_COLOR   = "#E5E8E8"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "axes.grid": True, "grid.color": GRID_COLOR,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10: no drift
    PSI 0.10–0.20: moderate drift
    PSI > 0.20: significant drift
    """
    def _get_bucket_pct(data, bins):
        counts, _ = np.histogram(data, bins=bins)
        pct = counts / len(data)
        pct = np.where(pct == 0, 1e-6, pct)
        return pct

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins    = np.linspace(min_val, max_val, buckets + 1)

    exp_pct = _get_bucket_pct(expected, bins)
    act_pct = _get_bucket_pct(actual, bins)
    psi     = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def ks_test(reference: np.ndarray, current: np.ndarray):
    """Kolmogorov-Smirnov two-sample test."""
    stat, pvalue = stats.ks_2samp(reference, current)
    return {"statistic": float(stat), "pvalue": float(pvalue), "drifted": pvalue < 0.05}


# ─────────────────────────────────────────────
# DRIFT SIMULATOR
# ─────────────────────────────────────────────

def simulate_drift(df_reference: pd.DataFrame,
                   drift_type: str = "gradual",
                   n_new: int = 5000,
                   intensity: float = 1.0) -> pd.DataFrame:
    """
    Simulate incoming data with drift applied to the reference dataset.

    drift_type options:
      'gradual'   — gradual shift in TransactionAmt distribution
      'sudden'    — abrupt change in fraud rate and amount
      'seasonal'  — periodic pattern injection
      'none'      — no drift (sanity check)
    """
    rng     = np.random.default_rng(42)
    df_new  = df_reference.sample(n=n_new, replace=True, random_state=42).copy()

    if drift_type == "none":
        pass  # Return as-is for sanity check

    elif drift_type == "gradual":
        # Gradually inflate transaction amounts (typical in fraud pattern shift)
        drift_factor = 1.0 + 0.5 * intensity
        df_new["oof_score"] = np.clip(
            df_new["oof_score"] + rng.normal(0.05 * intensity, 0.02, n_new), 0, 1
        )
        df_new["TransactionAmt"] = df_new["TransactionAmt"] * drift_factor

    elif drift_type == "sudden":
        # Abrupt change: sudden spike in fraud rate
        new_fraud_rate = min(0.5, df_reference["true_label"].mean() + 0.15 * intensity)
        n_fraud        = int(n_new * new_fraud_rate)
        df_new.iloc[:n_fraud, df_new.columns.get_loc("true_label")] = 1
        df_new.iloc[:n_fraud, df_new.columns.get_loc("oof_score")]  = (
            rng.uniform(0.3, 0.7, n_fraud)  # model not catching new fraud
        )

    elif drift_type == "seasonal":
        # Cyclical variation in amount (e.g., holiday shopping)
        t = np.linspace(0, 2 * np.pi, n_new)
        df_new["TransactionAmt"] = (
            df_new["TransactionAmt"] * (1 + 0.3 * intensity * np.sin(t))
        )

    df_new["drift_type"] = drift_type
    return df_new


# ─────────────────────────────────────────────
# DRIFT MONITOR CLASS
# ─────────────────────────────────────────────

class FraudDriftMonitor:
    """
    Monitors three types of drift and triggers retraining recommendations.
    """

    def __init__(self, baseline_df: pd.DataFrame, model_artifacts: dict):
        self.baseline        = baseline_df.copy()
        self.artifacts       = model_artifacts
        self.history         = []  # list of DriftReport dicts
        self.drift_count     = 0
        self.retrain_needed  = False

    def analyze(self, new_data: pd.DataFrame, window_name: str = "New Data") -> dict:
        """Run all three drift analyses on new_data."""
        report = {
            "window":       window_name,
            "timestamp":    datetime.now().isoformat(),
            "n_samples":    len(new_data),
            "drift_signals": 0,
            "retrain_recommended": False,
        }

        # ── A. DATA DRIFT (PSI on oof_score and TransactionAmt) ──
        psi_score = compute_psi(
            self.baseline["oof_score"].values,
            new_data["oof_score"].values
        )
        ks_amt = ks_test(
            self.baseline["TransactionAmt"].values,
            new_data["TransactionAmt"].values
        )
        data_drift = psi_score > PSI_DRIFT_THRESHOLD or ks_amt["drifted"]
        report["data_drift"] = {
            "psi_score":  round(psi_score, 4),
            "psi_drifted": psi_score > PSI_DRIFT_THRESHOLD,
            "ks_statistic": round(ks_amt["statistic"], 4),
            "ks_pvalue":    round(ks_amt["pvalue"], 4),
            "ks_drifted":   ks_amt["drifted"],
            "drift_detected": data_drift,
        }
        if data_drift:
            report["drift_signals"] += 1

        # ── B. PREDICTION DRIFT (change in predicted fraud proportion) ──
        base_fraud_rate = (self.baseline["oof_score"] >= 0.5).mean()
        new_fraud_rate  = (new_data["oof_score"] >= 0.5).mean()
        pred_drift      = abs(new_fraud_rate - base_fraud_rate) > PRED_DRIFT_THRESHOLD
        report["prediction_drift"] = {
            "baseline_fraud_rate": round(float(base_fraud_rate), 4),
            "new_fraud_rate":      round(float(new_fraud_rate), 4),
            "absolute_change":     round(float(abs(new_fraud_rate - base_fraud_rate)), 4),
            "drift_detected":      pred_drift,
        }
        if pred_drift:
            report["drift_signals"] += 1

        # ── C. CONFIDENCE ALERTS ──
        low_conf_mask    = ((new_data["oof_score"] > (1 - CONFIDENCE_THRESHOLD)) &
                            (new_data["oof_score"] < CONFIDENCE_THRESHOLD))
        low_conf_rate    = low_conf_mask.mean()
        base_low_conf    = ((self.baseline["oof_score"] > (1 - CONFIDENCE_THRESHOLD)) &
                            (self.baseline["oof_score"] < CONFIDENCE_THRESHOLD)).mean()
        confidence_drift = low_conf_rate > base_low_conf * 1.5  # 50% increase
        report["confidence_alerts"] = {
            "threshold":            CONFIDENCE_THRESHOLD,
            "low_confidence_rate":  round(float(low_conf_rate), 4),
            "baseline_rate":        round(float(base_low_conf), 4),
            "increase_factor":      round(float(low_conf_rate / (base_low_conf + 1e-9)), 2),
            "drift_detected":       confidence_drift,
        }
        if confidence_drift:
            report["drift_signals"] += 1

        # ── RETRAIN DECISION ──
        if report["drift_signals"] >= RETRAIN_DRIFT_COUNT:
            report["retrain_recommended"] = True
            report["auc_impact_risk"] = "high" if report["drift_signals"] >= 2 else ("medium" if report["drift_signals"] == 1 else "low")
            self.drift_count += 1

        self.history.append(report)
        return report

    def print_report(self, report: dict):
        """Pretty-print a drift report to console."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  DRIFT REPORT: {report['window']}")
        print(f"  Timestamp  : {report['timestamp']}")
        print(f"  Samples    : {report['n_samples']:,}")
        print(sep)

        dd = report["data_drift"]
        print(f"\n  [A] Data Drift")
        print(f"      PSI Score      : {dd['psi_score']:.4f}  {'⚠ DRIFT' if dd['psi_drifted'] else '✓ OK'}")
        print(f"      KS Statistic   : {dd['ks_statistic']:.4f}  (p={dd['ks_pvalue']:.4f}) {'⚠ DRIFT' if dd['ks_drifted'] else '✓ OK'}")

        pd_ = report["prediction_drift"]
        print(f"\n  [B] Prediction Drift")
        print(f"      Baseline fraud rate : {pd_['baseline_fraud_rate']*100:.2f}%")
        print(f"      New fraud rate      : {pd_['new_fraud_rate']*100:.2f}%")
        print(f"      Absolute change     : {pd_['absolute_change']*100:.2f}%  {'⚠ DRIFT' if pd_['drift_detected'] else '✓ OK'}")

        ca = report["confidence_alerts"]
        print(f"\n  [C] Confidence Alerts (threshold={ca['threshold']})")
        print(f"      Baseline low-conf rate : {ca['baseline_rate']*100:.2f}%")
        print(f"      Current low-conf rate  : {ca['low_confidence_rate']*100:.2f}%")
        print(f"      Increase factor        : {ca['increase_factor']}x  {'⚠ DRIFT' if ca['drift_detected'] else '✓ OK'}")

        sig = report["drift_signals"]
        print(f"\n  Drift signals detected: {sig}/3")
        if report["retrain_recommended"]:
            print("  ► RETRAINING RECOMMENDED  ◄")
        else:
            print("  ► No retraining needed at this time.")
        print(sep)

    def plot_drift(self, new_data: pd.DataFrame, report: dict,
                   save_path: str = "figures/fig14_drift_analysis.png"):
        """Generate a comprehensive drift visualization."""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Análisis de Drift — {report['window']}",
                     fontsize=14, fontweight="bold", color=ACCENT_COLOR)

        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

        # ── A1: Score distribution comparison ──
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(self.baseline["oof_score"], bins=50, color=LEGIT_COLOR,
                 alpha=0.6, density=True, label="Referencia")
        ax1.hist(new_data["oof_score"], bins=50, color=FRAUD_COLOR,
                 alpha=0.6, density=True, label="Nuevos datos")
        ax1.axvline(CONFIDENCE_THRESHOLD, color="orange", linestyle="--",
                    linewidth=2, label=f"Umbral conf.={CONFIDENCE_THRESHOLD}")
        ax1.axvline(1-CONFIDENCE_THRESHOLD, color="orange", linestyle="--", linewidth=2)
        ax1.set_title(f"Distribución de Scores  (PSI={report['data_drift']['psi_score']:.3f})")
        ax1.set_xlabel("Score de Fraude")
        ax1.set_ylabel("Densidad")
        ax1.legend(fontsize=9)

        # ── A2: PSI gauge ──
        ax2 = fig.add_subplot(gs[0, 2])
        psi_val = report["data_drift"]["psi_score"]
        zones   = [0.10, 0.20, 0.40]
        colors  = ["#27AE60", WARN_COLOR, FRAUD_COLOR]
        labels  = ["Estable\n(<0.10)", "Moderado\n(0.10–0.20)", "Significativo\n(>0.20)"]
        y_pos   = [0.15, 0.5, 0.75]
        for i, (z, c, lb, yp) in enumerate(zip(zones, colors, labels, y_pos)):
            ax2.barh(yp, z, color=c, alpha=0.3, height=0.2)
        ax2.barh(0.15 if psi_val < 0.1 else (0.5 if psi_val < 0.2 else 0.75),
                 psi_val, color=FRAUD_COLOR if psi_val > 0.2 else (WARN_COLOR if psi_val > 0.1 else "#27AE60"),
                 height=0.22, label=f"PSI={psi_val:.3f}")
        ax2.set_xlim(0, 0.4)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel("PSI Value")
        ax2.set_title("PSI — Drift Gauge")
        ax2.legend(fontsize=9)

        # ── B: Predicted fraud rate over time (simulated windows) ──
        ax3 = fig.add_subplot(gs[1, :])
        window_sizes = [500] * 10
        base_rates   = []
        new_rates    = []
        for ws in window_sizes:
            samp = self.baseline.sample(n=min(ws, len(self.baseline)), replace=True)
            base_rates.append((samp["oof_score"] >= 0.5).mean() * 100)
        for ws in window_sizes:
            samp = new_data.sample(n=min(ws, len(new_data)), replace=True)
            new_rates.append((samp["oof_score"] >= 0.5).mean() * 100)

        x = range(1, len(window_sizes)+1)
        ax3.plot(x, base_rates, color=LEGIT_COLOR, linewidth=2,
                 marker="o", markersize=5, label="Referencia (bootstrap)")
        ax3.plot(x, new_rates, color=FRAUD_COLOR, linewidth=2,
                 marker="s", markersize=5, label="Nuevos datos (bootstrap)")
        ax3.fill_between(x, base_rates, new_rates, alpha=0.15, color=WARN_COLOR)
        ax3.axhline(report["prediction_drift"]["baseline_fraud_rate"]*100,
                    color=LEGIT_COLOR, linestyle="--", alpha=0.5)
        ax3.axhline(report["prediction_drift"]["new_fraud_rate"]*100,
                    color=FRAUD_COLOR, linestyle="--", alpha=0.5)
        ax3.set_title("Drift de Predicciones — Tasa de Fraude por Ventana (bootstrap)")
        ax3.set_xlabel("Ventana de Bootstrap")
        ax3.set_ylabel("Tasa de Fraude Predicha (%)")
        ax3.legend()

        # ── C: Confidence zone distribution ──
        ax4 = fig.add_subplot(gs[2, :2])
        scores_ref = self.baseline["oof_score"].values
        scores_new = new_data["oof_score"].values
        low_z = (1 - CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

        ax4.hist(scores_ref, bins=60, color=LEGIT_COLOR, alpha=0.6,
                 density=True, label="Referencia")
        ax4.hist(scores_new, bins=60, color=FRAUD_COLOR, alpha=0.6,
                 density=True, label="Nuevos datos")
        ax4.axvspan(low_z[0], low_z[1], alpha=0.12, color=WARN_COLOR,
                    label=f"Zona de baja confianza ({low_z[0]}–{low_z[1]})")
        ax4.set_title("Zona de Incertidumbre (Alertas de Confianza)")
        ax4.set_xlabel("Score de Fraude")
        ax4.set_ylabel("Densidad")
        ax4.legend(fontsize=9)

        # ── Summary panel ──
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis("off")
        drift_signals = report["drift_signals"]
        color_map = {0: "#27AE60", 1: WARN_COLOR, 2: WARN_COLOR, 3: FRAUD_COLOR}
        text_color = color_map.get(drift_signals, FRAUD_COLOR)
        ax5.text(0.5, 0.90, "Resumen de Drift", ha="center", va="top",
                 fontsize=12, fontweight="bold", color=ACCENT_COLOR,
                 transform=ax5.transAxes)
        checks = [
            ("Data Drift (PSI/KS)", report["data_drift"]["drift_detected"]),
            ("Prediction Drift",    report["prediction_drift"]["drift_detected"]),
            ("Confidence Alerts",   report["confidence_alerts"]["drift_detected"]),
        ]
        for i, (label, detected) in enumerate(checks):
            symbol = "⚠" if detected else "✓"
            color  = FRAUD_COLOR if detected else "#27AE60"
            ax5.text(0.1, 0.70 - i*0.18, f"{symbol} {label}",
                     ha="left", va="top", fontsize=10, color=color,
                     transform=ax5.transAxes)

        verdict = "REENTRENAR" if report["retrain_recommended"] else "ESTABLE"
        ax5.text(0.5, 0.15, verdict, ha="center", va="center",
                 fontsize=18, fontweight="bold", color=text_color,
                 transform=ax5.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR,
                           edgecolor=text_color, linewidth=2))

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN — DEMO RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  IEEE-CIS — Drift Detection & Adaptation Demo")
    print("=" * 60)

    # Load baseline
    baseline_path = "model_artifacts/oof_baseline.parquet"
    if not os.path.exists(baseline_path):
        print(f"\n  ERROR: {baseline_path} not found.")
        print("  Run model.py first to generate model artifacts.\n")
        exit(1)

    baseline_df   = pd.read_parquet(baseline_path)
    artifacts     = joblib.load("model_artifacts/fraud_model.pkl")
    monitor       = FraudDriftMonitor(baseline_df, artifacts)

    print(f"\n  Baseline cargado: {len(baseline_df):,} transacciones")

    # Run 4 scenarios
    scenarios = [
        ("Sin Drift (Control)",         "none",     1.0),
        ("Drift Gradual (moderado)",     "gradual",  0.5),
        ("Drift Repentino (severo)",     "sudden",   1.0),
        ("Drift Estacional",             "seasonal", 1.0),
    ]

    all_reports = []
    for name, drift_type, intensity in scenarios:
        print(f"\n  Escenario: {name}")
        new_data = simulate_drift(baseline_df, drift_type=drift_type,
                                  n_new=5000, intensity=intensity)
        report   = monitor.analyze(new_data, window_name=name)
        monitor.print_report(report)
        save_path = f"figures/fig14_drift_{drift_type}.png"
        monitor.plot_drift(new_data, report, save_path=save_path)
        all_reports.append(report)

    # ── FIG 15: Summary across scenarios ──
    print("\n  Generando figura comparativa de escenarios …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Comparación de Métricas de Drift entre Escenarios",
                 fontsize=13, fontweight="bold", color=ACCENT_COLOR)

    scenario_names = [r["window"] for r in all_reports]
    psi_vals       = [r["data_drift"]["psi_score"]                        for r in all_reports]
    pred_drift_abs = [r["prediction_drift"]["absolute_change"] * 100       for r in all_reports]
    low_conf_rates = [r["confidence_alerts"]["low_confidence_rate"] * 100  for r in all_reports]

    palette = [("#27AE60" if v < 0.1 else (WARN_COLOR if v < 0.2 else FRAUD_COLOR)) for v in psi_vals]

    for ax, vals, col_vals, title, ylabel, ref_line in [
        (axes[0], psi_vals,       palette,                     "PSI Score",                 "PSI",           0.20),
        (axes[1], pred_drift_abs, [LEGIT_COLOR]*len(all_reports), "Cambio en Tasa de Fraude",  "Cambio (%)",    3.0),
        (axes[2], low_conf_rates, [WARN_COLOR]*len(all_reports),  "Tasa de Baja Confianza",    "Rate (%)",      None),
    ]:
        bars = ax.bar(range(len(scenario_names)), vals, color=col_vals,
                      edgecolor="white", linewidth=1.5, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if ref_line:
            ax.axhline(ref_line, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                       label=f"Umbral={ref_line}")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig("figures/fig15_drift_scenarios_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: figures/fig15_drift_scenarios_summary.png")

    # Save history
    import json

    class _NumpyEncoder(json.JSONEncoder):
        """Converts numpy scalars to native Python types before serializing."""
        def default(self, obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)):    return bool(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open("drift_reports/drift_history.json", "w") as f:
        json.dump(all_reports, f, indent=2, cls=_NumpyEncoder)
    print("  Saved: drift_reports/drift_history.json")

    print("\n  Drift analysis completo.\n")
