"""
eda.py  —  Seccion 3.2: Analisis y Preparacion de Datos
IEEE-CIS Fraud Detection
=================================================================
Cubre todos los requisitos de la seccion 3.2:
  1. Exploracion exhaustiva con evolucion temporal de variables
  2. Patrones, cambios en distribucion y anomalias
  3. Tecnicas con dimension temporal (ventanas deslizantes,
     variables rezagadas, agregaciones por periodo)
  4. Division temporal train/val/test sin data leakage

Figuras generadas en ./figures/
  - Prefijo EDA_  : figuras para el informe tecnico (alta calidad)
  - Prefijo INFO_ : figuras informativas del comportamiento del dataset

Ejecutar desde la carpeta raiz del proyecto.
"""

import sys
import subprocess
subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "pandas", "numpy", "matplotlib", "seaborn", "scipy", "pyarrow"],
    capture_output=True,
)

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# ESTILO GLOBAL — coherente con model.py
# ─────────────────────────────────────────────────────────────
P = {
    "primary":   "#4C3BC4",
    "secondary": "#2D8EFF",
    "accent":    "#C0392B",
    "success":   "#22C55E",
    "warning":   "#F59E0B",
    "neutral":   "#6B7280",
    "bg":        "#FAFBFF",
    "bg2":       "#F0EEFF",
    "grid":      "#E8E4F5",
    "text":      "#1A1A2E",
    "fraud":     "#C0392B",
    "legit":     "#2D8EFF",
}

plt.rcParams.update({
    "figure.facecolor":   P["bg"],
    "axes.facecolor":     P["bg"],
    "axes.grid":          True,
    "grid.color":         P["grid"],
    "grid.linewidth":     0.7,
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.edgecolor":     P["grid"],
    "xtick.color":        P["neutral"],
    "ytick.color":        P["neutral"],
    "text.color":         P["text"],
    "axes.labelcolor":    P["text"],
    "figure.dpi":         150,
})

def wm(fig, text="IEEE-CIS Fraud Detection | UTEC — Seccion 3.2"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=7, color=P["neutral"], alpha=0.5, style="italic")

def clean_axes(axes_list):
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)

def save(fig, name, tight=True):
    kw = dict(dpi=180, facecolor=P["bg"])
    if tight:
        kw["bbox_inches"] = "tight"
    fig.savefig(f"figures/{name}.png", **kw)
    plt.close(fig)
    print(f"  Saved: figures/{name}.png")

# ─────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  IEEE-CIS Fraud Detection — Analisis Exploratorio de Datos")
print("  Seccion 3.2: Analisis y Preparacion de Datos")
print("=" * 65)
print("\n[1/10] Cargando datos ...")

train_tx = pd.read_csv("train_transaction.csv")
train_id = pd.read_csv("train_identity.csv")

def normalise_id_cols(df):
    return df.rename(columns={c: c.replace("-","_") for c in df.columns if c.startswith("id-")})

train_id = normalise_id_cols(train_id)
df = pd.merge(train_tx, train_id, on="TransactionID", how="left")

# Feature temporal base
df["TransactionDay"]     = df["TransactionDT"] // 86400
df["TransactionHour"]    = (df["TransactionDT"] % 86400) // 3600
df["TransactionWeekday"] = df["TransactionDay"] % 7
df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

print(f"  Dataset combinado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f"  Tasa de fraude   : {df['isFraud'].mean()*100:.3f}%")
print(f"  Rango temporal   : dia {df['TransactionDay'].min()} a {df['TransactionDay'].max()} "
      f"({df['TransactionDay'].max()-df['TransactionDay'].min()} dias)")

# ─────────────────────────────────────────────────────────────
# EDA_01 — Desbalance de clases (figura de informe)
# ─────────────────────────────────────────────────────────────
print("\n[2/10] EDA_01: Distribucion de clases ...")

counts = df["isFraud"].value_counts()
pcts   = counts / counts.sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribucion de Clases — Desbalance del Dataset",
             fontsize=14, fontweight="bold", color=P["text"], y=1.02)

# Barras
bar_colors = [P["legit"], P["fraud"]]
bars = axes[0].bar(["Legitima", "Fraudulenta"], counts.values,
                   color=bar_colors, edgecolor="white", linewidth=2, width=0.5)
for bar, pct, cnt in zip(bars, pcts.values, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.015,
                 f"{cnt:,}\n({pct:.2f}%)",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Numero de Transacciones", fontsize=11)
axes[0].set_title("Conteo por Clase", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, counts.max() * 1.18)
clean_axes(axes[0])

# Donut
wedge_props = dict(width=0.52, edgecolor="white", linewidth=3)
wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=["Legitima","Fraudulenta"],
    colors=bar_colors, autopct="%1.2f%%",
    startangle=90, wedgeprops=wedge_props,
    textprops={"fontsize": 11},
)
for at in autotexts:
    at.set_fontsize(13); at.set_fontweight("bold")
axes[1].set_title("Proporcion de Clases", fontsize=12, fontweight="bold")

# Nota sobre imbalance
ratio = counts[0] / counts[1]
fig.text(0.5, -0.04,
         f"Ratio de desbalance: {ratio:.1f}:1  —  Se requiere scale_pos_weight o class_weight='balanced'",
         ha="center", fontsize=10, color=P["neutral"], style="italic")

wm(fig); save(fig, "EDA_01_class_imbalance")

# ─────────────────────────────────────────────────────────────
# EDA_02 — Evolucion temporal completa (figura de informe)
# ─────────────────────────────────────────────────────────────
print("\n[3/10] EDA_02: Evolucion temporal ...")

daily = df.groupby("TransactionDay").agg(
    total      = ("isFraud", "count"),
    fraud      = ("isFraud", "sum"),
    amt_mean   = ("TransactionAmt", "mean"),
    amt_median = ("TransactionAmt", "median"),
).reset_index()
daily["fraud_rate"]    = daily["fraud"] / daily["total"] * 100
daily["fraud_rate_ma"] = daily["fraud_rate"].rolling(7, center=True).mean()
daily["total_ma"]      = daily["total"].rolling(7, center=True).mean()
daily["amt_ma"]        = daily["amt_mean"].rolling(7, center=True).mean()

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Evolucion Temporal del Dataset — Analisis de Series de Tiempo",
             fontsize=15, fontweight="bold", color=P["text"], y=1.01)
gs = gridspec.GridSpec(4, 1, hspace=0.5)

axs = [fig.add_subplot(gs[i]) for i in range(4)]
for ax in axs: ax.set_facecolor(P["bg"])

# Panel 1: volumen total
axs[0].fill_between(daily["TransactionDay"], daily["total"],
                    alpha=0.25, color=P["legit"])
axs[0].plot(daily["TransactionDay"], daily["total"],
            color=P["legit"], linewidth=1, alpha=0.5)
axs[0].plot(daily["TransactionDay"], daily["total_ma"],
            color=P["primary"], linewidth=2, label="Media movil 7d")
axs[0].set_ylabel("Transacciones/dia", fontsize=10)
axs[0].set_title("Volumen Diario de Transacciones", fontsize=11, fontweight="bold")
axs[0].legend(fontsize=9); clean_axes(axs[0])

# Panel 2: fraudes diarios
axs[1].fill_between(daily["TransactionDay"], daily["fraud"],
                    alpha=0.25, color=P["fraud"])
axs[1].plot(daily["TransactionDay"], daily["fraud"],
            color=P["fraud"], linewidth=1, alpha=0.5)
axs[1].set_ylabel("Fraudes/dia", fontsize=10)
axs[1].set_title("Volumen Diario de Fraudes", fontsize=11, fontweight="bold")
clean_axes(axs[1])

# Panel 3: tasa de fraude con media movil
axs[2].fill_between(daily["TransactionDay"], daily["fraud_rate"],
                    alpha=0.15, color=P["fraud"])
axs[2].plot(daily["TransactionDay"], daily["fraud_rate"],
            color=P["fraud"], linewidth=0.8, alpha=0.4)
axs[2].plot(daily["TransactionDay"], daily["fraud_rate_ma"],
            color=P["accent"], linewidth=2.5, label="Media movil 7d")
axs[2].axhline(daily["fraud_rate"].mean(), color=P["neutral"],
               linestyle="--", linewidth=1.2, alpha=0.6,
               label=f"Media global ({daily['fraud_rate'].mean():.2f}%)")
axs[2].set_ylabel("Tasa fraude (%)", fontsize=10)
axs[2].set_title("Tasa de Fraude Diaria + Tendencia", fontsize=11, fontweight="bold")
axs[2].legend(fontsize=9); clean_axes(axs[2])

# Panel 4: monto promedio
axs[3].fill_between(daily["TransactionDay"], daily["amt_mean"],
                    alpha=0.20, color=P["warning"])
axs[3].plot(daily["TransactionDay"], daily["amt_mean"],
            color=P["warning"], linewidth=0.8, alpha=0.5)
axs[3].plot(daily["TransactionDay"], daily["amt_ma"],
            color=P["primary"], linewidth=2, label="Media movil 7d")
axs[3].set_ylabel("Monto promedio (USD)", fontsize=10)
axs[3].set_xlabel("Dia (desde referencia)", fontsize=10)
axs[3].set_title("Monto Promedio de Transaccion Diario", fontsize=11, fontweight="bold")
axs[3].legend(fontsize=9); clean_axes(axs[3])

wm(fig); save(fig, "EDA_02_temporal_evolution")

# ─────────────────────────────────────────────────────────────
# EDA_03 — Distribucion de monto por clase (figura de informe)
# ─────────────────────────────────────────────────────────────
print("\n[4/10] EDA_03: Distribucion de montos ...")

df_l = df[df["isFraud"]==0]["TransactionAmt"]
df_f = df[df["isFraud"]==1]["TransactionAmt"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribucion del Monto de Transaccion por Clase",
             fontsize=14, fontweight="bold", color=P["text"])

# Histograma lineal
axes[0,0].hist(df_l.clip(upper=2000), bins=100, color=P["legit"],
               alpha=0.65, density=True, label="Legitima",    linewidth=0)
axes[0,0].hist(df_f.clip(upper=2000), bins=100, color=P["fraud"],
               alpha=0.65, density=True, label="Fraudulenta", linewidth=0)
axes[0,0].set_xlabel("Monto USD (truncado $2000)"); axes[0,0].set_ylabel("Densidad")
axes[0,0].set_title("Distribucion Lineal"); axes[0,0].legend()

# Histograma log
axes[0,1].hist(np.log1p(df_l), bins=100, color=P["legit"],
               alpha=0.65, density=True, label="Legitima",    linewidth=0)
axes[0,1].hist(np.log1p(df_f), bins=100, color=P["fraud"],
               alpha=0.65, density=True, label="Fraudulenta", linewidth=0)
axes[0,1].set_xlabel("log(1 + Monto)"); axes[0,1].set_ylabel("Densidad")
axes[0,1].set_title("Distribucion Logaritmica"); axes[0,1].legend()

# Boxplot
bp_data  = [df_l.clip(upper=5000).values, df_f.clip(upper=5000).values]
bp_labels = ["Legitima", "Fraudulenta"]
bp = axes[1,0].boxplot(bp_data, labels=bp_labels, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=2, alpha=0.2))
for patch, color in zip(bp["boxes"], [P["legit"], P["fraud"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1,0].set_ylabel("Monto USD (truncado $5000)")
axes[1,0].set_title("Boxplot por Clase")

# Estadisticas resumen
stats_data = {
    "Estadistico":  ["Media", "Mediana", "Std Dev", "Percentil 75", "Percentil 95", "Maximo"],
    "Legitima":     [f"${df_l.mean():.2f}", f"${df_l.median():.2f}",
                     f"${df_l.std():.2f}", f"${df_l.quantile(.75):.2f}",
                     f"${df_l.quantile(.95):.2f}", f"${df_l.max():.2f}"],
    "Fraudulenta":  [f"${df_f.mean():.2f}", f"${df_f.median():.2f}",
                     f"${df_f.std():.2f}", f"${df_f.quantile(.75):.2f}",
                     f"${df_f.quantile(.95):.2f}", f"${df_f.max():.2f}"],
}
axes[1,1].axis("off")
tbl = axes[1,1].table(
    cellText=list(zip(stats_data["Estadistico"],
                      stats_data["Legitima"],
                      stats_data["Fraudulenta"])),
    colLabels=["Estadistico", "Legitima", "Fraudulenta"],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(P["bg2"] if r == 0 else P["bg"])
    cell.set_edgecolor(P["grid"])
    if r == 0: cell.set_text_props(fontweight="bold", color=P["primary"])
axes[1,1].set_title("Estadisticas Descriptivas", fontsize=11, fontweight="bold", pad=12)

for ax in axes.flatten():
    clean_axes(ax)
    ax.set_facecolor(P["bg"])

wm(fig); plt.tight_layout(); save(fig, "EDA_03_transaction_amount")

# ─────────────────────────────────────────────────────────────
# EDA_04 — Patrones temporales horarios y semanales
# ─────────────────────────────────────────────────────────────
print("\n[5/10] EDA_04: Patrones ciclicos hora/dia ...")

hourly = df.groupby("TransactionHour").agg(
    total=("isFraud","count"), fraud=("isFraud","sum")).reset_index()
hourly["fraud_rate"] = hourly["fraud"] / hourly["total"] * 100

weekly = df.groupby("TransactionWeekday").agg(
    total=("isFraud","count"), fraud=("isFraud","sum")).reset_index()
weekly["fraud_rate"] = weekly["fraud"] / weekly["total"] * 100
day_names = ["Lun","Mar","Mie","Jue","Vie","Sab","Dom"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Patrones Ciclicos de Transacciones — Hora del Dia y Dia de la Semana",
             fontsize=14, fontweight="bold", color=P["text"])

# Volumen por hora
axes[0,0].bar(hourly["TransactionHour"], hourly["total"],
              color=P["secondary"], alpha=0.75, edgecolor="white", linewidth=0.5)
axes[0,0].set_xlabel("Hora del dia"); axes[0,0].set_ylabel("Transacciones")
axes[0,0].set_title("Volumen por Hora", fontweight="bold")
axes[0,0].set_xticks(range(0,24,2))

# Tasa de fraude por hora
axes[0,1].fill_between(hourly["TransactionHour"], hourly["fraud_rate"],
                        alpha=0.25, color=P["fraud"])
axes[0,1].plot(hourly["TransactionHour"], hourly["fraud_rate"],
               color=P["fraud"], linewidth=2.5, marker="o", markersize=5)
axes[0,1].axhline(hourly["fraud_rate"].mean(), color=P["neutral"],
                  linestyle="--", linewidth=1.2, alpha=0.7, label="Media")
axes[0,1].set_xlabel("Hora del dia"); axes[0,1].set_ylabel("Tasa de fraude (%)")
axes[0,1].set_title("Tasa de Fraude por Hora", fontweight="bold")
axes[0,1].set_xticks(range(0,24,2))
# shade night hours
axes[0,1].axvspan(0, 6, alpha=0.07, color=P["primary"], label="Madrugada (0-6h)")
axes[0,1].legend(fontsize=9)

# Volumen por dia de semana
axes[1,0].bar(range(7), weekly["total"],
              color=[P["primary"] if d<5 else P["warning"] for d in range(7)],
              alpha=0.8, edgecolor="white", linewidth=0.5)
axes[1,0].set_xticks(range(7)); axes[1,0].set_xticklabels(day_names)
axes[1,0].set_ylabel("Transacciones"); axes[1,0].set_title("Volumen por Dia", fontweight="bold")

# Tasa fraude por dia
axes[1,1].bar(range(7), weekly["fraud_rate"],
              color=[P["fraud"] if r > weekly["fraud_rate"].mean() else P["legit"]
                     for r in weekly["fraud_rate"]],
              alpha=0.8, edgecolor="white", linewidth=0.5)
axes[1,1].axhline(weekly["fraud_rate"].mean(), color=P["neutral"],
                  linestyle="--", linewidth=1.2, alpha=0.7)
for i, v in enumerate(weekly["fraud_rate"]):
    axes[1,1].text(i, v+0.02, f"{v:.2f}%", ha="center", fontsize=9, fontweight="bold")
axes[1,1].set_xticks(range(7)); axes[1,1].set_xticklabels(day_names)
axes[1,1].set_ylabel("Tasa de fraude (%)"); axes[1,1].set_title("Fraude por Dia", fontweight="bold")

for ax in axes.flatten():
    clean_axes(ax); ax.set_facecolor(P["bg"])

wm(fig); plt.tight_layout(); save(fig, "EDA_04_cyclic_patterns")

# ─────────────────────────────────────────────────────────────
# EDA_05 — Valores faltantes (figura de informe)
# ─────────────────────────────────────────────────────────────
print("\n[6/10] EDA_05: Valores faltantes ...")

miss = (df.isnull().mean() * 100).sort_values(ascending=False)
miss = miss[miss > 0]

def prefix(c):
    for p in ["V","C","D","M","id_","card","addr","dist","email"]:
        if c.startswith(p): return p
    return "Other"

miss_df = pd.DataFrame({"col": miss.index, "pct": miss.values})
miss_df["group"] = miss_df["col"].apply(prefix)
group_stats = miss_df.groupby("group").agg(
    mean_pct=("pct","mean"), count=("col","count"), max_pct=("pct","max")
).sort_values("mean_pct", ascending=False).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Analisis de Valores Faltantes por Columna y Grupo",
             fontsize=14, fontweight="bold", color=P["text"])

# Top 40 columnas
top40 = miss_df.head(40)
cmap_miss = plt.cm.YlOrRd
norm_miss  = plt.Normalize(0, 100)
bar_colors_m = [cmap_miss(norm_miss(v)) for v in top40["pct"][::-1]]
axes[0].barh(range(len(top40)), top40["pct"][::-1].values,
             color=bar_colors_m, edgecolor="white", linewidth=0.3)
axes[0].set_yticks(range(len(top40)))
axes[0].set_yticklabels(top40["col"][::-1].values, fontsize=7)
axes[0].axvline(50, color=P["accent"], linestyle="--", linewidth=1.5,
                alpha=0.7, label="50%")
axes[0].axvline(75, color=P["fraud"],  linestyle=":",  linewidth=1.5,
                alpha=0.7, label="75% (umbral descarte)")
axes[0].set_xlabel("% Valores Faltantes"); axes[0].legend(fontsize=9)
axes[0].set_title("Top 40 Columnas con Mas Faltantes", fontweight="bold")

# Por grupo
bar_g = axes[1].bar(group_stats["group"], group_stats["mean_pct"],
                    color=[cmap_miss(norm_miss(v)) for v in group_stats["mean_pct"]],
                    edgecolor="white", linewidth=1.5, width=0.6)
for bar, row in zip(bar_g, group_stats.itertuples()):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"n={row.count}\n{row.mean_pct:.0f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_ylabel("% Promedio Faltantes"); axes[1].set_title("Promedio por Grupo", fontweight="bold")
axes[1].tick_params(axis="x", rotation=30)

for ax in axes: clean_axes(ax); ax.set_facecolor(P["bg"])

wm(fig); plt.tight_layout(); save(fig, "EDA_05_missing_values")

# ─────────────────────────────────────────────────────────────
# EDA_06 — D-columns (variables temporales clave)
# ─────────────────────────────────────────────────────────────
print("\n[7/10] EDA_06: Columnas D (temporales) ...")

d_cols = [f"D{i}" for i in range(1,16) if f"D{i}" in df.columns][:8]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Distribucion de Variables Temporales D1-D8: Legitima vs Fraudulenta\n"
             "(Normalizadas: TransactionDay - Di  =>  identidad de cliente)",
             fontsize=13, fontweight="bold", color=P["text"])
axes = axes.flatten()

for ax, col in zip(axes, d_cols):
    d_l = df[df["isFraud"]==0][col].dropna()
    d_f = df[df["isFraud"]==1][col].dropna()
    p99 = df[col].quantile(0.99)
    ax.hist(d_l.clip(upper=p99), bins=60, color=P["legit"],
            alpha=0.65, density=True, linewidth=0, label="Legitima")
    ax.hist(d_f.clip(upper=p99), bins=60, color=P["fraud"],
            alpha=0.65, density=True, linewidth=0, label="Fraude")
    # KS test
    ks_stat, ks_p = stats.ks_2samp(d_l.dropna(), d_f.dropna())
    ax.set_title(f"{col}  (KS={ks_stat:.3f}, p={ks_p:.2e})",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Dias"); ax.legend(fontsize=7)
    clean_axes(ax); ax.set_facecolor(P["bg"])

wm(fig); plt.tight_layout(); save(fig, "EDA_06_d_columns")

# ─────────────────────────────────────────────────────────────
# EDA_07 — Variables categoricas vs fraude
# ─────────────────────────────────────────────────────────────
print("\n[8/10] EDA_07: Variables categoricas ...")

cat_features = [
    ("ProductCD",        "Producto"),
    ("card4",            "Red de tarjeta"),
    ("card6",            "Tipo de tarjeta"),
    ("P_emaildomain",    "Email comprador (top 12)"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Tasa de Fraude por Variable Categorica",
             fontsize=14, fontweight="bold", color=P["text"])
axes = axes.flatten()

for ax, (col, title) in zip(axes, cat_features):
    if col not in df.columns:
        ax.set_visible(False); continue
    top_vals  = df[col].value_counts().head(12).index
    sub       = df[df[col].isin(top_vals)]
    fr        = sub.groupby(col)["isFraud"].mean().sort_values(ascending=False) * 100
    counts_c  = sub[col].value_counts()
    base_rate = df["isFraud"].mean() * 100
    bar_cols  = [P["fraud"] if v > base_rate else P["legit"] for v in fr.values]
    bars = ax.bar(range(len(fr)), fr.values, color=bar_cols,
                  edgecolor="white", linewidth=1, width=0.7)
    for i, (idx, v) in enumerate(fr.items()):
        ax.text(i, v + 0.08,
                f"n={counts_c.get(idx,0):,}",
                ha="center", va="bottom", fontsize=7, rotation=60)
    ax.axhline(base_rate, color=P["neutral"], linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Media ({base_rate:.2f}%)")
    ax.set_xticks(range(len(fr)))
    ax.set_xticklabels(fr.index.astype(str), rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Tasa de Fraude (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    clean_axes(ax); ax.set_facecolor(P["bg"])

wm(fig); plt.tight_layout(); save(fig, "EDA_07_categorical_features")

# ─────────────────────────────────────────────────────────────
# EDA_08 — Division temporal sin leakage (figura de informe)
# ─────────────────────────────────────────────────────────────
print("\n[9/10] EDA_08: Division temporal + ventanas deslizantes ...")

# Definir splits respetando orden temporal estricto (Req. Seccion 3.2)
# Porcentajes coherentes con model.py para garantizar reproducibilidad
TRAIN_PCT = 0.70   # 70% train
VAL_PCT   = 0.85   # 15% val (del 70 al 85%)
# TEST_PCT = 0.15  # 15% test (del 85% al 100%)

day_min = df["TransactionDay"].min()
day_max = df["TransactionDay"].max()
day_range = day_max - day_min

cut_train = day_min + int(day_range * TRAIN_PCT)
cut_val   = day_min + int(day_range * VAL_PCT)

daily["split"] = np.where(daily["TransactionDay"] <= cut_train, "Train",
                 np.where(daily["TransactionDay"] <= cut_val,   "Validacion",
                                                                 "Test"))
split_colors = {"Train": P["primary"], "Validacion": P["warning"], "Test": P["accent"]}

# Ventana deslizante de 14 dias para tasa de fraude (Req. Seccion 3.2)
# Variables rezagadas y agregaciones por periodo
window_days = 14
daily["fraud_rate_roll14"] = daily["fraud_rate"].rolling(window_days, center=True).mean()
daily["total_roll14"]      = daily["total"].rolling(window_days, center=True).mean()
daily["fraud_rate_lag7"]   = daily["fraud_rate"].shift(7)   # variable rezagada 7 dias
daily["amt_roll14"]        = daily["amt_mean"].rolling(window_days, center=True).mean()

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Division Temporal del Dataset y Analisis de Ventanas Deslizantes\n"
             "Particion respeta el orden cronologico — sin data leakage",
             fontsize=14, fontweight="bold", color=P["text"], y=1.02)
gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.35)

# Panel 1: volumen con splits
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(P["bg"])
for split, color in split_colors.items():
    mask = daily["split"] == split
    ax1.fill_between(daily["TransactionDay"][mask], daily["total"][mask],
                     alpha=0.30, color=color, label=split)
    ax1.plot(daily["TransactionDay"][mask], daily["total_roll14"][mask],
             color=color, linewidth=2)
ax1.axvline(cut_train, color=P["neutral"], linestyle="--", linewidth=1.5, alpha=0.8)
ax1.axvline(cut_val,   color=P["neutral"], linestyle="--", linewidth=1.5, alpha=0.8)
ax1.text(cut_train+0.5, ax1.get_ylim()[1]*0.9, "Corte\nTrain/Val",
         fontsize=8, color=P["neutral"])
ax1.text(cut_val+0.5,   ax1.get_ylim()[1]*0.9, "Corte\nVal/Test",
         fontsize=8, color=P["neutral"])
ax1.set_ylabel("Transacciones/dia"); ax1.set_xlabel("Dia")
ax1.set_title("Volumen Diario por Particion Temporal (linea=media movil 14d)",
              fontweight="bold")
ax1.legend(fontsize=10, loc="upper left"); clean_axes(ax1)

# Panel 2: tasa de fraude con splits
ax2 = fig.add_subplot(gs[1, :])
ax2.set_facecolor(P["bg"])
for split, color in split_colors.items():
    mask = daily["split"] == split
    ax2.fill_between(daily["TransactionDay"][mask], daily["fraud_rate"][mask],
                     alpha=0.20, color=color)
    ax2.plot(daily["TransactionDay"][mask], daily["fraud_rate_roll14"][mask],
             color=color, linewidth=2.5, label=f"{split}")
ax2.axvline(cut_train, color=P["neutral"], linestyle="--", linewidth=1.5, alpha=0.8)
ax2.axvline(cut_val,   color=P["neutral"], linestyle="--", linewidth=1.5, alpha=0.8)
ax2.set_ylabel("Tasa de fraude (%)"); ax2.set_xlabel("Dia")
ax2.set_title("Tasa de Fraude por Particion — Ventana Deslizante 14 dias",
              fontweight="bold")
ax2.legend(fontsize=10); clean_axes(ax2)

# Panel 3 & 4: estadisticas por split
split_stats = df.copy()
split_stats["split"] = np.where(split_stats["TransactionDay"] <= cut_train, "Train",
                       np.where(split_stats["TransactionDay"] <= cut_val,   "Validacion",
                                                                              "Test"))
stats_by_split = split_stats.groupby("split").agg(
    n           = ("isFraud", "count"),
    n_fraud     = ("isFraud", "sum"),
    fraud_rate  = ("isFraud", "mean"),
    amt_mean    = ("TransactionAmt", "mean"),
    amt_median  = ("TransactionAmt", "median"),
).reset_index()
stats_by_split["fraud_rate"] = stats_by_split["fraud_rate"] * 100

ax3 = fig.add_subplot(gs[2, 0])
ax3.set_facecolor(P["bg"])
order = ["Train", "Validacion", "Test"]
stats_by_split = stats_by_split.set_index("split").reindex(order).reset_index()
bar_s = ax3.bar(stats_by_split["split"], stats_by_split["fraud_rate"],
                color=[split_colors[s] for s in stats_by_split["split"]],
                edgecolor="white", linewidth=1.5, width=0.5)
for bar, row in zip(bar_s, stats_by_split.itertuples()):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f"{row.fraud_rate:.2f}%\nn={row.n:,}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax3.set_ylabel("Tasa de Fraude (%)"); ax3.set_title("Tasa de Fraude por Split", fontweight="bold")
clean_axes(ax3)

# Tabla resumen de splits
ax4 = fig.add_subplot(gs[2, 1])
ax4.axis("off")
table_data = []
for _, row in stats_by_split.iterrows():
    table_data.append([row["split"], f"{row['n']:,}", f"{row['n_fraud']:,}",
                       f"{row['fraud_rate']:.2f}%", f"${row['amt_mean']:.1f}"])
tbl2 = ax4.table(
    cellText=table_data,
    colLabels=["Split", "Transacciones", "Fraudes", "Tasa Fraude", "Monto Prom."],
    loc="center", cellLoc="center",
)
tbl2.auto_set_font_size(False); tbl2.set_fontsize(10)
tbl2.scale(1, 2.0)
for (r, c), cell in tbl2.get_celld().items():
    if r == 0:
        cell.set_facecolor(P["bg2"])
        cell.set_text_props(fontweight="bold", color=P["primary"])
    else:
        split_name = table_data[r-1][0]
        cell.set_facecolor(P["bg"])
    cell.set_edgecolor(P["grid"])
ax4.set_title("Estadisticas por Particion Temporal", fontsize=11, fontweight="bold", pad=12)

# Guardar info del split para uso en model.py
split_info = {
    "cut_train": int(cut_train),
    "cut_val":   int(cut_val),
    "day_min":   int(day_min),
    "day_max":   int(day_max),
}
pd.Series(split_info).to_json("model_artifacts/temporal_split.json"
                               if os.path.exists("model_artifacts")
                               else "temporal_split.json")

wm(fig); save(fig, "EDA_08_temporal_split")
print(f"    Cortes: Train <= dia {cut_train} | Val <= dia {cut_val} | Test > dia {cut_val}")

# ─────────────────────────────────────────────────────────────
# INFO_01 — Heatmap de correlacion V-features (informativo)
# ─────────────────────────────────────────────────────────────
print("\n[10/10] INFO_01: Correlacion features V + INFO_02: Anomalias ...")

v_valid = [c for c in df.columns
           if c.startswith("V")
           and df[c].isnull().mean() < 0.30
           and df[c].nunique() > 5][:20]
corr_df = df[v_valid + ["isFraud"]].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.zeros_like(corr_df, dtype=bool)
mask[np.triu_indices_from(mask)] = True
cmap_corr = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(corr_df, mask=mask, cmap=cmap_corr, center=0,
            vmin=-1, vmax=1, linewidths=0.4, linecolor="white",
            ax=ax, cbar_kws={"shrink": 0.75, "label": "Correlacion de Pearson"},
            annot=False)
ax.set_title("Correlacion entre Features V (seleccionadas, <30% NaN) e isFraud\n"
             "Figura Informativa — Estructura de dependencia lineal",
             fontsize=13, fontweight="bold", color=P["text"], pad=12)
ax.set_facecolor(P["bg"])
wm(fig); plt.tight_layout(); save(fig, "INFO_01_correlation_heatmap")

# INFO_02 — Deteccion de anomalias por Z-score en TransactionAmt
amt_log   = df["TransactionAmt_log"]
z_scores  = np.abs(stats.zscore(amt_log.dropna()))
anomalies = df.loc[amt_log.dropna().index[z_scores > 3], "TransactionAmt"]
pct_anom  = len(anomalies) / len(df) * 100

daily_anom = df.copy()
daily_anom["is_anomaly"] = False
daily_anom.loc[amt_log.dropna().index[z_scores > 3], "is_anomaly"] = True
anom_daily = daily_anom.groupby("TransactionDay").agg(
    anomaly_rate = ("is_anomaly", "mean"),
    fraud_rate   = ("isFraud",    "mean"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Deteccion de Anomalias en Monto (Z-score > 3) — {pct_anom:.2f}% del dataset\n"
             "Figura Informativa",
             fontsize=13, fontweight="bold", color=P["text"])

axes[0].scatter(df["TransactionAmt_log"],
                np.random.uniform(-0.3, 0.3, len(df)),
                c=np.where(
                np.abs(stats.zscore(df["TransactionAmt_log"].fillna(df["TransactionAmt_log"].median()))) > 3,
                P["fraud"], P["legit"]),
                alpha=0.15, s=3)
axes[0].axvline(amt_log.mean() + 3*amt_log.std(), color=P["accent"],
                linestyle="--", linewidth=2, label="Umbral Z=3")
axes[0].set_xlabel("log(1 + TransactionAmt)")
axes[0].set_title("Distribucion con Anomalias Marcadas\n(rojo=anomalia, azul=normal)", fontweight="bold")
axes[0].legend(); axes[0].set_yticks([])

axes[1].plot(anom_daily["TransactionDay"], anom_daily["anomaly_rate"]*100,
             color=P["warning"], linewidth=2, label="Tasa anomalias Z>3 (%)")
axes[1].plot(anom_daily["TransactionDay"], anom_daily["fraud_rate"]*100,
             color=P["fraud"],   linewidth=2, label="Tasa fraude (%)")
axes[1].set_xlabel("Dia"); axes[1].set_ylabel("%")
axes[1].set_title("Evolucion Temporal: Anomalias vs Fraude", fontweight="bold")
axes[1].legend(fontsize=9)

for ax in axes: clean_axes(ax); ax.set_facecolor(P["bg"])

wm(fig); plt.tight_layout(); save(fig, "INFO_02_anomaly_detection")

# ─────────────────────────────────────────────────────────────
# INFO_03 — Degradacion temporal del dataset: estabilidad de
#            la tasa de fraude por periodo (Req. Seccion 3.2)
# ─────────────────────────────────────────────────────────────
print("    INFO_03: Estabilidad temporal del dataset ...")

# Calcular AUC de un modelo simple (score=TransactionAmt_log) por ventana
# para mostrar como cambia la separabilidad de clases a lo largo del tiempo
from scipy.stats import ks_2samp

window_days = 14
day_centers, ks_stats, fraud_rates_w, n_windows = [], [], [], []
d = int(df["TransactionDay"].min())
while d + window_days <= int(df["TransactionDay"].max()):
    mask_w = (df["TransactionDay"] >= d) & (df["TransactionDay"] < d + window_days)
    sub_w  = df[mask_w]
    if len(sub_w) > 100 and sub_w["isFraud"].sum() > 5:
        legit_amt = sub_w.loc[sub_w["isFraud"]==0, "TransactionAmt_log"].dropna()
        fraud_amt = sub_w.loc[sub_w["isFraud"]==1, "TransactionAmt_log"].dropna()
        if len(legit_amt) > 5 and len(fraud_amt) > 5:
            ks_stat, _ = ks_2samp(legit_amt, fraud_amt)
            day_centers.append(d + window_days // 2)
            ks_stats.append(ks_stat)
            fraud_rates_w.append(sub_w["isFraud"].mean() * 100)
            n_windows.append(len(sub_w))
    d += window_days // 2

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
title_info3 = ("Estabilidad Temporal del Dataset"
               f" — Ventanas Deslizantes de {window_days} dias | Req. Seccion 3.2")
fig.suptitle(title_info3, fontsize=13, fontweight="bold", color=P["text"])

# Panel 1: KS statistic (separabilidad clases)
axes[0].fill_between(day_centers, ks_stats, alpha=0.2, color=P["primary"])
axes[0].plot(day_centers, ks_stats, color=P["primary"], linewidth=2.5,
             marker="o", markersize=4, label="KS Statistic (separabilidad)")
axes[0].axhline(np.mean(ks_stats), color=P["neutral"], linestyle="--",
                linewidth=1.5, alpha=0.6, label=f"Media ({np.mean(ks_stats):.3f})")
axes[0].set_ylabel("KS Statistic (Legitima vs Fraude)", fontsize=10)
axes[0].set_title("Separabilidad de Clases (monto) por Ventana Temporal",
                  fontweight="bold")
axes[0].legend(fontsize=9); clean_axes(axes[0])

# Panel 2: tasa de fraude por ventana (concept drift indicator)
axes[1].fill_between(day_centers, fraud_rates_w, alpha=0.25, color=P["fraud"])
axes[1].plot(day_centers, fraud_rates_w, color=P["fraud"], linewidth=2.5,
             marker="s", markersize=4)
axes[1].axhline(np.mean(fraud_rates_w), color=P["neutral"], linestyle="--",
                linewidth=1.5, alpha=0.6)
# Agregar banda de 1 std para detectar anomalias
mean_fr = np.mean(fraud_rates_w); std_fr = np.std(fraud_rates_w)
axes[1].fill_between(day_centers,
                     [mean_fr - std_fr]*len(day_centers),
                     [mean_fr + std_fr]*len(day_centers),
                     alpha=0.10, color=P["fraud"], label="+-1 std")
axes[1].set_ylabel("Tasa de Fraude (%)", fontsize=10)
axes[1].set_title("Tasa de Fraude por Ventana — Deteccion de Concept Drift",
                  fontweight="bold")
axes[1].legend(fontsize=9); clean_axes(axes[1])

# Panel 3: variable rezagada (lag 7d) vs valor actual
lag7  = pd.Series(fraud_rates_w).shift(7).bfill().values
axes[2].plot(day_centers, fraud_rates_w, color=P["fraud"], linewidth=2,
             label="Tasa fraude (actual)")
axes[2].plot(day_centers, lag7, color=P["warning"], linewidth=2,
             linestyle="--", label="Tasa fraude (lag 7 ventanas)")
axes[2].fill_between(day_centers,
                     np.array(fraud_rates_w), lag7,
                     where=np.array(fraud_rates_w) > lag7,
                     alpha=0.15, color=P["fraud"], label="Incremento vs lag")
axes[2].set_ylabel("Tasa de Fraude (%)", fontsize=10)
axes[2].set_xlabel("Dia (desde referencia)", fontsize=10)
axes[2].set_title("Variable Rezagada (Lag) — Comparacion con Periodo Anterior",
                  fontweight="bold")
axes[2].legend(fontsize=9); clean_axes(axes[2])

for ax in axes: ax.set_facecolor(P["bg"])
wm(fig); plt.tight_layout(); save(fig, "INFO_03_temporal_degradation_dataset")

# ─────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN — Analisis Exploratorio Completado")
print("=" * 65)
print(f"\n  Dataset: {len(df):,} transacciones x {df.shape[1]} columnas")
print(f"  Tasa de fraude     : {df['isFraud'].mean()*100:.3f}%")
print(f"  Ratio de desbalance: {(y==0).sum()/(y==1).sum() if 'y' in dir() else int(counts[0]/counts[1]):.0f}:1")
print(f"  Rango temporal     : {day_range} dias")
print(f"  Valores faltantes  : {(df.isnull().mean()>0).sum()} columnas afectadas")
print(f"\n  FIGURAS PARA INFORME (prefijo EDA_):")
for name in ["EDA_01_class_imbalance","EDA_02_temporal_evolution",
             "EDA_03_transaction_amount","EDA_04_cyclic_patterns",
             "EDA_05_missing_values","EDA_06_d_columns",
             "EDA_07_categorical_features","EDA_08_temporal_split"]:
    print(f"    figures/{name}.png")
print(f"\n  FIGURAS INFORMATIVAS (prefijo INFO_):")
for name in ["INFO_01_correlation_heatmap","INFO_02_anomaly_detection"]:
    print(f"    figures/{name}.png")
print("=" * 65)
print("  EDA completo.\n")