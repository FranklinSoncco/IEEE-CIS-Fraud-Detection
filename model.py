"""
model.py  —  Seccion 3.3: Modelado y Evaluacion (v3 — PDF compliant)
IEEE-CIS Fraud Detection
=================================================================
Cumplimiento completo del PDF de requisitos:

Seccion 3.2 — Division temporal sin leakage:
  - Train : primeros 70% de dias cronologicos
  - Val   : dias 70%-85%
  - Test  : ultimos 15% de dias
  - CV    : TimeSeriesSplit (5 folds) sobre el conjunto TRAIN
            respetando orden cronologico en cada fold

Seccion 3.3 — Modelado y evaluacion:
  - Dos enfoques tradicionales: Logistic Regression, Random Forest
  - Enfoques avanzados: LightGBM, XGBoost, CatBoost (ensemble)
  - Evaluacion con AUC, F1, Balanced Accuracy
  - Analisis de degradacion temporal del modelo (AUC por ventana de tiempo)
  - Interpretacion de resultados en entorno dinamico

Otras mejoras:
  - Frequency encoding, interacciones, D-norm, UID proxy
  - Postprocessing por UID (Chris Deotte, 1st place)
  - Figuras de calidad de publicacion (180 DPI)

Compatibilidad: Python 3.9-3.12, XGBoost 1.x/2.x, LightGBM 3.x/4.x
"""

# ── Dependencias ──────────────────────────────────────────────────────────────
import sys
import subprocess
subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "xgboost", "lightgbm", "catboost", "scikit-learn",
     "pandas", "numpy", "matplotlib", "seaborn", "joblib", "pyarrow"],
    capture_output=True,
)

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics         import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    roc_curve, confusion_matrix, precision_recall_curve,
)
import xgboost  as xgb
import lightgbm as lgb
from catboost  import CatBoostClassifier

warnings.filterwarnings("ignore")
os.makedirs("model_artifacts", exist_ok=True)
os.makedirs("figures",         exist_ok=True)

# ── Constantes ────────────────────────────────────────────────────────────────
SEED    = 42
N_FOLDS = 5
W_LGB   = 0.40
W_XGB   = 0.30
W_CAT   = 0.30

# Particion temporal (porcentajes del rango de dias)
TRAIN_PCT = 0.70
VAL_PCT   = 0.85   # del 70% al 85%
# TEST_PCT  = resto (85% al 100%)

# ── Estilo publicacion ────────────────────────────────────────────────────────
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

def wm(fig, text="IEEE-CIS Fraud Detection | UTEC — Seccion 3.3"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=7, color=P["neutral"], alpha=0.5, style="italic")

def clean_ax(ax):
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

def save_fig(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=180, bbox_inches="tight",
                facecolor=P["bg"])
    plt.close(fig)
    print(f"  Saved: figures/{name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def normalise_id_cols(df):
    rename = {c: c.replace("-", "_") for c in df.columns if c.startswith("id-")}
    if rename:
        print(f"    Renombrando {len(rename)} columnas id-XX -> id_XX")
    return df.rename(columns=rename)


def get_xgb_best_iteration(booster):
    for attr in ("best_iteration", "best_ntree_limit"):
        val = getattr(booster, attr, None)
        if val is not None:
            return max(1, int(val))
    attrs = booster.attributes()
    if "best_iteration" in attrs:
        return max(1, int(attrs["best_iteration"]) + 1)
    return 500


def get_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC":               round(roc_auc_score(y_true, y_prob), 5),
        "F1":                round(f1_score(y_true, y_pred), 5),
        "Balanced Accuracy": round(balanced_accuracy_score(y_true, y_pred), 5),
    }


def frequency_encode(train_df, test_df, col):
    freq = train_df[col].value_counts(normalize=True)
    train_df[f"{col}_freq"] = train_df[col].map(freq).fillna(0)
    test_df[f"{col}_freq"]  = test_df[col].map(freq).fillna(0)
    return train_df, test_df


def temporal_degradation(y_true, y_prob, days, window_days=14):
    """
    Calcula AUC en ventanas temporales deslizantes para analizar
    degradacion del modelo a lo largo del tiempo (req. seccion 3.3).
    """
    day_min = days.min()
    day_max = days.max()
    records = []
    d = day_min
    while d + window_days <= day_max:
        mask = (days >= d) & (days < d + window_days)
        if mask.sum() > 50 and y_true[mask].sum() > 5:
            auc = roc_auc_score(y_true[mask], y_prob[mask])
            records.append({
                "day_start":  d,
                "day_center": d + window_days // 2,
                "n":          int(mask.sum()),
                "n_fraud":    int(y_true[mask].sum()),
                "fraud_rate": float(y_true[mask].mean() * 100),
                "auc":        round(auc, 5),
            })
        d += window_days // 2   # overlap 50%
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  IEEE-CIS Fraud Detection — Modelado v3 (PDF compliant)")
print("=" * 65)
print("\n[1/10] Cargando datos ...")

train_tx = pd.read_csv("train_transaction.csv")
train_id = pd.read_csv("train_identity.csv")
test_tx  = pd.read_csv("test_transaction.csv")
test_id  = pd.read_csv("test_identity.csv")

train_id = normalise_id_cols(train_id)
test_id  = normalise_id_cols(test_id)

train = pd.merge(train_tx, train_id, on="TransactionID", how="left")
test  = pd.merge(test_tx,  test_id,  on="TransactionID", how="left")
print(f"  Train merged: {train.shape}  |  Test merged: {test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/10] Feature engineering ...")

def feature_engineering(df):
    df = df.copy()
    df["TransactionDay"]     = df["TransactionDT"] // 86400
    df["TransactionHour"]    = (df["TransactionDT"] % 86400) // 3600
    df["TransactionWeekday"] = df["TransactionDay"] % 7
    df["HourBin"] = pd.cut(df["TransactionHour"],
                            bins=[-1, 6, 12, 18, 23],
                            labels=[0, 1, 2, 3]).astype(float)
    for i in range(1, 16):
        col = f"D{i}"
        if col in df.columns:
            df[f"{col}_norm"] = df["TransactionDay"] - df[col]
    df["TransactionAmt_log"]     = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = df["TransactionAmt"] - df["TransactionAmt"].astype(int)
    df["AmtIsRound"]             = (df["TransactionAmt_decimal"] == 0).astype(int)
    if "D1_norm" in df.columns and "addr1" in df.columns:
        df["uid"] = (df["card1"].astype(str) + "_"
                     + df["addr1"].fillna(-1).astype(int).astype(str) + "_"
                     + df["D1_norm"].round(0).astype(str))
    else:
        df["uid"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
    uid_agg = (df.groupby("uid")["TransactionAmt"]
               .agg(uid_amt_mean="mean", uid_amt_std="std",
                    uid_amt_count="count", uid_amt_sum="sum")
               .reset_index())
    df = df.merge(uid_agg, on="uid", how="left")
    df["uid_amt_zscore"] = ((df["TransactionAmt"] - df["uid_amt_mean"])
                            / (df["uid_amt_std"] + 1e-9))
    for group_col in ["card1", "card2", "addr1"]:
        if group_col in df.columns:
            agg = (df.groupby(group_col)["TransactionAmt"]
                   .agg(**{f"{group_col}_amt_mean": "mean",
                           f"{group_col}_amt_std":  "std",
                           f"{group_col}_count":    "count"})
                   .reset_index())
            df = df.merge(agg, on=group_col, how="left")
    if "addr1" in df.columns:
        df["card1_addr1"] = (df["card1"].astype(str) + "_"
                             + df["addr1"].fillna(-1).astype(int).astype(str))
    if "P_emaildomain" in df.columns:
        df["card1_email"] = (df["card1"].astype(str) + "_"
                             + df["P_emaildomain"].fillna("NA").astype(str))
    m_cols = [c for c in df.columns if c.startswith("M")]
    if m_cols:
        df["M_sum"]   = (df[m_cols] == "T").sum(axis=1)
        df["M_total"] = df[m_cols].notna().sum(axis=1)
    return df

train = feature_engineering(train)
test  = feature_engineering(test)
print(f"  Columnas — train: {train.shape[1]}, test: {test.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DIVISION TEMPORAL ESTRICTA (Req. Seccion 3.2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/10] Division temporal estricta (sin data leakage) ...")

day_min   = train["TransactionDay"].min()
day_max   = train["TransactionDay"].max()
day_range = day_max - day_min

cut_train = day_min + int(day_range * TRAIN_PCT)
cut_val   = day_min + int(day_range * VAL_PCT)

# Mascaras temporales
mask_train = train["TransactionDay"] <= cut_train
mask_val   = (train["TransactionDay"] > cut_train) & (train["TransactionDay"] <= cut_val)
mask_test  = train["TransactionDay"] > cut_val

n_train = mask_train.sum()
n_val   = mask_val.sum()
n_test  = mask_test.sum()
n_total = len(train)

print(f"  Rango dias: {day_min} a {day_max} ({day_range} dias)")
print(f"  Corte Train : dia <= {cut_train}  ({n_train:,} muestras, {n_train/n_total*100:.1f}%)")
print(f"  Corte Val   : dia <= {cut_val}    ({n_val:,} muestras, {n_val/n_total*100:.1f}%)")
print(f"  Corte Test  : dia >  {cut_val}    ({n_test:,} muestras, {n_test/n_total*100:.1f}%)")
print(f"  Fraud rate — Train: {train[mask_train]['isFraud'].mean()*100:.2f}%  "
      f"Val: {train[mask_val]['isFraud'].mean()*100:.2f}%  "
      f"Test: {train[mask_test]['isFraud'].mean()*100:.2f}%")

# Guardar info de cortes
import json
os.makedirs("model_artifacts", exist_ok=True)
with open("model_artifacts/temporal_split.json", "w") as f:
    json.dump({"day_min": int(day_min), "day_max": int(day_max),
               "cut_train": int(cut_train), "cut_val": int(cut_val),
               "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test)}, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPARACION DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/10] Preparando features ...")

TARGET    = "isFraud"
DROP_COLS = {TARGET, "TransactionID", "uid", "TransactionDT"}

# Alinear columnas train / test
for col in set(train.columns) - DROP_COLS - set(test.columns):  test[col]  = np.nan
for col in set(test.columns)  - DROP_COLS - set(train.columns): train[col] = np.nan

# Frequency encoding (calculado SOLO sobre split train para evitar leakage)
freq_cols = ["card1", "card2", "addr1", "P_emaildomain",
             "R_emaildomain", "card1_addr1", "card1_email"]
freq_cols = [c for c in freq_cols if c in train.columns]
for col in freq_cols:
    # Frecuencia calculada solo en train cronologico
    freq = train.loc[mask_train, col].value_counts(normalize=True)
    train[f"{col}_freq"] = train[col].map(freq).fillna(0)
    test[f"{col}_freq"]  = test[col].map(freq).fillna(0)
print(f"  Frequency-encoded (basado en train): {freq_cols}")

# Label encoding (vocabulario train+test)
cat_cols = [c for c in train.select_dtypes(include=["object"]).columns
            if c not in DROP_COLS]
le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    tv = train[col].fillna("missing").astype(str)
    ev = test[col].fillna("missing").astype(str)
    le.fit(pd.concat([tv, ev], axis=0))
    train[col] = le.transform(tv)
    test[col]  = le.transform(ev)
    le_map[col] = le
print(f"  Label-encoded: {len(cat_cols)} columnas")

# Seleccion de features
candidate_cols = [c for c in train.columns
                  if c not in DROP_COLS and c in test.columns]
v_sparse = {c for c in candidate_cols
            if c.startswith("V") and train[c].isnull().mean() > 0.75}
feature_cols = [c for c in candidate_cols if c not in v_sparse]

print(f"  V-columns descartadas (>75% NaN): {len(v_sparse)}")
print(f"  Features finales: {len(feature_cols)}")

# Crear splits con orden temporal preservado
X_tr  = train.loc[mask_train, feature_cols].fillna(-999)
y_tr  = train.loc[mask_train, TARGET].astype(int)
X_val = train.loc[mask_val,   feature_cols].fillna(-999)
y_val = train.loc[mask_val,   TARGET].astype(int)
X_te  = train.loc[mask_test,  feature_cols].fillna(-999)
y_te  = train.loc[mask_test,  TARGET].astype(int)

# dias para analisis de degradacion
days_tr  = train.loc[mask_train, "TransactionDay"].values
days_val = train.loc[mask_val,   "TransactionDay"].values
days_te  = train.loc[mask_test,  "TransactionDay"].values

X_kaggle_test = test[feature_cols].fillna(-999)
test_uids     = test["uid"].values

assert list(X_tr.columns) == list(X_kaggle_test.columns), "ERROR: columnas desalineadas"
print(f"  X_tr: {X_tr.shape} | X_val: {X_val.shape} | X_te: {X_te.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION TEMPORAL (TimeSeriesSplit sobre X_tr)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/10] Cross-validation temporal (TimeSeriesSplit, sin shuffle) ...")

# TimeSeriesSplit garantiza que el fold de validacion es siempre
# POSTERIOR al fold de entrenamiento — sin data leakage
tss   = TimeSeriesSplit(n_splits=N_FOLDS)
w_pos = float((y_tr == 0).sum() / (y_tr == 1).sum())

lgb_params = {
    "objective":         "binary",
    "metric":            "auc",
    "learning_rate":     0.03,
    "num_leaves":        128,
    "max_depth":         8,
    "min_child_samples": 80,
    "feature_fraction":  0.70,
    "bagging_fraction":  0.70,
    "bagging_freq":      1,
    "reg_alpha":         0.3,
    "reg_lambda":        1.5,
    "min_split_gain":    0.01,
    "scale_pos_weight":  w_pos,
    "verbose":           -1,
    "seed":              SEED,
}
xgb_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "learning_rate":    0.03,
    "max_depth":        5,
    "min_child_weight": 20,
    "gamma":            2,
    "subsample":        0.70,
    "colsample_bytree": 0.70,
    "reg_alpha":        0.3,
    "reg_lambda":       2.0,
    "scale_pos_weight": w_pos,
    "tree_method":      "hist",
    "seed":             SEED,
}

oof_lgb = np.zeros(len(y_tr))
oof_xgb = np.zeros(len(y_tr))
oof_cat = np.zeros(len(y_tr))

scores_lgb, scores_xgb, scores_cat = [], [], []
models_lgb, models_xgb = [], []
fold_auc_records = []

for fold, (tr_idx, va_idx) in enumerate(tss.split(X_tr), 1):
    print(f"\n  === Fold {fold}/{N_FOLDS} (temporal) ===")
    Xf_tr, Xf_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    yf_tr, yf_va = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
    day_fold_tr_end   = days_tr[tr_idx[-1]]
    day_fold_va_start = days_tr[va_idx[0]]
    print(f"    Train dias: {days_tr[tr_idx[0]]}..{day_fold_tr_end}  "
          f"({len(tr_idx):,} muestras)")
    print(f"    Val   dias: {day_fold_va_start}..{days_tr[va_idx[-1]]}  "
          f"({len(va_idx):,} muestras)")

    # LightGBM
    dl = lgb.Dataset(Xf_tr, label=yf_tr)
    dv = lgb.Dataset(Xf_va, label=yf_va, reference=dl)
    m_lgb = lgb.train(lgb_params, dl, num_boost_round=5000,
                      valid_sets=[dv],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(-1)])
    p_lgb        = m_lgb.predict(Xf_va)
    oof_lgb[va_idx] = p_lgb
    auc_l = roc_auc_score(yf_va, p_lgb)
    scores_lgb.append(auc_l)
    models_lgb.append(m_lgb)
    print(f"    LightGBM  AUC={auc_l:.5f}  iter={m_lgb.best_iteration}")

    # XGBoost
    dx_tr = xgb.DMatrix(Xf_tr, label=yf_tr)
    dx_va = xgb.DMatrix(Xf_va, label=yf_va)
    m_xgb = xgb.train(xgb_params, dx_tr, num_boost_round=5000,
                      evals=[(dx_va, "val")],
                      early_stopping_rounds=150,
                      verbose_eval=False)
    p_xgb        = m_xgb.predict(xgb.DMatrix(Xf_va))
    oof_xgb[va_idx] = p_xgb
    auc_x = roc_auc_score(yf_va, p_xgb)
    scores_xgb.append(auc_x)
    models_xgb.append(m_xgb)
    print(f"    XGBoost   AUC={auc_x:.5f}  iter={get_xgb_best_iteration(m_xgb)}")

    # CatBoost
    m_cat = CatBoostClassifier(
        iterations=5000, learning_rate=0.03, depth=6,
        l2_leaf_reg=5, min_data_in_leaf=50,
        subsample=0.70, colsample_bylevel=0.70,
        scale_pos_weight=w_pos, eval_metric="AUC",
        od_type="Iter", od_wait=150,
        random_seed=SEED, verbose=False,
    )
    m_cat.fit(Xf_tr, yf_tr, eval_set=(Xf_va, yf_va), use_best_model=True)
    p_cat        = m_cat.predict_proba(Xf_va)[:, 1]
    oof_cat[va_idx] = p_cat
    auc_c = roc_auc_score(yf_va, p_cat)
    scores_cat.append(auc_c)
    print(f"    CatBoost  AUC={auc_c:.5f}")

    fold_auc_records.append({
        "fold": fold,
        "day_va_start": int(day_fold_va_start),
        "n_val": len(va_idx),
        "lgb_auc": auc_l, "xgb_auc": auc_x, "cat_auc": auc_c,
        "ens_auc": round(W_LGB*auc_l + W_XGB*auc_x + W_CAT*auc_c, 5),
    })

oof_ens_tr = W_LGB * oof_lgb + W_XGB * oof_xgb + W_CAT * oof_cat
auc_cv_ens = roc_auc_score(y_tr, oof_ens_tr)

print(f"\n  LightGBM  CV AUC: {np.mean(scores_lgb):.5f} +/- {np.std(scores_lgb):.5f}")
print(f"  XGBoost   CV AUC: {np.mean(scores_xgb):.5f} +/- {np.std(scores_xgb):.5f}")
print(f"  CatBoost  CV AUC: {np.mean(scores_cat):.5f} +/- {np.std(scores_cat):.5f}")
print(f"  Ensemble  CV AUC: {auc_cv_ens:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO FINAL Y EVALUACION EN VAL + TEST TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/10] Entrenamiento final + evaluacion en Val y Test temporal ...")

best_iter_lgb = max(1, int(np.mean([m.best_iteration for m in models_lgb])))
best_iter_xgb = max(1, int(np.mean([get_xgb_best_iteration(m) for m in models_xgb])))
print(f"  LightGBM best iter: {best_iter_lgb}  |  XGBoost best iter: {best_iter_xgb}")

# Train en conjunto TRAIN completo (70%)
print("  Entrenando LightGBM final en Train (70%) ...")
final_lgb = lgb.train(lgb_params,
                       lgb.Dataset(X_tr, label=y_tr),
                       num_boost_round=best_iter_lgb,
                       callbacks=[lgb.log_evaluation(200)])

print("  Entrenando XGBoost final en Train (70%) ...")
final_xgb = xgb.train(xgb_params,
                       xgb.DMatrix(X_tr, label=y_tr),
                       num_boost_round=best_iter_xgb,
                       verbose_eval=200)

print("  Entrenando CatBoost final en Train (70%) ...")
final_cat = CatBoostClassifier(
    iterations=5000, learning_rate=0.03, depth=6,
    l2_leaf_reg=5, min_data_in_leaf=50,
    subsample=0.70, colsample_bylevel=0.70,
    scale_pos_weight=w_pos, eval_metric="AUC",
    random_seed=SEED, verbose=200,
)
final_cat.fit(X_tr, y_tr)

# Logistic Regression (enfoque tradicional 1) — sobre train
print("  Entrenando Logistic Regression (baseline) ...")
scaler = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)
X_val_sc = scaler.transform(X_val)
X_te_sc  = scaler.transform(X_te)
lr_model = LogisticRegression(max_iter=1000, C=0.1,
                               class_weight="balanced",
                               solver="saga", random_state=SEED)
lr_model.fit(X_tr_sc, y_tr)

# Random Forest (enfoque tradicional 2) — muestra por velocidad
print("  Entrenando Random Forest (tradicional 2) ...")
sample_idx = np.random.RandomState(SEED).choice(
    len(y_tr), size=min(100_000, len(y_tr)), replace=False)
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=50,
    class_weight="balanced", n_jobs=-1, random_state=SEED)
rf_model.fit(X_tr.iloc[sample_idx], y_tr.iloc[sample_idx])

# Predicciones en Val y Test temporal
def predict_ensemble(X_arr, X_sc_arr):
    p_lgb = final_lgb.predict(X_arr)
    p_xgb = final_xgb.predict(xgb.DMatrix(X_arr))
    p_cat = final_cat.predict_proba(X_arr)[:, 1]
    p_lr  = lr_model.predict_proba(X_sc_arr)[:, 1]
    p_rf  = rf_model.predict_proba(X_arr)[:, 1]
    p_ens = W_LGB * p_lgb + W_XGB * p_xgb + W_CAT * p_cat
    return p_lgb, p_xgb, p_cat, p_lr, p_rf, p_ens

print("\n  Evaluando en Val temporal (15%) ...")
p_lgb_v, p_xgb_v, p_cat_v, p_lr_v, p_rf_v, p_ens_v = predict_ensemble(X_val, X_val_sc)

print("  Evaluando en Test temporal (15%) ...")
p_lgb_t, p_xgb_t, p_cat_t, p_lr_t, p_rf_t, p_ens_t = predict_ensemble(X_te, X_te_sc)

# Tabla de metricas completa
splits_eval = {
    "Train (CV-OOF)": (y_tr,  oof_ens_tr),
    "Val (temporal)": (y_val, p_ens_v),
    "Test (temporal)":(y_te,  p_ens_t),
}
models_eval = {
    "Logistic Reg.":  {"val": p_lr_v,  "test": p_lr_t},
    "Random Forest":  {"val": p_rf_v,  "test": p_rf_t},
    "LightGBM":       {"val": p_lgb_v, "test": p_lgb_t},
    "XGBoost":        {"val": p_xgb_v, "test": p_xgb_t},
    "CatBoost":       {"val": p_cat_v, "test": p_cat_t},
    "Ensemble":       {"val": p_ens_v, "test": p_ens_t},
}

metrics_table = {}
for name, preds in models_eval.items():
    metrics_table[name] = {
        "Val AUC":  round(roc_auc_score(y_val, preds["val"]),  5),
        "Test AUC": round(roc_auc_score(y_te,  preds["test"]), 5),
        "Val F1":   round(f1_score(y_val, (preds["val"]  >= 0.5).astype(int)), 5),
        "Test F1":  round(f1_score(y_te,  (preds["test"] >= 0.5).astype(int)), 5),
        "Val BaAcc":  round(balanced_accuracy_score(y_val, (preds["val"]  >= 0.5).astype(int)), 5),
        "Test BaAcc": round(balanced_accuracy_score(y_te,  (preds["test"] >= 0.5).astype(int)), 5),
    }

print(f"\n  {'Modelo':<18} {'Val AUC':>9} {'Test AUC':>9} {'Val F1':>8} {'Test F1':>8} {'Val BaAcc':>10} {'Test BaAcc':>10}")
print("  " + "-" * 78)
for name, m in metrics_table.items():
    print(f"  {name:<18} {m['Val AUC']:>9.5f} {m['Test AUC']:>9.5f} "
          f"{m['Val F1']:>8.5f} {m['Test F1']:>8.5f} "
          f"{m['Val BaAcc']:>10.5f} {m['Test BaAcc']:>10.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ANALISIS DE DEGRADACION TEMPORAL (Req. Seccion 3.3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/10] Analisis de degradacion temporal del modelo ...")

# Combinar val + test para el analisis temporal completo
days_valtest = np.concatenate([days_val, days_te])
y_valtest    = np.concatenate([y_val.values, y_te.values])
p_ens_valtest = np.concatenate([p_ens_v, p_ens_t])
p_lr_valtest  = np.concatenate([p_lr_v,  p_lr_t])
p_rf_valtest  = np.concatenate([p_rf_v,  p_rf_t])

deg_ens = temporal_degradation(y_valtest, p_ens_valtest, days_valtest, window_days=14)
deg_lr  = temporal_degradation(y_valtest, p_lr_valtest,  days_valtest, window_days=14)
deg_rf  = temporal_degradation(y_valtest, p_rf_valtest,  days_valtest, window_days=14)

print(f"  Ventanas analizadas: {len(deg_ens)}")
print(f"  AUC Ensemble — min: {deg_ens['auc'].min():.4f}  "
      f"max: {deg_ens['auc'].max():.4f}  "
      f"std: {deg_ens['auc'].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. POSTPROCESSING POR UID + SUBMISSIONS KAGGLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/10] Postprocessing por UID y generando submissions ...")

# Predicciones en Kaggle test (reentrenar en Train+Val+Test = 100% de datos)
print("  Entrenando modelos finales en 100% de datos (Train+Val+Test) ...")
X_full = train[feature_cols].fillna(-999)
y_full = train[TARGET].astype(int)

final_lgb_full = lgb.train(lgb_params,
                             lgb.Dataset(X_full, label=y_full),
                             num_boost_round=best_iter_lgb,
                             callbacks=[lgb.log_evaluation(-1)])
final_xgb_full = xgb.train(xgb_params,
                             xgb.DMatrix(X_full, label=y_full),
                             num_boost_round=best_iter_xgb,
                             verbose_eval=False)
final_cat_full = CatBoostClassifier(
    iterations=best_iter_lgb, learning_rate=0.03, depth=6,
    l2_leaf_reg=5, min_data_in_leaf=50,
    subsample=0.70, colsample_bylevel=0.70,
    scale_pos_weight=float((y_full==0).sum()/(y_full==1).sum()),
    random_seed=SEED, verbose=False)
final_cat_full.fit(X_full, y_full)

test_lgb = final_lgb_full.predict(X_kaggle_test)
test_xgb = final_xgb_full.predict(xgb.DMatrix(X_kaggle_test))
test_cat = final_cat_full.predict_proba(X_kaggle_test)[:, 1]
test_ens = W_LGB * test_lgb + W_XGB * test_xgb + W_CAT * test_cat

# UID postprocessing sobre test Kaggle
test_uid_df   = pd.DataFrame({"uid": test_uids, "pred": test_ens})
test_uid_mean = test_uid_df.groupby("uid")["pred"].transform("mean")
test_postproc = test_uid_mean.values

# UID postprocessing sobre val+test temporal para medir ganancia
oof_full_df  = pd.DataFrame({"uid": train["uid"].values,
                              "pred": np.concatenate([oof_ens_tr,
                                                      p_ens_v, p_ens_t]),
                              "label": y_full.values,
                              "day": train["TransactionDay"].values})
oof_uid_mean  = oof_full_df.groupby("uid")["pred"].transform("mean")
auc_postproc  = roc_auc_score(y_full, oof_uid_mean)
print(f"  OOF AUC sin postproc : {auc_cv_ens:.5f}")
print(f"  OOF AUC con postproc : {auc_postproc:.5f}  "
      f"(+{auc_postproc - auc_cv_ens:.5f})")

sub = pd.read_csv("sample_submission.csv")
sub["isFraud"] = test_ens
sub.to_csv("submission_v3.csv", index=False)
print("  Saved: submission_v3.csv  (sin postproc)")

sub["isFraud"] = test_postproc
sub.to_csv("submission_v3_postproc.csv", index=False)
print("  Saved: submission_v3_postproc.csv  (con UID postproc) <-- SUBIR ESTA")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURAS DE PUBLICACION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/10] Generando figuras ...")

palette_models = {
    "Logistic Reg.": P["neutral"],
    "Random Forest": P["success"],
    "LightGBM":      P["primary"],
    "XGBoost":       P["secondary"],
    "CatBoost":      P["warning"],
    "Ensemble":      P["accent"],
}

# FIG M01 — Division temporal del dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Division Temporal Estricta del Dataset — Sin Data Leakage\n"
             f"Train 70% | Val 15% | Test 15% (por orden cronologico de dias)",
             fontsize=13, fontweight="bold", color=P["text"])

split_colors  = {"Train": P["primary"], "Val": P["warning"], "Test": P["accent"]}
split_sizes   = [n_train, n_val, n_test]
split_labels  = [f"Train\n{n_train:,}\n({n_train/n_total*100:.0f}%)",
                 f"Val\n{n_val:,}\n({n_val/n_total*100:.0f}%)",
                 f"Test\n{n_test:,}\n({n_test/n_total*100:.0f}%)"]

bars = axes[0].bar(["Train", "Val", "Test"], split_sizes,
                   color=[split_colors[s] for s in ["Train","Val","Test"]],
                   edgecolor="white", linewidth=2, width=0.55)
for bar, label in zip(bars, split_labels):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Numero de Transacciones")
axes[0].set_title("Tamano de cada Particion", fontweight="bold")
clean_ax(axes[0]); axes[0].set_facecolor(P["bg"])

fr_splits = [train.loc[mask_train,"isFraud"].mean()*100,
             train.loc[mask_val,  "isFraud"].mean()*100,
             train.loc[mask_test, "isFraud"].mean()*100]
bars2 = axes[1].bar(["Train", "Val", "Test"], fr_splits,
                    color=[split_colors[s] for s in ["Train","Val","Test"]],
                    edgecolor="white", linewidth=2, width=0.55)
for bar, val in zip(bars2, fr_splits):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Tasa de Fraude (%)")
axes[1].set_title("Tasa de Fraude por Particion", fontweight="bold")
axes[1].axhline(train["isFraud"].mean()*100, color=P["neutral"],
                linestyle="--", linewidth=1.5, alpha=0.6, label="Media global")
axes[1].legend(fontsize=9)
clean_ax(axes[1]); axes[1].set_facecolor(P["bg"])

wm(fig); plt.tight_layout()
save_fig(fig, "MOD_01_temporal_split")

# FIG M02 — ROC curves Val + Test
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Curvas ROC — Evaluacion en Particiones Temporales Val y Test",
             fontsize=13, fontweight="bold", color=P["text"])

preds_dict = {
    "Logistic Reg.": (p_lr_v,  p_lr_t),
    "Random Forest": (p_rf_v,  p_rf_t),
    "LightGBM":      (p_lgb_v, p_lgb_t),
    "XGBoost":       (p_xgb_v, p_xgb_t),
    "CatBoost":      (p_cat_v, p_cat_t),
    "Ensemble":      (p_ens_v, p_ens_t),
}
for ax, (y_eval, split_name), idx in zip(
        axes,
        [(y_val, "Val (temporal)"), (y_te, "Test (temporal)")],
        [0, 1]):
    for name, (pv, pt) in preds_dict.items():
        p_eval = pv if idx == 0 else pt
        fpr, tpr, _ = roc_curve(y_eval, p_eval)
        auc_v = roc_auc_score(y_eval, p_eval)
        lw = 3 if name == "Ensemble" else 1.5
        ls = "-" if name == "Ensemble" else "--"
        ax.plot(fpr, tpr, label=f"{name} ({auc_v:.4f})",
                color=palette_models[name], linewidth=lw, linestyle=ls)
    ax.plot([0,1],[0,1],":", color=P["neutral"], alpha=0.4, linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"Curva ROC — {split_name}", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    clean_ax(ax); ax.set_facecolor(P["bg"])

wm(fig); plt.tight_layout()
save_fig(fig, "MOD_02_roc_curves_temporal")

# FIG M03 — Metricas comparativas (tabla visual)
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
fig.suptitle("Tabla de Metricas por Modelo y Particion Temporal\n"
             "Seccion 3.3 — Evaluacion en Entorno Dinamico",
             fontsize=13, fontweight="bold", color=P["text"])

col_labels = ["Modelo", "Val AUC", "Test AUC", "Val F1", "Test F1",
              "Val Bal.Acc", "Test Bal.Acc"]
cell_data = []
for name, m in metrics_table.items():
    cell_data.append([name,
                      f"{m['Val AUC']:.4f}", f"{m['Test AUC']:.4f}",
                      f"{m['Val F1']:.4f}",  f"{m['Test F1']:.4f}",
                      f"{m['Val BaAcc']:.4f}", f"{m['Test BaAcc']:.4f}"])

tbl = ax.table(cellText=cell_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
tbl.scale(1, 2.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor(P["primary"])
        cell.set_text_props(color="white", fontweight="bold")
    elif cell_data[r-1][0] == "Ensemble":
        cell.set_facecolor("#FFE8E8")
        cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor(P["bg"] if r % 2 == 0 else P["bg2"])
    cell.set_edgecolor(P["grid"])

wm(fig); plt.tight_layout()
save_fig(fig, "MOD_03_metrics_table")

# FIG M04 — Degradacion temporal del modelo (Req. Seccion 3.3)
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Analisis de Degradacion Temporal del Modelo — Seccion 3.3\n"
             "AUC por ventana deslizante de 14 dias en Val+Test",
             fontsize=13, fontweight="bold", color=P["text"])

# Panel 1: AUC por ventana
axes[0].fill_between(deg_ens["day_center"], deg_ens["auc"],
                     alpha=0.15, color=P["accent"])
axes[0].plot(deg_ens["day_center"], deg_ens["auc"],
             color=P["accent"], linewidth=2.5, marker="o", markersize=5,
             label=f"Ensemble (AUC={deg_ens['auc'].mean():.4f} avg)")
axes[0].plot(deg_lr["day_center"],  deg_lr["auc"],
             color=P["neutral"], linewidth=1.5, linestyle="--",
             label=f"Logistic Reg. (AUC={deg_lr['auc'].mean():.4f} avg)")
axes[0].plot(deg_rf["day_center"],  deg_rf["auc"],
             color=P["success"], linewidth=1.5, linestyle="--",
             label=f"Random Forest (AUC={deg_rf['auc'].mean():.4f} avg)")

# Banda de confianza +/- 1 std
std_w = deg_ens["auc"].rolling(3, center=True).std().fillna(0)
axes[0].fill_between(deg_ens["day_center"],
                     deg_ens["auc"] - std_w,
                     deg_ens["auc"] + std_w,
                     alpha=0.10, color=P["accent"])
axes[0].axhline(deg_ens["auc"].mean(), color=P["accent"],
                linestyle=":", linewidth=1.5, alpha=0.6)
axes[0].axhline(0.90, color=P["success"], linestyle="--",
                linewidth=1.5, alpha=0.5, label="Objetivo 0.90")

# Marcar corte train/val y val/test
ax0_ylim = (max(0.7, deg_ens["auc"].min()-0.02), min(1.0, deg_ens["auc"].max()+0.02))
axes[0].set_ylim(ax0_ylim)
axes[0].set_ylabel("ROC-AUC (ventana 14d)", fontsize=11)
axes[0].set_title("AUC a lo Largo del Tiempo — Detectando Degradaciones",
                  fontweight="bold")
axes[0].legend(fontsize=9, loc="lower left")
clean_ax(axes[0]); axes[0].set_facecolor(P["bg"])

# Panel 2: tasa de fraude por ventana (contexto del entorno)
axes[1].fill_between(deg_ens["day_center"], deg_ens["fraud_rate"],
                     alpha=0.25, color=P["fraud"])
axes[1].plot(deg_ens["day_center"], deg_ens["fraud_rate"],
             color=P["fraud"], linewidth=2, marker="s", markersize=4)
axes[1].set_ylabel("Tasa de Fraude (%)", fontsize=11)
axes[1].set_xlabel("Dia (desde referencia)", fontsize=11)
axes[1].set_title("Tasa de Fraude Real por Ventana — Entorno Dinamico",
                  fontweight="bold")
clean_ax(axes[1]); axes[1].set_facecolor(P["bg"])

plt.tight_layout(h_pad=0.5)
wm(fig)
save_fig(fig, "MOD_04_temporal_degradation")

# FIG M05 — TimeSeriesSplit: AUC por fold cronologico
fig, ax = plt.subplots(figsize=(12, 5))
fold_df = pd.DataFrame(fold_auc_records)
x = range(1, len(fold_df)+1)
ax.plot(x, fold_df["lgb_auc"], color=P["primary"],  marker="o", linewidth=2,
        label="LightGBM")
ax.plot(x, fold_df["xgb_auc"], color=P["secondary"], marker="s", linewidth=2,
        label="XGBoost")
ax.plot(x, fold_df["cat_auc"], color=P["warning"],   marker="^", linewidth=2,
        label="CatBoost")
ax.plot(x, fold_df["ens_auc"], color=P["accent"],    marker="D", linewidth=3,
        markersize=8, label="Ensemble")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i}\n(dia {row.day_va_start}+)"
                    for i, row in fold_df.iterrows()], fontsize=9)
ax.set_ylabel("AUC (Validacion del Fold)")
ax.set_title("AUC por Fold — TimeSeriesSplit Cronologico\n"
             "El eje X avanza en el tiempo (sin shuffle)",
             fontsize=13, fontweight="bold", color=P["text"])
ax.legend(fontsize=10)
ax.axhline(0.90, color=P["success"], linestyle="--", linewidth=1.5,
           alpha=0.6, label="Objetivo 0.90")
clean_ax(ax); ax.set_facecolor(P["bg"])
wm(fig); plt.tight_layout()
save_fig(fig, "MOD_05_timeseries_cv_folds")

# FIG M06 — Confusion matrix (Test temporal)
cm = confusion_matrix(y_te, (p_ens_t >= 0.5).astype(int))
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(7, 6))
cmap_cm = sns.light_palette(P["primary"], as_cmap=True)
sns.heatmap(cm, annot=False, cmap=cmap_cm, linewidths=2,
            linecolor="white", ax=ax, cbar=False)
for i in range(2):
    for j in range(2):
        ax.text(j+0.5, i+0.38, f"{cm[i,j]:,}",
                ha="center", fontsize=18, fontweight="bold",
                color="white" if cm_pct[i,j] > 50 else P["text"])
        ax.text(j+0.5, i+0.65, f"({cm_pct[i,j]:.1f}%)",
                ha="center", fontsize=11,
                color="white" if cm_pct[i,j] > 50 else P["neutral"])
ax.set_xticklabels(["Legitima","Fraudulenta"], fontsize=11)
ax.set_yticklabels(["Legitima","Fraudulenta"], fontsize=11, rotation=0)
ax.set_xlabel("Prediccion", fontsize=12, fontweight="bold")
ax.set_ylabel("Real", fontsize=12, fontweight="bold")
ax.set_title("Matriz de Confusion — Ensemble sobre Test Temporal (15%)",
             fontsize=12, fontweight="bold", color=P["text"], pad=14)
wm(fig); plt.tight_layout()
save_fig(fig, "MOD_06_confusion_matrix_test")

# FIG M07 — Feature importance LightGBM
fi = (pd.DataFrame({
    "feature":    final_lgb.feature_name(),
    "importance": final_lgb.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False).head(25))
fig, ax = plt.subplots(figsize=(11, 9))
cmap_fi = sns.light_palette(P["primary"], n_colors=25)
ax.barh(fi["feature"][::-1], fi["importance"][::-1],
        color=cmap_fi, edgecolor="white", linewidth=0.5, height=0.75)
ax.set_xlabel("Information Gain")
ax.set_title("Top 25 Features — LightGBM (Information Gain)\n"
             "Entrenado en Train temporal (70%)",
             fontsize=13, fontweight="bold", color=P["text"])
clean_ax(ax); ax.set_facecolor(P["bg"])
ax.grid(axis="x"); ax.set_axisbelow(True)
wm(fig); plt.tight_layout()
save_fig(fig, "MOD_07_feature_importance")

# ─────────────────────────────────────────────────────────────────────────────
# 10. GUARDAR ARTEFACTOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10/10] Guardando artefactos ...")

artifacts = {
    "lgb_model":     final_lgb_full,
    "xgb_model":     final_xgb_full,
    "cat_model":     final_cat_full,
    "lr_model":      lr_model,
    "rf_model":      rf_model,
    "feature_cols":  feature_cols,
    "le_map":        le_map,
    "scaler":        scaler,
    "cat_cols":      cat_cols,
    "weights":       {"lgb": W_LGB, "xgb": W_XGB, "cat": W_CAT},
    "metrics":       metrics_table,
    "threshold":     0.5,
    "train_stats": {
        "fraud_rate":    float(y_full.mean()),
        "n_samples":     int(len(y_full)),
        "n_train":       int(n_train),
        "n_val":         int(n_val),
        "n_test":        int(n_test),
        "cut_train":     int(cut_train),
        "cut_val":       int(cut_val),
        "auc_cv":        float(auc_cv_ens),
        "auc_val":       float(roc_auc_score(y_val, p_ens_v)),
        "auc_test":      float(roc_auc_score(y_te,  p_ens_t)),
        "auc_postproc":  float(auc_postproc),
    },
}
joblib.dump(artifacts, "model_artifacts/fraud_model.pkl")
print("  Saved: model_artifacts/fraud_model.pkl")

# Baseline para drift monitoring (basado en val+test temporal)
pd.DataFrame({
    "oof_score":      p_ens_valtest,
    "true_label":     y_valtest,
    "TransactionAmt": train.loc[mask_val | mask_test, "TransactionAmt"].values,
    "TransactionDay": days_valtest,
}).to_parquet("model_artifacts/oof_baseline.parquet", index=False)
print("  Saved: model_artifacts/oof_baseline.parquet")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN FINAL — Modelado v3")
print("=" * 65)
print(f"\n  Division temporal:")
print(f"    Train : {n_train:,} muestras ({n_train/n_total*100:.0f}%) — dias <= {cut_train}")
print(f"    Val   : {n_val:,} muestras ({n_val/n_total*100:.0f}%) — dias {cut_train+1}-{cut_val}")
print(f"    Test  : {n_test:,} muestras ({n_test/n_total*100:.0f}%) — dias > {cut_val}")
print(f"\n  CV temporal (TimeSeriesSplit, 5 folds sobre Train):")
print(f"    LightGBM : {np.mean(scores_lgb):.5f} +/- {np.std(scores_lgb):.5f}")
print(f"    XGBoost  : {np.mean(scores_xgb):.5f} +/- {np.std(scores_xgb):.5f}")
print(f"    CatBoost : {np.mean(scores_cat):.5f} +/- {np.std(scores_cat):.5f}")
print(f"    Ensemble : {auc_cv_ens:.5f}")
print(f"\n  Evaluacion en particiones temporales independientes:")
print(f"    Ensemble Val AUC  : {roc_auc_score(y_val, p_ens_v):.5f}")
print(f"    Ensemble Test AUC : {roc_auc_score(y_te, p_ens_t):.5f}")
print(f"    + UID Postproc    : {auc_postproc:.5f}")
print(f"\n  Referencia ganador Kaggle (private LB): 0.94588")
print(f"\n  Subir: submission_v3_postproc.csv")
print(f"  Figuras (7) -> ./figures/MOD_0*.png")
print(f"  Artefactos  -> ./model_artifacts/")
print("=" * 65)
print("  Modelado v3 completo.\n")
