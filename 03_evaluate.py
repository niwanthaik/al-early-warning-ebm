"""
Step 3: evaluation on the untouched test set.

  * metrics: ROC-AUC (95% bootstrap CI), PR-AUC for the at-risk class, accuracy, balanced accuracy,
    precision/recall/F1 (macro), Brier score, log-loss, train AUC (overfitting), latency, size
  * paired bootstrap ΔAUC of every model against the EBM
  * ROC, precision-recall and calibration curves; confusion matrices
  * early-warning threshold for the EBM chosen on the VALIDATION set (catch 70% of at-risk students)
Outputs: results/metrics.json, results/test_predictions.parquet, figures/eval_*.png
"""
import json, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score, brier_score_loss, log_loss, roc_curve,
                             precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)

from config import PROC, MODELS, RESULTS, FIGS, SEED, BOOT, QUICK

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

MODELS_ORDER = {  # name: (pickle, group)
    "Logistic Regression": ("logistic_regression", "Taught"), "Decision Tree": ("decision_tree", "Taught"),
    "k-NN": ("knn", "Taught"), "Random Forest": ("random_forest", "Taught"), "XGBoost": ("xgboost", "Taught"),
    "LightGBM": ("lightgbm", "Black-box reference"),
    "EBM": ("ebm", "New algorithm"), "EBM (no gender)": ("ebm_no_gender", "New algorithm")}
COLORS = {"Taught": "#8faadc", "Black-box reference": "#7f7f7f", "New algorithm": "#c55a11"}

load = lambda n: pd.read_parquet(PROC / f"{n}.parquet")
tr, va, te = load("train"), load("val"), load("test")
if QUICK:
    tr, va, te = tr.sample(20_000, random_state=SEED), va.sample(5_000, random_state=SEED), te.sample(8_000, random_state=SEED)
split = lambda d: (d.drop(columns="eligible"), d["eligible"].values)
(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(tr), split(va), split(te)
X_tr_s = X_tr.sample(min(50_000, len(X_tr)), random_state=SEED); y_tr_s = tr.loc[X_tr_s.index, "eligible"].values
tuning = json.loads((RESULTS / "tuning.json").read_text())

M, P, rows = {}, {}, []
for name, (fn, group) in MODELS_ORDER.items():
    M[name] = joblib.load(MODELS / f"{fn}.pkl")
    t = time.time(); P[name] = M[name].predict_proba(X_te)[:, 1]; pred_s = time.time() - t
    p_tr = M[name].predict_proba(X_tr_s)[:, 1]
    yhat = (P[name] >= 0.5).astype(int)
    rows.append({"model": name, "group": group,
                 "ROC-AUC": roc_auc_score(y_te, P[name]), "PR-AUC (at-risk)": average_precision_score(1 - y_te, 1 - P[name]),
                 "accuracy": accuracy_score(y_te, yhat), "balanced acc": balanced_accuracy_score(y_te, yhat),
                 "precision (macro)": precision_score(y_te, yhat, average="macro"),
                 "recall (macro)": recall_score(y_te, yhat, average="macro"),
                 "F1 (macro)": f1_score(y_te, yhat, average="macro"),
                 "Brier": brier_score_loss(y_te, P[name]), "log-loss": log_loss(y_te, P[name]),
                 "train ROC-AUC": roc_auc_score(y_tr_s, p_tr),
                 "fit time (s)": tuning.get(name, {}).get("fit_seconds"),
                 "ms / 1k preds": round(pred_s / len(X_te) * 1e6, 1),
                 "size (MB)": tuning.get(name, {}).get("pickle_mb")})
    print(f"{name:20s} AUC {rows[-1]['ROC-AUC']:.4f}", flush=True)
res = pd.DataFrame(rows).set_index("model")
res["overfit gap"] = res["train ROC-AUC"] - res["ROC-AUC"]

# ------------------------------------------------------------------ bootstrap CIs + paired ΔAUC vs EBM
rng = np.random.default_rng(SEED)
boots = [rng.integers(0, len(y_te), len(y_te)) for _ in range(BOOT)]
A = {n: np.array([roc_auc_score(y_te[b], P[n][b]) for b in boots]) for n in P}
res["AUC CI low"] = [np.percentile(A[n], 2.5) for n in res.index]
res["AUC CI high"] = [np.percentile(A[n], 97.5) for n in res.index]
pairs = []
for n in P:
    if n == "EBM":
        continue
    d = A["EBM"] - A[n]
    lo, hi = np.percentile(d, [2.5, 97.5])
    pairs.append({"model": n, "ΔAUC (EBM − model)": d.mean(), "CI low": lo, "CI high": hi,
                  "verdict": "EBM better" if lo > 0 else ("EBM worse" if hi < 0 else "no significant difference")})
pairs = pd.DataFrame(pairs).set_index("model")

# ------------------------------------------------------------------ early-warning threshold (validation set)
p_va = M["EBM"].predict_proba(X_va)[:, 1]
grid = np.linspace(0.3, 0.9, 121)
rec = np.array([recall_score(1 - y_va, (p_va < t).astype(int)) for t in grid])
thr = float(grid[np.argmax(rec >= 0.70)])
def op(t, p):
    f, yr = (p < t).astype(int), 1 - y_te
    return {"threshold": round(t, 3), "at-risk recall": recall_score(yr, f), "at-risk precision": precision_score(yr, f),
            "flagged %": f.mean(), "accuracy": accuracy_score(y_te, 1 - f), "balanced acc": balanced_accuracy_score(y_te, 1 - f)}
ew = pd.DataFrame({"default (0.5)": op(0.5, P["EBM"]), "early warning": op(thr, P["EBM"])}).T

# ------------------------------------------------------------------ figures
order = res.sort_values("ROC-AUC").index
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(order, res.loc[order, "ROC-AUC"], color=[COLORS[res.loc[n, "group"]] for n in order],
        xerr=[res.loc[order, "ROC-AUC"] - res.loc[order, "AUC CI low"], res.loc[order, "AUC CI high"] - res.loc[order, "ROC-AUC"]])
for i, n in enumerate(order):
    ax.text(res.loc[n, "ROC-AUC"] + .002, i, f"{res.loc[n, 'ROC-AUC']:.3f}", va="center", fontsize=8)
ax.set_xlim(0.68, 0.79); ax.set_xlabel("test ROC-AUC (95% CI)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], labels=list(COLORS), fontsize=7, loc="lower right")
ax.set_title("Model comparison on 58k unseen candidates")
plt.tight_layout(); plt.savefig(FIGS / "eval_auc.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
show = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "EBM"]
for n in show:
    lw = 2.4 if n == "EBM" else 1.1
    f, t, _ = roc_curve(y_te, P[n]); ax[0].plot(f, t, lw=lw, label=f"{n} ({res.loc[n, 'ROC-AUC']:.3f})")
    pr, rc, _ = precision_recall_curve(1 - y_te, 1 - P[n]); ax[1].plot(rc, pr, lw=lw, label=n)
    fp, mp = calibration_curve(y_te, P[n], n_bins=12); ax[2].plot(mp, fp, marker="o", ms=3, lw=lw, label=n)
ax[0].plot([0, 1], [0, 1], "k--", lw=.6); ax[0].set(title="ROC", xlabel="false positive rate", ylabel="true positive rate")
ax[1].axhline((1 - y_te).mean(), c="grey", ls="--", lw=.6); ax[1].set(title="Precision–recall (at-risk class)", xlabel="recall", ylabel="precision")
ax[2].plot([0, 1], [0, 1], "k--", lw=.6); ax[2].set(title="Calibration", xlabel="predicted P(eligible)", ylabel="observed rate")
for a in ax: a.legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(FIGS / "eval_curves.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, (n, t) in zip(ax, [("XGBoost", 0.5), ("EBM", 0.5), ("EBM", thr)]):
    ConfusionMatrixDisplay(confusion_matrix(y_te, (P[n] >= t).astype(int)), display_labels=["Not eligible", "Eligible"]).plot(
        ax=a, colorbar=False, values_format=",")
    a.set_title(f"{n}, threshold {t:.2f}")
plt.tight_layout(); plt.savefig(FIGS / "eval_confusion.png"); plt.close()

# ------------------------------------------------------------------ save
pd.DataFrame({n: P[n] for n in P}).assign(y=y_te).to_parquet(RESULTS / "test_predictions.parquet")
out = {"metrics": res.reset_index().to_dict(orient="records"), "pairs_vs_ebm": pairs.reset_index().to_dict(orient="records"),
       "early_warning_threshold": thr, "early_warning": ew.reset_index(names="setting").to_dict(orient="records"),
       "roc": {n: dict(zip(["fpr", "tpr"], [a[np.linspace(0, len(a) - 1, 200).astype(int)].round(4).tolist()
                                            for a in roc_curve(y_te, P[n])[:2]])) for n in P},
       "confusion": {n: confusion_matrix(y_te, (P[n] >= .5).astype(int)).tolist() for n in P},
       "n_test": int(len(y_te)), "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "eligible_rate": float(y_tr.mean())}
(RESULTS / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
pd.set_option("display.width", 200)
print(res[["group", "ROC-AUC", "AUC CI low", "AUC CI high", "PR-AUC (at-risk)", "accuracy", "balanced acc", "F1 (macro)", "Brier", "overfit gap"]].round(4))
print(pairs.round(4)); print(ew.round(3))
