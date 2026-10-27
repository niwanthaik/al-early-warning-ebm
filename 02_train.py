"""
Step 2: hyperparameter tuning and training.

New algorithm (not taught in lectures): Explainable Boosting Machine (EBM, InterpretML).
Comparison models taught in lectures: Logistic Regression, Decision Tree, k-NN, Random Forest, XGBoost.
Black-box reference (state-of-the-art GBDT): LightGBM.

Tuning: RandomizedSearchCV, 3-fold stratified CV, ROC-AUC, on a 60k stratified subsample of the
training set (k-NN: 30k). Final models are refit on the full training set (k-NN reference set: 80k).
Outputs: models/*.pkl, results/tuning.json
"""
import json, time, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

import pipelines as PL
from config import PROC, MODELS, RESULTS, SEED, N_JOBS, TUNE_ROWS, KNN_ROWS, QUICK

warnings.filterwarnings("ignore")
tr = pd.read_parquet(PROC / "train.parquet")
if QUICK:
    tr = tr.sample(20_000, random_state=SEED)
X_train, y_train = tr.drop(columns="eligible"), tr["eligible"]
X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=TUNE_ROWS, stratify=y_train, random_state=SEED)
cv3 = StratifiedKFold(3, shuffle=True, random_state=SEED)
T = {}


def clean(d):
    return {k.split("__", 1)[1]: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def search(name, pipe, space, n_iter, Xs=X_tune, ys=y_tune, grid=False):
    t = time.time()
    s = (GridSearchCV(pipe, space, scoring="roc_auc", cv=cv3, n_jobs=1) if grid else
         RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv3, random_state=SEED, n_jobs=1))
    s.fit(Xs, ys)
    cvr = pd.DataFrame(s.cv_results_).sort_values("mean_test_score", ascending=False)
    T[name] = {"best_params": clean(s.best_params_), "cv_auc": float(s.best_score_),
               "cv_auc_std": float(cvr.iloc[0].std_test_score), "search_seconds": round(time.time() - t),
               "n_candidates": len(cvr),
               "top5": [{"params": clean(r.params), "auc": round(r.mean_test_score, 4)} for r in cvr.head(5).itertuples()],
               "space": {k.split("__", 1)[1]: str(v) for k, v in space.items()}}
    print(f"[{name}] CV AUC {s.best_score_:.4f}  {T[name]['best_params']}  ({T[name]['search_seconds']}s)", flush=True)
    return T[name]["best_params"]


def fit_save(name, fn, pipe, Xs=X_train, ys=y_train):
    t = time.time(); pipe.fit(Xs, ys); T[name]["fit_seconds"] = round(time.time() - t, 1)
    joblib.dump(pipe, MODELS / f"{fn}.pkl", compress=("xz", 6))
    T[name]["pickle_mb"] = round((MODELS / f"{fn}.pkl").stat().st_size / 1e6, 2)
    print(f"[{name}] fitted in {T[name]['fit_seconds']}s, {T[name]['pickle_mb']} MB", flush=True)


# ------------------------------------------------------------------ taught comparison models
p = search("Logistic Regression", PL.make(PL.onehot_prep(), LogisticRegression(max_iter=3000)),
           {"model__C": [0.01, 0.1, 0.5, 1.0, 5.0]}, 0, grid=True)
fit_save("Logistic Regression", "logistic_regression", PL.make(PL.onehot_prep(), LogisticRegression(**p, max_iter=3000)))

p = search("Decision Tree", PL.make(PL.onehot_prep(False), DecisionTreeClassifier(random_state=SEED)),
           {"model__max_depth": randint(4, 16), "model__min_samples_leaf": randint(20, 400),
            "model__criterion": ["gini", "entropy"]}, 12)
fit_save("Decision Tree", "decision_tree", PL.make(PL.onehot_prep(False), DecisionTreeClassifier(**p, random_state=SEED)))

Xk, _, yk, _ = train_test_split(X_tune, y_tune, train_size=min(30_000, len(X_tune) // 2), stratify=y_tune, random_state=SEED)
p = search("k-NN", PL.make(PL.onehot_prep(), KNeighborsClassifier(n_jobs=N_JOBS)),
           {"model__n_neighbors": [25, 51, 101, 201], "model__weights": ["uniform", "distance"]}, 0, Xk, yk, grid=True)
Xkr, _, ykr, _ = train_test_split(X_train, y_train, train_size=KNN_ROWS, stratify=y_train, random_state=SEED)
fit_save("k-NN", "knn", PL.make(PL.onehot_prep(), KNeighborsClassifier(**p, n_jobs=N_JOBS)), Xkr, ykr)

p = search("Random Forest", PL.make(PL.onehot_prep(False), RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED)),
           {"model__n_estimators": randint(150, 401), "model__max_depth": [12, 16, 20, 25, None],
            "model__min_samples_leaf": randint(5, 80), "model__max_features": ["sqrt", 0.2, 0.3, 0.5],
            "model__max_samples": uniform(0.5, 0.5)}, 10)
fit_save("Random Forest", "random_forest", PL.make(PL.onehot_prep(False), RandomForestClassifier(**p, n_jobs=N_JOBS, random_state=SEED)))

p = search("XGBoost", PL.make(PL.boost_prep(), XGBClassifier(tree_method="hist", eval_metric="auc", n_jobs=N_JOBS, random_state=SEED)),
           {"model__n_estimators": randint(200, 901), "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_depth": randint(3, 9), "model__min_child_weight": randint(1, 31), "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4), "model__gamma": uniform(0, 5), "model__reg_lambda": loguniform(0.1, 10)}, 14)
fit_save("XGBoost", "xgboost", PL.make(PL.boost_prep(), XGBClassifier(**p, tree_method="hist", eval_metric="auc", n_jobs=N_JOBS, random_state=SEED)))

# ------------------------------------------------------------------ black-box reference
p = search("LightGBM", PL.make(PL.boost_prep(), LGBMClassifier(n_jobs=N_JOBS, random_state=SEED, verbose=-1)),
           {"model__n_estimators": randint(200, 1201), "model__learning_rate": loguniform(0.01, 0.15),
            "model__num_leaves": randint(15, 256), "model__max_depth": [-1, 8, 12, 16], "model__min_child_samples": randint(20, 401),
            "model__subsample": uniform(0.6, 0.4), "model__subsample_freq": [1], "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_alpha": loguniform(1e-3, 5), "model__reg_lambda": loguniform(1e-3, 10)}, 16)
fit_save("LightGBM", "lightgbm", PL.make(PL.boost_prep(), LGBMClassifier(**p, n_jobs=N_JOBS, random_state=SEED, verbose=-1)))

# ------------------------------------------------------------------ NEW ALGORITHM: Explainable Boosting Machine
# Search space rationale:
#   interactions      number of pairwise terms f_ij(xi,xj) added after the main effects (FAST detection);
#                     the EDA showed stream × CGT and gender × stream interactions
#   learning_rate     shrinkage of each cyclic boosting step (small = smoother shape functions)
#   max_bins          resolution of numeric shape functions (CGT has 0–100 distinct scores)
#   max_leaves        leaves per tiny tree inside the cyclic booster (2–3 keeps each step a simple split)
#   min_samples_leaf  minimum students per leaf: regularises noisy bins
#   smoothing_rounds  initial rounds with random splits -> smoother, less jagged shape functions
EBM_BASE = dict(outer_bags=4, max_rounds=8000, early_stopping_rounds=100, n_jobs=N_JOBS, random_state=SEED)
p = search("EBM", PL.make(PL.ebm_prep(), ExplainableBoostingClassifier(**EBM_BASE)),
           {"model__interactions": [20, 40, 80], "model__learning_rate": loguniform(0.005, 0.05),
            "model__max_bins": [256, 512, 1024], "model__max_leaves": [2, 3], "model__min_samples_leaf": [2, 10, 25, 50],
            "model__smoothing_rounds": [0, 50, 200]}, 10)
ebm_final = {**p, "outer_bags": 8}          # more bags for the final model = smoother, more stable shape functions
T["EBM"]["final_params"] = ebm_final
fit_save("EBM", "ebm", PL.make(PL.ebm_prep(), ExplainableBoostingClassifier(**{**EBM_BASE, **ebm_final})))

# same configuration without the gender feature (fairness / ethics comparison)
T["EBM (no gender)"] = {"best_params": ebm_final, "note": "EBM configuration re-used; gender column removed"}
fit_save("EBM (no gender)", "ebm_no_gender",
         PL.make(PL.ebm_prep(gender=False), ExplainableBoostingClassifier(**{**EBM_BASE, **ebm_final})))

(RESULTS / "tuning.json").write_text(json.dumps(T, indent=1, default=str))
print("done")
