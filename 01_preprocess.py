"""
Step 1: data loading, validation, ethics checks, cleaning, feature engineering, EDA and splitting.

Outputs
  data/processed/{train,val,test}.parquet   unified feature frames + target
  data/holdout_candidate.csv                one real candidate kept out of everything (inference demo)
  results/data_summary.json                 quality, validation, leakage, ethics and split statistics
  figures/eda_*.png                         exploratory figures
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import al_features as F
from config import RAW_FILE, PROC, DATA, RESULTS, FIGS, SEED, SPLIT, OFFICIAL

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
S = {}

# ------------------------------------------------------------------ 1. load + data quality
raw = pd.read_csv(RAW_FILE, dtype=str)
S["raw_rows"], S["raw_cols"] = raw.shape
dirty = ["Absent", "Withheld", "Invalid nic error", "Invalid day error", "Major error", "-"]
cols = ["stream", "Zscore", "sub1_r", "sub2_r", "sub3_r", "cgt_r", "ge_r", "birth_year", "birth_month", "gender"]
quality = pd.DataFrame({c: [int(raw[c].isna().sum())] + [int((raw[c] == t).sum()) for t in dirty] for c in cols},
                       index=["NaN"] + dirty).T
S["quality"] = quality.to_dict(orient="index")

# ------------------------------------------------------------------ 2. cohort + target
sat = F.filter_sat_exam(raw)
y = F.make_target(sat)
S["sat_all_three"], S["absent_or_withheld_removed"] = len(sat), len(raw) - len(sat)
S["eligible_n"], S["eligible_rate"] = int(y.sum()), float(y.mean())
S["validation"] = {"official_sat": OFFICIAL["sat"], "official_qualified": OFFICIAL["qualified"],
                   "official_rate": OFFICIAL["rate"], "coverage_sat": len(sat) / OFFICIAL["sat"],
                   "coverage_qualified": int(y.sum()) / OFFICIAL["qualified"]}

# ------------------------------------------------------------------ 3. leakage audit + derived stream
S["leakage"] = {
    "stream_blank_vs_eligible": pd.crosstab(sat.stream.eq("-"), y).to_dict(),
    "zscore_blank_vs_eligible": pd.crosstab(sat.Zscore.eq("-"), y).to_dict()}
passed = sat[sat.stream != "-"]
der = F.derive_stream(passed)
std = passed.stream.isin(F.STREAMS)
S["derived_stream_agreement"] = {"overall": float((der == passed.stream).mean()),
                                 "standard_streams": float((der[std] == passed.stream[std]).mean())}

# ------------------------------------------------------------------ 4. ethics: re-identification risk (k-anonymity)
# Quasi-identifiers available in the public file vs. the ones this project actually uses.
def k_anon(df, qi):
    k = df.groupby(qi, dropna=False).size()
    sizes = df.groupby(qi, dropna=False)[qi[0]].transform("size")
    return {"unique_records_pct": float((sizes == 1).mean()), "k_lt_5_pct": float((sizes < 5).mean()),
            "median_k": float(sizes.median()), "groups": int(len(k))}
qi_df = sat.assign(combo=F.build_cat_features(sat)["combo"])
S["k_anonymity"] = {
    "public_file: birth_year+month+day+gender+combo": k_anon(qi_df, ["birth_year", "birth_month", "birth_day", "gender", "combo"]),
    "used_in_model: birth_year+month+gender+combo": k_anon(qi_df, ["birth_year", "birth_month", "gender", "combo"]),
    "used_in_model_coarse: age+gender+stream": k_anon(qi_df.assign(stream=F.derive_stream(sat)), ["birth_year", "gender", "stream"])}

# ------------------------------------------------------------------ 5. features
X = F.unified_frame(sat)
_age_raw = 2020 - pd.to_numeric(sat["birth_year"], errors="coerce")
S["implausible_age_set_missing"] = int((_age_raw.notna() & ~_age_raw.between(15, 70)).sum())
E = X.assign(eligible=y.values)

miss = {}
for name, m in {"CGT score": X.cgt_absent == 1, "General English": X.ge_absent == 1,
                "age (NIC error)": X.age_at_exam.isna(), "gender": X.gender.isna()}.items():
    miss[name] = {"missing_pct": float(m.mean()), "pass_if_missing": float(y[m].mean()), "pass_if_present": float(y[~m].mean())}
S["missingness"] = miss
cc = X.combo.value_counts()
S["cardinality"] = {"subjects": int(pd.concat([sat.sub1, sat.sub2, sat.sub3]).nunique()), "combinations": int(len(cc)),
                    "top20_share": float(cc.head(20).sum() / len(X)), "combos_lt_30": int((cc < 30).sum())}
S["cohort_effect"] = {int(a): {"jan": float(y[(X.age_at_exam == a) & (X.birth_month == 1)].mean()),
                               "feb_dec": float(y[(X.age_at_exam == a) & (X.birth_month > 1)].mean())} for a in [18, 19, 20]}
S["pass_by_gender"] = E.groupby("gender")["eligible"].mean().to_dict()
S["pass_by_stream"] = E.groupby("stream")["eligible"].mean().to_dict()
S["features"] = {"numeric": F.NUMERIC_FEATURES, "categorical": F.CATEGORICAL_FEATURES, "flags": F.FLAG_FEATURES,
                 "subject_indicators": len(F.SUBJECT_FEATURES)}

# ------------------------------------------------------------------ 6. EDA figures
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
y.value_counts().sort_index().plot.bar(ax=ax[0], color=["#c0504d", "#4f81bd"])
ax[0].set_xticklabels(["Not eligible", "Eligible"], rotation=0); ax[0].set_title(f"Target ({y.mean():.1%} eligible)")
for lab, c in [(0, "#c0504d"), (1, "#4f81bd")]:
    E.loc[E.eligible == lab, "cgt_score"].plot.hist(ax=ax[1], bins=40, alpha=.55, color=c, density=True)
ax[1].set(title="CGT score by outcome", xlabel="CGT score"); ax[1].legend(["Not eligible", "Eligible"], fontsize=7)
E.groupby("stream")["eligible"].mean().sort_values().plot.barh(ax=ax[2], color="#4f81bd")
ax[2].axvline(y.mean(), ls="--", c="grey"); ax[2].set(title="Pass rate by stream", ylabel="")
plt.tight_layout(); plt.savefig(FIGS / "eda_overview.png"); plt.close()

E["cgt_band"] = pd.cut(E.cgt_score, [0, 25, 35, 45, 55, 100], labels=["≤25", "26-35", "36-45", "46-55", ">55"])
heat = E.pivot_table(index="stream", columns="cgt_band", values="eligible", aggfunc="mean", observed=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
im = ax[0].imshow(heat.values, cmap="RdYlBu", vmin=0.2, vmax=0.95, aspect="auto")
ax[0].set_xticks(range(heat.shape[1]), heat.columns); ax[0].set_yticks(range(heat.shape[0]), [s.title() for s in heat.index])
for (i, j), v in np.ndenumerate(heat.values):
    ax[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
ax[0].set(title="Pass rate: stream × CGT band", xlabel="CGT score band"); plt.colorbar(im, ax=ax[0])
g = E.pivot_table(index="stream", columns="gender", values="eligible", aggfunc="mean")
g.index = [s.title() for s in g.index]
g.plot.barh(ax=ax[1]); ax[1].set(title="Pass rate: stream × gender", xlabel="pass rate")
plt.tight_layout(); plt.savefig(FIGS / "eda_interactions.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 3.2))
a = E[E.age_at_exam.between(17, 25)].groupby("age_at_exam")["eligible"].mean()
a.plot(marker="o", ax=ax[0]); ax[0].set(title="Pass rate by age at exam", xlabel="age", ylabel="pass rate")
b = E[E.age_at_exam.between(18, 19)].groupby(["age_at_exam", "birth_month"])["eligible"].mean().unstack(0)
b.plot(marker="o", ax=ax[1]); ax[1].set(title="Pass rate by birth month (ages 18–19)", xlabel="birth month", ylabel="pass rate")
ax[1].set_xticks(range(1, 13)); ax[1].legend(title="age", fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / "eda_age_cohort.png"); plt.close()

top = cc.head(15).index
cr = E[E.combo.isin(top)].groupby("combo")["eligible"].mean().sort_values()
ax = cr.plot.barh(figsize=(9, 4.6), color="#4f81bd"); ax.axvline(y.mean(), ls="--", c="grey")
ax.set(title="Pass rate for the 15 most common subject combinations", xlabel="pass rate", ylabel="")
ax.tick_params(axis="y", labelsize=7); plt.tight_layout(); plt.savefig(FIGS / "eda_combos.png"); plt.close()

# ------------------------------------------------------------------ 7. split: holdout + 70/10/20 stratified
hold_idx = X.sample(1, random_state=7).index
# files shipped with the front end are minimised: no day of birth, no row index, no Z-score/ranks
SHARE_DROP = ["birth_day", "index", "Zscore", "district_rank", "island_rank"]
share = lambda idx: sat.loc[idx].drop(columns=[c for c in SHARE_DROP if c in sat.columns])
share(hold_idx).to_csv(DATA / "holdout_candidate.csv", index=False)
Xr, yr = X.drop(hold_idx), y.drop(hold_idx)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(Xr, yr, test_size=SPLIT["val"] + SPLIT["test"], stratify=yr, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=SPLIT["test"] / (SPLIT["val"] + SPLIT["test"]),
                                          stratify=y_tmp, random_state=SEED)
for name, Xs, ys in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
    Xs.assign(eligible=ys.values).to_parquet(PROC / f"{name}.parquet")
# raw-format rows for the front end (batch demo, LIME background)
share(X_te.sample(500, random_state=1).index).to_csv(DATA / "sample_upload.csv", index=False)      # real test candidates (demo class)
share(X_tr.sample(5000, random_state=SEED).index).to_csv(DATA / "lime_background.csv", index=False)  # real training sample (LIME)
S["split"] = {k: {"rows": int(len(v)), "eligible_rate": float(v.mean())} for k, v in
              [("train", y_tr), ("val", y_va), ("test", y_te)]}

(RESULTS / "data_summary.json").write_text(json.dumps(S, indent=1, default=str))
print(json.dumps({k: S[k] for k in ["raw_rows", "sat_all_three", "eligible_rate", "validation",
                                    "derived_stream_agreement", "k_anonymity", "split"]}, indent=1))
