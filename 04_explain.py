"""
Step 4: explainability and interpretation.

  A. EBM global explanation: term importances and the learned shape functions (glass-box)
  B. EBM pairwise interaction terms
  C. EBM local explanations for three students (exact additive contributions)
  D. Partial dependence (PDP) + ICE, EBM vs LightGBM
  E. Permutation feature importance (model-agnostic), EBM vs LightGBM
  F. SHAP on the black-box LightGBM vs EBM's own contributions (do they learn the same thing?)
  G. LIME on the EBM: agreement with the EBM's exact explanation + stability
  H. Domain-knowledge alignment checks
  I. Fairness: subgroup performance and flag rates, EBM with vs without gender
Outputs: results/explain.json, figures/xai_*.png
"""
import json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import roc_auc_score, recall_score

import al_features as F
from config import PROC, MODELS, RESULTS, FIGS, DATA, SEED, QUICK

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
BLUE, RED = "#2f5597", "#c0504d"
OUT = {}

load = lambda n: pd.read_parquet(PROC / f"{n}.parquet")
tr, te = load("train"), load("test")
if QUICK:
    tr, te = tr.sample(20_000, random_state=SEED), te.sample(8_000, random_state=SEED)
X_tr, X_te, y_te = tr.drop(columns="eligible"), te.drop(columns="eligible"), te["eligible"].values
ebm_pipe, lgbm = joblib.load(MODELS / "ebm.pkl"), joblib.load(MODELS / "lightgbm.pkl")
ebm_ng = joblib.load(MODELS / "ebm_no_gender.pkl")
ebm = ebm_pipe.named_steps["model"]
M = json.loads((RESULTS / "metrics.json").read_text())
THR = M["early_warning_threshold"]
p_ebm = ebm_pipe.predict_proba(X_te)[:, 1]
nice = lambda s: s.replace("SUBJ_", "").replace("___", " & ").replace("_", " ").title() if s.startswith("SUBJ_") else s

# ------------------------------------------------------------------ A. global: term importances + shape functions
names, imp = ebm.term_names_, np.array(ebm.term_importances())
order = np.argsort(-imp)
OUT["term_importance"] = {names[i]: float(imp[i]) for i in order}
top = order[:18][::-1]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh([nice(names[i]) for i in top], imp[top], color=[RED if " & " in names[i] else BLUE for i in top])
ax.set(title="EBM term importance (mean |contribution|, log-odds)", xlabel="importance")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=RED)],
          labels=["main effect", "pairwise interaction"], fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(FIGS / "xai_ebm_importance.png"); plt.close()

G = ebm.explain_global()
idx = {n: i for i, n in enumerate(names)}
def shape(term):
    d = G.data(idx[term])
    return d["names"], np.asarray(d["scores"]), np.asarray(d.get("lower_bounds", d["scores"])), np.asarray(d.get("upper_bounds", d["scores"]))

SHAPES = {}
fig, ax = plt.subplots(2, 4, figsize=(15, 6.4))
for a, term in zip(ax.flat, ["cgt_score", "ge_grade", "age_at_exam", "birth_month", "stream", "gender", "syllabus", "cgt_absent"]):
    if term not in idx:
        a.axis("off"); continue
    nm, sc, lo, hi = shape(term)
    numeric = len(nm) == len(sc) + 1           # EBM returns bin edges for numeric/binary terms, labels for nominal
    if numeric and term in F.NUMERIC_FEATURES:
        edges = np.asarray(nm, dtype=float)
        a.step(edges[:-1], sc, where="post", color=BLUE)
        a.fill_between(edges[:-1], lo, hi, step="post", alpha=.25, color=BLUE)
        SHAPES[term] = {"x": edges[:-1].tolist(), "score": sc.tolist()}
        if term == "age_at_exam": a.set_xlim(16.5, 30)
        a.axhline(0, c="grey", lw=.6)
    else:
        labels = ([f"{term} = {v}" for v in ("no", "yes")] if numeric else [str(n).title() for n in nm])
        a.barh(labels, sc, color=[BLUE if v > 0 else RED for v in sc]); a.tick_params(axis="y", labelsize=7)
        SHAPES[term] = dict(zip((["0", "1"] if numeric else [str(n) for n in nm]), sc.tolist()))
        a.axvline(0, c="grey", lw=.6)
    a.set_title(f"f({term})", fontsize=9)
fig.suptitle("EBM shape functions: what the model learned for each feature (log-odds; band = uncertainty across bags)", fontsize=10)
plt.tight_layout(); plt.savefig(FIGS / "xai_ebm_shapes.png"); plt.close()
subj = {nice(t): float(shape(t)[1][-1] - shape(t)[1][0]) for t in names if t.startswith("SUBJ_") and " & " not in t
        and len(shape(t)[1]) >= 2}
OUT["subject_effects"] = dict(sorted(subj.items(), key=lambda kv: kv[1]))
OUT["shapes"] = SHAPES
# Stream is a deterministic function of the subjects, so EBM's split of credit between the stream term and
# the individual subject terms is not unique. The interpretable quantity is their SUM per subject combination.
top_combos = X_tr["combo"].value_counts().head(15).index
proto = X_tr.loc[X_tr.combo.isin(top_combos)].groupby("combo").head(1).set_index("combo")
sub_terms = [t for t in names if all(f.startswith("SUBJ_") or f == "stream" for f in t.split(" & "))]
ct = pd.DataFrame(ebm.eval_terms(F.ebm_view(proto.reset_index().set_index(proto.index))), columns=names, index=proto.index)
combo_eff = ct[sub_terms].sum(axis=1).sort_values()
cnt = X_tr["combo"].value_counts()
OUT["combo_effects"] = {c: {"effect": float(v), "n_train": int(cnt[c])} for c, v in combo_eff.items()}
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh([c.title() for c in combo_eff.index], combo_eff.values, color=[BLUE if v > 0 else RED for v in combo_eff.values])
ax.axvline(0, c="grey", lw=.6); ax.tick_params(axis="y", labelsize=6.5)
ax.set(title="EBM: learned effect of the 15 most common subject combinations\n(stream + subject terms combined, log-odds)", xlabel="effect on log-odds of eligibility")
plt.tight_layout(); plt.savefig(FIGS / "xai_ebm_subjects.png"); plt.close()

# ------------------------------------------------------------------ B. interactions
inter_all = [names[i] for i in order if " & " in names[i]]
OUT["top_interactions"] = inter_all[:6]
numeric_inter = [t for t in inter_all if any(f in t.split(" & ") for f in ["cgt_score", "age_at_exam", "ge_grade", "birth_month"])]
inter = inter_all[:2] + [t for t in numeric_inter if t not in inter_all[:2]][:1]
fig, ax = plt.subplots(1, len(inter[:3]), figsize=(15, 3.8))
for a, term in zip(np.atleast_1d(ax), inter[:3]):
    d = G.data(idx[term])
    sc = np.asarray(d["scores"])                     # shape (left bins, right bins)
    L, R = list(d["left_names"]), list(d["right_names"])
    def lab(names, n):
        if len(names) == n + 1:                       # numeric: label each bin by its midpoint
            e = np.asarray(names, dtype=float)
            mids = (e[:-1] + e[1:]) / 2
            return [f"{m:.1f}".rstrip("0").rstrip(".") for m in mids]
        return [str(v).title()[:14] for v in names[:n]]
    f1, f2 = term.split(" & ")
    rl = lab(R, sc.shape[1])
    if f2 == "ge_grade" and sc.shape[1] == 5:
        rl = ["F", "S", "C", "B", "A"]
    if len(R) == sc.shape[1] + 1 and sc.shape[1] == 2:
        rl = ["no", "yes"]
    if len(L) == sc.shape[0]:                     # nominal left feature: hide the tiny 'unknown' level
        keep = [i for i, v in enumerate(L) if str(v).lower() != "unknown"]
        sc, L = sc[keep, :], [L[i] for i in keep]
    if len(R) == sc.shape[1]:
        keep = [i for i, v in enumerate(R) if str(v).lower() != "unknown"]
        sc, R = sc[:, keep], [R[i] for i in keep]
        rl = lab(R, sc.shape[1])
    vmax = np.abs(sc).max()
    im = a.imshow(sc.T, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower")
    xl = lab(L, sc.shape[0]); step = max(1, len(xl) // 12)
    if f1 == "ge_grade" and sc.shape[0] == 5:
        xl = ["F", "S", "C", "B", "A"]
    a.set_xticks(range(0, len(xl), step)); a.set_xticklabels(xl[::step], rotation=90, fontsize=6)
    a.set_yticks(range(len(rl))); a.set_yticklabels(rl, fontsize=6)
    a.set(title=f"{nice(f1)} × {nice(f2)}", xlabel=nice(f1)); a.title.set_fontsize(8)
    plt.colorbar(im, ax=a)
plt.tight_layout(); plt.savefig(FIGS / "xai_ebm_interactions.png"); plt.close()

# ------------------------------------------------------------------ C. local explanations
holdout = F.unified_frame(pd.read_csv(DATA / "holdout_candidate.csv", dtype=str))
cases = {"Held-out candidate": holdout,
         "Confidently at risk": X_te.iloc[[int(np.argmin(np.where(X_te.age_at_exam.between(17, 22).values, p_ebm, 9)))]],
         "Borderline (near threshold)": X_te.iloc[[int(np.argmin(np.abs(p_ebm - THR)))]]}
contrib = lambda X: pd.DataFrame(ebm.eval_terms(F.ebm_view(X)), columns=names, index=X.index)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
OUT["local_cases"] = {}
for a, (title, xr) in zip(ax, cases.items()):
    c = contrib(xr).iloc[0]
    c = c[c.abs() > 1e-6]
    c = c.reindex(c.abs().sort_values().index).tail(10)
    p = float(ebm_pipe.predict_proba(xr)[0, 1])
    lab = []
    for t in c.index:
        parts = []
        for f in t.split(" & "):
            v = xr[f].iloc[0]
            parts.append(f"{nice(f)}={'yes' if (f.startswith('SUBJ_') and v == 1) else ('no' if f.startswith('SUBJ_') else v)}")
        lab.append(" & ".join(parts))
    a.barh(lab, c.values, color=[BLUE if v > 0 else RED for v in c.values]); a.axvline(0, c="grey", lw=.6)
    a.set_title(f"{title}: P(eligible) = {p:.2f}", fontsize=9); a.tick_params(axis="y", labelsize=6.5)
    a.set_xlabel("contribution (log-odds)")
    OUT["local_cases"][title] = {"p": p, "intercept": float(np.ravel(ebm.intercept_)[0]),
                                 "top": dict(zip(lab[::-1][:6], c.values[::-1][:6].round(3).tolist())),
                                 "subjects": xr["combo"].iloc[0], "cgt": None if pd.isna(xr.cgt_score.iloc[0]) else float(xr.cgt_score.iloc[0])}
plt.tight_layout(); plt.savefig(FIGS / "xai_ebm_local.png"); plt.close()

# ------------------------------------------------------------------ D. PDP + ICE (EBM vs LightGBM)
Xp = X_te.dropna(subset=["cgt_score", "ge_grade", "age_at_exam", "birth_month"]).sample(min(3000, len(X_te)), random_state=SEED)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
OUT["pdp"] = {}
for a, (feat, grid) in zip(ax, [("cgt_score", np.arange(10, 85, 3)), ("ge_grade", np.arange(0, 5)),
                                ("age_at_exam", np.arange(17, 27)), ("birth_month", np.arange(1, 13))]):
    for mdl, col, lab in [(ebm_pipe, BLUE, "EBM"), (lgbm, "#7f7f7f", "LightGBM")]:
        r = partial_dependence(mdl, Xp, [feat], custom_values={feat: grid}, kind="both", response_method="predict_proba")
        if lab == "EBM" and feat == "cgt_score":
            for ice in r["individual"][0][:60]:
                a.plot(grid, ice, color=BLUE, alpha=.06, lw=.7)
        a.plot(grid, r["average"][0], color=col, lw=2.2, label=lab)
        OUT["pdp"].setdefault(feat, {})[lab] = dict(zip(grid.tolist(), np.round(r["average"][0], 4).tolist()))
    a.set(title=f"PDP: {feat}", xlabel=feat, ylabel="P(eligible)"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / "xai_pdp.png"); plt.close()
r2 = partial_dependence(ebm_pipe, Xp, ["age_at_exam", "birth_month"],
                        custom_values={"age_at_exam": np.arange(17, 23), "birth_month": np.arange(1, 13)}, kind="average",
                        response_method="predict_proba")
fig, ax = plt.subplots(figsize=(6, 3.6))
im = ax.imshow(r2["average"][0].T, aspect="auto", cmap="RdYlBu", origin="lower")
ax.set_xticks(range(6), range(17, 23)); ax.set_yticks(range(12), range(1, 13))
ax.set(title="2-way PDP (EBM): age × birth month", xlabel="age at exam", ylabel="birth month"); plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(FIGS / "xai_pdp_2d.png"); plt.close()
OUT["pdp_2d_age_month"] = np.round(r2["average"][0], 4).tolist()

# ------------------------------------------------------------------ E. permutation importance
Xs = X_te.sample(min(10_000, len(X_te)), random_state=SEED); ys = te.loc[Xs.index, "eligible"].values
cols = F.NUMERIC_FEATURES + ["gender", "syllabus", "stream"] + F.FLAG_FEATURES + F.SUBJECT_FEATURES
perm = {}
for lab, mdl in [("EBM", ebm_pipe), ("LightGBM", lgbm)]:
    r = permutation_importance(mdl, Xs[cols + ["combo", "subj_a", "subj_b", "subj_c"]], ys, scoring="roc_auc",
                               n_repeats=3, random_state=SEED, n_jobs=1)
    perm[lab] = pd.Series(r.importances_mean[:len(cols)], index=cols)
perm = pd.DataFrame(perm).sort_values("EBM", ascending=False)
OUT["permutation"] = perm.head(15).round(5).to_dict()
p15 = perm.head(14).iloc[::-1]; p15.index = [nice(i) for i in p15.index]
ax = p15.plot.barh(figsize=(7.5, 4.8), color=[BLUE, "#7f7f7f"])
ax.set(title="Permutation importance (drop in test ROC-AUC when shuffled)", xlabel="ΔAUC")
plt.tight_layout(); plt.savefig(FIGS / "xai_permutation.png"); plt.close()

# ------------------------------------------------------------------ F/G. SHAP (LightGBM) vs EBM vs LIME, in the same 9-feature space
def fold_term(t):
    out = []
    for f in t.split(" & "):
        out.append("combo" if (f.startswith("SUBJ_") or f == "stream") else f)
    return out
def ebm_folded(X):
    c = contrib(X)
    res = pd.DataFrame(0.0, index=X.index, columns=F.LIME_FEATURES)
    for t in c.columns:
        par = fold_term(t)
        for pfeat in par:                                   # an interaction is shared equally between its parents
            if pfeat in res.columns:
                res[pfeat] += c[t] / len(par)
    return res
S_idx = X_te.sample(200 if not QUICK else 60, random_state=SEED).index
E_f = ebm_folded(X_te.loc[S_idx])
SH = F.shap_by_feature(lgbm, X_te.loc[S_idx])
glob = pd.DataFrame({"EBM |contribution|": E_f.abs().mean(), "LightGBM |SHAP|": SH.abs().mean()})
lime_al = F.LimeAL(X_tr.sample(5000, random_state=SEED), seed=SEED)
L = pd.DataFrame([F.LimeAL.weights(lime_al.explain(ebm_pipe, X_te.loc[[i]], num_features=9, num_samples=2000))
                  .reindex(F.LIME_FEATURES, fill_value=0) for i in S_idx], index=S_idx)
glob["LIME |weight| (on EBM)"] = L.abs().mean()
for c in list(glob.columns):
    glob[c + " rank"] = glob[c].rank(ascending=False).astype(int)
OUT["global_agreement"] = glob.round(4).to_dict()

def agree(A, B):
    rows = []
    for i in A.index:
        a, b = A.loc[i], B.loc[i]
        ta, tb = set(a.abs().nlargest(3).index), set(b.abs().nlargest(3).index)
        both = [f for f in ta | tb if abs(a[f]) > 1e-4 and abs(b[f]) > 1e-4]
        rows.append({"spearman": spearmanr(a.abs(), b.abs()).statistic, "top1": a.abs().idxmax() == b.abs().idxmax(),
                     "top3 overlap": len(ta & tb), "sign agreement": np.mean([np.sign(a[f]) == np.sign(b[f]) for f in both]) if both else np.nan})
    d = pd.DataFrame(rows)
    return {"spearman_median": float(d.spearman.median()), "top1_match": float(d.top1.mean()),
            "top3_overlap_mean": float(d["top3 overlap"].mean()), "sign_agreement": float(d["sign agreement"].mean())}
OUT["agreement"] = {"EBM vs LIME(EBM)": agree(E_f, L), "EBM vs SHAP(LightGBM)": agree(E_f, SH.set_index(S_idx))}

fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
g = glob[["EBM |contribution|", "LightGBM |SHAP|"]].sort_values("EBM |contribution|")
g.plot.barh(ax=ax[0], color=[BLUE, "#7f7f7f"]); ax[0].set(title="Global importance: EBM (exact) vs SHAP on LightGBM", xlabel="mean |log-odds contribution|")
(glob["LIME |weight| (on EBM)"].reindex(g.index)).plot.barh(ax=ax[1], color="#c55a11")
ax[1].set(title="LIME on the EBM (local surrogate, averaged)", xlabel="mean |effect on P(eligible)|")
plt.tight_layout(); plt.savefig(FIGS / "xai_agreement.png"); plt.close()

xb = cases["Borderline (near threshold)"]
runs = pd.DataFrame([F.LimeAL.weights(F.LimeAL(X_tr.sample(5000, random_state=SEED), seed=s).explain(ebm_pipe, xb, num_features=9, num_samples=3000))
                     .reindex(F.LIME_FEATURES, fill_value=0) for s in range(8 if not QUICK else 3)])
t3 = runs.apply(lambda r: tuple(sorted(r.abs().nlargest(3).index)), axis=1).value_counts()
OUT["lime_stability"] = {"runs": len(runs), "same_top3": int(t3.iloc[0]), "top3": list(t3.index[0]),
                         "cv_by_feature": (runs.std() / runs.mean().abs()).round(2).to_dict(),
                         "ebm_top3": list(ebm_folded(xb).iloc[0].abs().nlargest(3).index)}

# ------------------------------------------------------------------ H. domain-knowledge alignment
ds = json.loads((RESULTS / "data_summary.json").read_text())
cg = SHAPES["cgt_score"]; ge = SHAPES["ge_grade"]; age = SHAPES["age_at_exam"]; bm = SHAPES["birth_month"]
st = SHAPES["stream"]
rho_cgt = spearmanr(cg["x"], cg["score"]).statistic
rho_ge = spearmanr(ge["x"], ge["score"]).statistic
age_s = pd.Series(age["score"], index=age["x"])
jan = float(bm["score"][0]); rest = float(np.mean(bm["score"][1:]))
science = np.mean([st.get("BIOLOGICAL SCIENCE", 0), st.get("PHYSICAL SCIENCE", 0)])
arts_com = np.mean([st.get("ARTS", 0), st.get("COMMERCE", 0)])
OUT["domain_alignment"] = [
    {"knowledge": "Higher general aptitude (CGT) means a higher chance of passing", "learned": f"f(CGT) rises with score (Spearman ρ = {rho_cgt:.2f})", "aligned": bool(rho_cgt > 0.8)},
    {"knowledge": "English proficiency supports performance", "learned": f"f(General English) increases from F to A (ρ = {rho_ge:.2f})", "aligned": bool(rho_ge > 0.8)},
    {"knowledge": "Science streams are the hardest to pass all three subjects (DoE statistics)", "learned": f"stream effect: science {science:+.2f} vs Arts/Commerce {arts_com:+.2f}; Physics/Chemistry/Agri Sci terms negative", "aligned": bool(science < arts_com)},
    {"knowledge": "School-entry cut-off is 31 January, so January-born students belong to the year-earlier cohort (often repeaters at the same age)", "learned": f"f(birth month = Jan) = {jan:+.3f} vs mean of other months {rest:+.3f}; age × month interaction", "aligned": bool(jan > rest)},
    {"knowledge": "Repeat candidates are exempt from the CGT and General English, and repeaters pass more often", "learned": "CGT/GE missing (absent) bins carry a positive contribution", "aligned": None},
    {"knowledge": "Repeat candidates around 20–21 do better; much older candidates do worse", "learned": f"f(age) peak near {age_s.idxmax():.0f}; f(age ≥ 23) = {age_s[age_s.index >= 23].mean():+.2f}", "aligned": bool(19 <= age_s.idxmax() <= 22)},
    {"knowledge": "Female candidates outperform males at A/L (also visible in this data: 71.5% vs 54.7%)", "learned": f"f(gender): female {SHAPES['gender'].get('female', 0):+.2f}, male {SHAPES['gender'].get('male', 0):+.2f}", "aligned": bool(SHAPES['gender'].get('female', 0) > SHAPES['gender'].get('male', 0))},
]
# missing-bin check for CGT (EBM stores the missing bin at index 0 of term_scores_)
ci = idx["cgt_score"]; ts = np.asarray(ebm.term_scores_[ci])
OUT["cgt_missing_bin_score"] = float(ts[0]); OUT["domain_alignment"][4]["learned"] = f"f(CGT missing) = {ts[0]:+.2f} log-odds"
OUT["domain_alignment"][4]["aligned"] = bool(ts[0] > 0)

# ------------------------------------------------------------------ I. fairness
p_ng = ebm_ng.predict_proba(X_te)[:, 1]
fair = []
for col in ["gender", "stream", "syllabus"]:
    for k in X_te[col].dropna().unique():
        m = (X_te[col] == k).values
        if m.sum() < 300:
            continue
        for lab, p in [("EBM", p_ebm), ("EBM (no gender)", p_ng)]:
            yr = 1 - y_te[m]
            fair.append({"group": f"{col} = {k}", "model": lab, "n": int(m.sum()), "actual fail %": float(yr.mean()),
                         "flagged %": float((p[m] < THR).mean()), "at-risk recall": float(recall_score(yr, (p[m] < THR).astype(int))),
                         "AUC": float(roc_auc_score(y_te[m], p[m])), "mean predicted": float(p[m].mean()), "actual pass %": float(y_te[m].mean())})
fair = pd.DataFrame(fair)
OUT["fairness"] = fair.round(4).to_dict(orient="records")
g = fair[fair.group.str.startswith("gender")].pivot(index="group", columns="model", values=["flagged %", "actual fail %", "at-risk recall"])
labels = [i.replace("gender = ", "").title() for i in g.index]
xx, w = np.arange(len(g)), 0.36
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].bar(xx - w / 2, g[("flagged %", "EBM")], w, label="flagged: EBM", color=BLUE)
ax[0].bar(xx + w / 2, g[("flagged %", "EBM (no gender)")], w, label="flagged: EBM (no gender)", color="#c55a11")
ax[0].scatter(xx, g[("actual fail %", "EBM")], marker="D", color="black", zorder=3, label="actual fail rate")
ax[0].set_xticks(xx, labels); ax[0].set(title=f"Share flagged at risk (threshold {THR:.2f}) vs actual fail rate", ylabel="share")
ax[0].legend(fontsize=7)
ax[1].bar(xx - w / 2, g[("at-risk recall", "EBM")], w, label="EBM", color=BLUE)
ax[1].bar(xx + w / 2, g[("at-risk recall", "EBM (no gender)")], w, label="EBM (no gender)", color="#c55a11")
ax[1].set_xticks(xx, labels); ax[1].set(title="At-risk students caught (recall = equal opportunity)", ylabel="recall", ylim=(0, 1))
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / "xai_fairness.png"); plt.close()

(RESULTS / "explain.json").write_text(json.dumps(OUT, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps({k: OUT[k] for k in ["agreement", "lime_stability", "domain_alignment", "top_interactions"]}, indent=1, default=str))
print(pd.Series(OUT["term_importance"]).head(12).round(4))
