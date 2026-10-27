"""
A/L At-Risk Early-Warning System: Streamlit front end for the Explainable Boosting Machine.
Run:  streamlit run app.py      (models/ and results/ are produced by 01–04 scripts)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

import al_features as F

ROOT = Path(__file__).parent
MODELS, RES, FIGS = ROOT / "models", ROOT / "results", ROOT / "figures"
FILES = {"EBM": "ebm", "EBM (no gender)": "ebm_no_gender", "LightGBM": "lightgbm", "XGBoost": "xgboost",
         "Random Forest": "random_forest", "Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree"}
BLUE, RED = "#2f5597", "#c0504d"

st.set_page_config(page_title="A/L Early Warning (EBM)", page_icon="🎓", layout="wide")


@st.cache_resource
def model(name):
    return joblib.load(MODELS / f"{FILES[name]}.pkl")


@st.cache_resource
def lime_al():
    bg = pd.read_csv(ROOT / "data" / "lime_background.csv", dtype=str)   # 5,000 real training candidates
    return F.LimeAL(F.unified_frame(bg), seed=42)


@st.cache_data
def results():
    return (json.loads((RES / "metrics.json").read_text()), json.loads((RES / "explain.json").read_text()),
            json.loads((RES / "tuning.json").read_text()))


def ebm_local(name, X):
    """Exact additive contributions of an EBM (log-odds): one bar per term."""
    pipe = model(name); ebm = pipe.named_steps["model"]
    Xv = pipe.named_steps["prep"].transform(X)
    c = pd.Series(ebm.eval_terms(Xv)[0], index=ebm.term_names_)
    def label(t):
        out = []
        for f in t.split(" & "):
            v = X[f].iloc[0]
            if f.startswith("SUBJ_"):
                out.append(("takes " if v == 1 else "no ") + f[5:].replace("___", " & ").replace("_", " ").title())
            else:
                out.append(f"{f} = {'missing' if pd.isna(v) else v}")
        return " × ".join(out)
    c = c[c.abs() > 1e-6]
    c.index = [label(t) for t in c.index]
    return c, float(np.ravel(ebm.intercept_)[0])


def shap_local(name, X):
    pipe = model(name)
    Xt = pipe.named_steps["prep"].transform(X)
    names = pipe.named_steps["prep"].get_feature_names_out()
    if name == "LightGBM":
        sv = pipe.named_steps["model"].booster_.predict(Xt, pred_contrib=True)[0][:-1]
    else:
        import xgboost as xgb
        sv = pipe.named_steps["model"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0][:-1]
    out = {}
    for n, v in zip(names, sv):
        base = next((c for c in F.CATEGORICAL_FEATURES if n.startswith(c + "_")), None)
        lab = (f"{base} = {X[base].iloc[0]}" if base else
               (("takes " if X[n].iloc[0] else "no ") + n[5:].replace("___", " & ").replace("_", " ").title()) if n.startswith("SUBJ_")
               else f"{n} = {'missing' if pd.isna(X[n].iloc[0]) else X[n].iloc[0]}")
        out[lab] = out.get(lab, 0.0) + float(v)
    return pd.Series(out)


def bar(contrib, title, xlabel, n=10):
    c = contrib.reindex(contrib.abs().sort_values().index).tail(n)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(c.index, c.values, color=np.where(c.values > 0, BLUE, RED)); ax.axvline(0, c="grey", lw=.6)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel, fontsize=8); ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


if not (RES / "explain.json").exists():
    st.error("Run 01_preprocess.py → 04_explain.py first (creates models/ and results/).")
    st.stop()
MET, EXP, TUN = results()
res = pd.DataFrame(MET["metrics"]).set_index("model")
THR = float(MET["early_warning_threshold"])

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🎓 A/L Early Warning")
st.sidebar.caption("Flags GCE A/L candidates at risk of not passing all three subjects (basic university "
                   "eligibility) from information known before results. Model: Explainable Boosting Machine.")
choice = st.sidebar.selectbox("Model", list(FILES), index=0)
r = res.loc[choice]
a, b = st.sidebar.columns(2)
a.metric("ROC-AUC", f"{r['ROC-AUC']:.3f}"); b.metric("Accuracy", f"{r['accuracy']:.3f}")
thr = st.sidebar.slider("Flag as at risk if P(eligible) <", 0.30, 0.90, round(THR, 2), 0.01,
                        help=f"Default {THR:.2f}: catches ~70% of at-risk students (chosen on the validation set).")
st.sidebar.caption(f"Train {MET['n_train']:,} · validation {MET['n_val']:,} · test {MET['n_test']:,} candidates")

st.title("GCE A/L At-Risk Early-Warning System")
t1, t2, t3, t4, t5 = st.tabs(["🧑‍🎓 Check a student", "📁 Check a class (CSV)", "📊 Model comparison",
                              "🔍 What the model learned", "⚖️ Ethics & model card"])

# ------------------------------------------------------------------ 1. single student
with t1:
    st.markdown("Enter what the school knows **before results are released**.")
    c1, c2, c3 = st.columns(3)
    with c1:
        subjects = st.multiselect("Three A/L subjects", sorted(F.SUBJECTS),
                                  default=["COMBINED MATHEMATICS", "PHYSICS", "CHEMISTRY"], max_selections=3)
        syllabus = st.radio("Syllabus", ["new", "old"], horizontal=True)
    with c2:
        absent = st.checkbox("Did not sit the Common General Test (e.g. repeat candidate)")
        cgt = None if absent else st.number_input("Common General Test score (0–100)", 0, 100, 40)
        ge = st.selectbox("General English grade", ["A", "B", "C", "S", "F", "Absent"], index=3)
    with c3:
        by = st.number_input("Birth year", 1985, 2004, 2001)
        bm = st.selectbox("Birth month", list(F.MONTHS))
        gender = st.radio("Gender", ["female", "male"], horizontal=True,
                          help="Not used by the 'EBM (no gender)' model.")
    if len(subjects) != 3:
        st.warning("Select exactly three subjects.")
    else:
        X = F.unified_frame(F.raw_record(subjects, cgt, ge, by, bm, gender, syllabus))
        p = float(model(choice).predict_proba(X)[0, 1])
        k1, k2, k3 = st.columns([1.3, 1, 1])
        k1.metric(f"P(eligible), {choice}", f"{p:.1%}")
        k2.metric("Status", "⚠️ AT RISK" if p < thr else "On track")
        k3.metric("Derived stream", X["stream"].iloc[0].title())
        st.progress(p)
        left, right = st.columns([1, 1.25])
        with left:
            allp = pd.DataFrame({"P(eligible)": {n: float(model(n).predict_proba(X)[0, 1]) for n in FILES}})
            allp["flag"] = np.where(allp["P(eligible)"] < thr, "AT RISK", "on track")
            st.caption("Every model on the same student")
            st.dataframe(allp.style.format({"P(eligible)": "{:.1%}"}), use_container_width=True)
        with right:
            opts = (["EBM (exact)", "LIME"] if choice.startswith("EBM") else
                    ["SHAP", "LIME"] if choice in ("LightGBM", "XGBoost") else ["LIME"])
            method = st.radio("Explanation", opts, horizontal=True,
                              help="EBM: exact additive contributions of the glass-box model. SHAP: TreeSHAP for "
                                   "boosted trees. LIME: model-agnostic local surrogate (works for every model).")
            if method.startswith("EBM"):
                c, b0 = ebm_local(choice, X)
                st.pyplot(bar(c, f"Why? {choice}: baseline {b0:+.2f} + contributions", "← lowers   log-odds of eligibility   raises →"))
            elif method == "SHAP":
                st.pyplot(bar(shap_local(choice, X), f"Why? SHAP ({choice})", "← lowers   log-odds   raises →"))
            else:
                exp = lime_al().explain(model(choice), X, num_features=9, num_samples=3000)
                st.pyplot(bar(pd.Series(dict(exp.as_list(label=1))), f"Why? LIME ({choice})", "← lowers   P(eligible)   raises →"))

# ------------------------------------------------------------------ 2. batch
with t2:
    st.markdown("Upload a CSV in the raw A/L results format. If result columns (`sub1_r`…) are present, "
                "the predictions are scored too.")
    up = st.file_uploader("CSV", type="csv")
    demo = st.checkbox("Use the sample class (500 real 2020 test candidates, with their actual results)", value=up is None)
    if up is not None or demo:
        raw = pd.read_csv(ROOT / "data" / "sample_upload.csv" if demo or up is None else up, dtype=str)
        labelled = {"sub1_r", "sub2_r", "sub3_r"}.issubset(raw.columns) and raw["sub1_r"].isin(list(F.PASS_GRADES) + ["F"]).any()
        if labelled:
            raw = F.filter_sat_exam(raw)
        X = F.unified_frame(raw)
        p = model(choice).predict_proba(X)[:, 1]
        out = raw[["sub1", "sub2", "sub3", "cgt_r", "ge_r", "gender"]].copy()
        out.insert(0, "stream", X["stream"].values)
        out["P(eligible)"] = p.round(3); out["flag"] = np.where(p < thr, "AT RISK", "on track")
        k = st.columns(4)
        k[0].metric("Students", f"{len(out):,}"); k[1].metric("Flagged", f"{(p < thr).mean():.0%}")
        if labelled:
            yv = F.make_target(raw).values
            out["actual"] = np.where(yv == 1, "eligible", "not eligible")
            k[2].metric("At-risk caught", f"{((p < thr) & (yv == 0)).sum() / max((yv == 0).sum(), 1):.0%}")
            k[3].metric("ROC-AUC", f"{roc_auc_score(yv, p):.3f}")
        st.dataframe(out.sort_values("P(eligible)"), use_container_width=True, height=420)
        st.download_button("Download results (CSV)", out.to_csv(index=False), "at_risk_predictions.csv", "text/csv")

# ------------------------------------------------------------------ 3. comparison
with t3:
    st.subheader("Test-set performance (unseen candidates)")
    cols = ["group", "ROC-AUC", "AUC CI low", "AUC CI high", "PR-AUC (at-risk)", "accuracy", "balanced acc",
            "F1 (macro)", "Brier", "overfit gap", "fit time (s)", "size (MB)"]
    tab = res[cols].sort_values("ROC-AUC", ascending=False)
    st.dataframe(tab.style.format({c: "{:.4f}" for c in cols[1:10]}).background_gradient(subset=["ROC-AUC"], cmap="Blues"),
                 use_container_width=True)
    a, b = st.columns(2)
    with a:
        if (FIGS / "eval_auc.png").exists(): st.image(str(FIGS / "eval_auc.png"))
    with b:
        st.markdown("**Is the difference real?** Paired bootstrap ΔAUC (EBM − model)")
        st.dataframe(pd.DataFrame(MET["pairs_vs_ebm"]).set_index("model").round(4), use_container_width=True)
    if (FIGS / "eval_curves.png").exists(): st.image(str(FIGS / "eval_curves.png"))
    with st.expander("Tuned hyperparameters"):
        st.json({k: v.get("final_params", v.get("best_params")) for k, v in TUN.items()})

# ------------------------------------------------------------------ 4. insights
with t4:
    st.subheader("The EBM is a glass box: each feature's learned effect can be read directly")
    for f, cap in [("xai_ebm_shapes.png", "Shape functions"), ("xai_ebm_importance.png", "Term importance"),
                   ("xai_ebm_interactions.png", "Pairwise interactions"), ("xai_pdp.png", "Partial dependence (EBM vs LightGBM)"),
                   ("xai_agreement.png", "EBM vs SHAP (LightGBM) vs LIME")]:
        if (FIGS / f).exists():
            st.image(str(FIGS / f), caption=cap)
    st.markdown("**Does the model agree with what we know about the A/L system?**")
    st.dataframe(pd.DataFrame(EXP["domain_alignment"]), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ 5. ethics
with t5:
    st.markdown(f"""
**Intended use:** help schools and teachers prioritise *supportive* interventions (extra classes, counselling)
for A/L students before results are released. **Not** for admission, selection, streaming or any punitive decision.

**Data:** public, anonymised 2020 A/L results (Kaggle, CC BY-NC-SA 3.0 IGO). Names, NIC and index numbers removed
by the publisher. This project additionally drops **day of birth** (which made 24.6% of records unique together with
gender and subjects) and never redistributes the raw file. The sample class and the LIME background are **real 2020
candidates** from the public dataset (held-out test and training rows), minimised to the fields the model uses.

**Known limitations:** one exam year (2020, COVID-disrupted); no school/district/socio-economic data; AUC ≈ {res.loc['EBM','ROC-AUC']:.2f}
means many students are mis-ranked, so always combine with teacher judgement.

**Fairness:** gender is predictive (female 71.5% vs male 54.7% pass), but using it changes *who gets flagged*.
Choose **EBM (no gender)** in the sidebar to compare. Flag rates by group are reported below.
""")
    fr = pd.DataFrame(EXP["fairness"])
    st.dataframe(fr[fr.group.str.startswith("gender")].round(3), use_container_width=True, hide_index=True)
    if (FIGS / "xai_fairness.png").exists():
        st.image(str(FIGS / "xai_fairness.png"))
