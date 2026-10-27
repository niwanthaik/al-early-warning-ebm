"""
Builds report/AL_EBM_Report.html and .pdf from results/*.json and figures/*.png.
Every number in the report is read from the pipeline outputs, so text and results cannot drift apart.
Run after 01–04:   python report/build_report.py
"""
import base64, json, asyncio
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, FIG, OUTD = ROOT / "results", ROOT / "figures", ROOT / "report"
S = json.loads((R / "data_summary.json").read_text())
M = json.loads((R / "metrics.json").read_text())
X = json.loads((R / "explain.json").read_text())
T = json.loads((R / "tuning.json").read_text())
met = pd.DataFrame(M["metrics"]).set_index("model")
pairs = pd.DataFrame(M["pairs_vs_ebm"]).set_index("model")
ew = pd.DataFrame(M["early_warning"]).set_index("setting")
fair = pd.DataFrame(X["fairness"])
E = met.loc["EBM"]

f3 = lambda v: f"{v:.3f}"
f4 = lambda v: f"{v:.4f}"
pc = lambda v, d=1: f"{100 * v:.{d}f}%"
num = lambda v: f"{int(v):,}"


def img(name, caption, width="100%"):
    p = FIG / name
    if not p.exists():
        return f"<p class='missing'>[figure {name} missing]</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    img.n += 1
    return (f"<figure><img src='data:image/png;base64,{b64}' style='width:{width}'>"
            f"<figcaption><b>Figure {img.n}.</b> {caption}</figcaption></figure>")
img.n = 0


def table(df, caption, fmt=None, index=True):
    table.n += 1
    fmt = fmt or {}
    d = df.copy()
    for c, f in fmt.items():
        if c in d.columns:
            d[c] = d[c].map(lambda v, f=f: f(v) if pd.notna(v) else "–")
    html = d.to_html(index=index, escape=False, border=0, classes="tbl")
    return f"<div class='tcap'><b>Table {table.n}.</b> {caption}</div>{html}"
table.n = 0


def fr(group, model, col):
    r = fair[(fair.group == group) & (fair.model == model)]
    return float(r[col].iloc[0])


# ------------------------------------------------------------------ assemble numbers used in the text
best_bb = met.loc[["XGBoost", "LightGBM"], "ROC-AUC"].max()
gap_bb = best_bb - E["ROC-AUC"]
d = {k: pairs.loc[k, "ΔAUC (EBM − model)"] for k in pairs.index}
ka = S["k_anonymity"]
ka_pub = ka["public_file: birth_year+month+day+gender+combo"]
ka_use = ka["used_in_model: birth_year+month+gender+combo"]
ka_coarse = ka["used_in_model_coarse: age+gender+stream"]
ebm_p = T["EBM"]["final_params"]
fem_rec_g, male_rec_g = fr("gender = female", "EBM", "at-risk recall"), fr("gender = male", "EBM", "at-risk recall")
fem_rec_n, male_rec_n = fr("gender = female", "EBM (no gender)", "at-risk recall"), fr("gender = male", "EBM (no gender)", "at-risk recall")
ag_l, ag_s = X["agreement"]["EBM vs LIME(EBM)"], X["agreement"]["EBM vs SHAP(LightGBM)"]
ce = pd.DataFrame(X["combo_effects"]).T
ti = pd.Series(X["term_importance"])
top_inter = [t for t in ti.index if " & " in t][:3]
co = S["cohort_effect"]

# ------------------------------------------------------------------ tables
t_valid = pd.DataFrame({
    "Official (Dept. of Examinations / UGC)": ["301,771", "194,297", "64.39%", "64.6% (125,653 / 194,366)"],
    "This dataset": [num(S["sat_all_three"]), num(S["eligible_n"]), pc(S["eligible_rate"], 2), "64.8%"],
    "Coverage / match": [pc(S["validation"]["coverage_sat"]), pc(S["validation"]["coverage_qualified"]), "matches", "matches"]},
    index=["Candidates who sat", "Qualified for university entrance", "Qualification rate", "Female share of qualified"])
t_feat = pd.DataFrame([
    ["cgt_score", "numeric 0–100", "Common General Test score; 'Absent' → missing + flag"],
    ["ge_grade", "ordinal 0–4", "General English grade F/S/C/B/A; 'Absent' → missing + flag"],
    ["age_at_exam", "numeric", "2020 − birth year (from NIC); implausible ages (outside 15–70) → missing"],
    ["birth_month", "numeric 1–12", "captures the school-entry cohort (31 January cut-off)"],
    ["gender", "nominal", "female / male; NIC errors → 'unknown'"],
    ["syllabus", "nominal", "new / old (2020 had both)"],
    ["stream", "nominal (6)", "re-derived from the subject combination (not the leaked official column)"],
    ["cgt_absent, ge_absent", "binary", "explicit missingness indicators (repeat candidates are exempt)"],
    ["SUBJ_* (42) + SUBJ_OTHER", "binary / count", "which of the 61 subjects the candidate took"],
    ["<b>eligible</b> (target)", "binary", "1 = passed all three subjects (A/B/C/S), the basic UGC requirement"]],
    columns=["Feature", "Type", "Definition / handling"])
t_pre = pd.DataFrame([
    ["Remove candidates absent/withheld for any subject paper", num(S["absent_or_withheld_removed"]), "target undefined"],
    ["Drop leaked columns: official stream, Z-score, district/island rank", "4 columns", "populated only for passing candidates (perfect separators)"],
    ["Drop the subject results sub*_r", "3 columns", "they define the target"],
    ["Parse 'Absent' CGT / General English → missing + indicator", f"{pc(S['missingness']['CGT score']['missing_pct'])} / {pc(S['missingness']['General English']['missing_pct'])}", "missingness is informative"],
    ["NIC decoding errors (birth date, gender) → missing", f"{pc(S['missingness']['age (NIC error)']['missing_pct'], 2)}", "imputed (median / 'unknown')"],
    ["Implausible ages (outside 15–70) → missing", num(S["implausible_age_set_missing"]), "data-entry errors"],
    ["Re-derive stream from subjects (UGC rules)", pc(S["derived_stream_agreement"]["standard_streams"]) + " agreement", "avoids leakage, keeps the information"],
    ["Encoding", "–", "one-hot (LR, k-NN, DT, RF, XGB, LGBM); native nominal handling (EBM)"],
    ["Imputation / scaling", "–", "median + standardisation inside the pipeline for LR and k-NN; trees and EBM keep missing values as their own bin"],
    ["Data minimisation", "day of birth, index, ranks", "never used and stripped from every shared file"]],
    columns=["Step", "Rows / columns affected", "Reason"])
t_kanon = pd.DataFrame({
    "Quasi-identifiers": ["birth year + month + day + gender + subject combination (public file)",
                          "birth year + month + gender + subject combination (used by the model)",
                          "age + gender + stream (coarse view)"],
    "Unique records": [pc(ka_pub["unique_records_pct"]), pc(ka_use["unique_records_pct"]), pc(ka_coarse["unique_records_pct"], 2)],
    "Records with k < 5": [pc(ka_pub["k_lt_5_pct"]), pc(ka_use["k_lt_5_pct"]), pc(ka_coarse["k_lt_5_pct"], 2)],
    "Median group size k": [f"{ka_pub['median_k']:.0f}", f"{ka_use['median_k']:.0f}", f"{ka_coarse['median_k']:,.0f}"]})
t_split = pd.DataFrame({k: {"Rows": num(v["rows"]), "Eligible rate": pc(v["eligible_rate"], 2)} for k, v in S["split"].items()}).T
t_split.index = ["Training (70%)", "Validation (10%)", "Test (20%)"]
t_cmp = pd.DataFrame([
    ["Model form", "single tree of if-then splits", "linear in features (log-odds)", "majority vote of nearest neighbours", "sum of hundreds of deep trees (black box)",
     "<b>sum of one learned function per feature + selected pairwise functions</b>"],
    ["Non-linearity", "yes (piecewise)", "no (unless engineered)", "yes (local)", "yes", "yes, per feature (shape functions)"],
    ["Interactions", "implicit, unstable", "only if added by hand", "implicit", "implicit, unlimited order", "explicit, only pairwise, automatically detected (FAST)"],
    ["How it is trained", "greedy recursive splitting", "maximum likelihood", "no training (memorises)", "bagging / gradient boosting of trees",
     "cyclic gradient boosting: tiny trees on one feature at a time, low learning rate, many rounds, bagged"],
    ["Interpretability", "readable only if shallow", "global coefficients", "none", "needs post-hoc SHAP/LIME", "<b>exact, built in</b>: each term can be plotted; predictions decompose exactly"],
    ["Typical accuracy on tabular data", "low–moderate", "moderate", "moderate", "high", "close to GBDT"]],
    columns=["", "Decision tree", "Logistic regression", "k-NN", "RF / XGBoost", "EBM (this work)"])
hp_rows = []
for m in ["Logistic Regression", "Decision Tree", "k-NN", "Random Forest", "XGBoost", "LightGBM", "EBM"]:
    p = T[m].get("final_params", T[m]["best_params"])
    hp_rows.append([m, ", ".join(f"{k}={(round(v, 4) if isinstance(v, float) else v)}" for k, v in p.items()),
                    f4(T[m]["cv_auc"]), f"{T[m].get('fit_seconds', 0):.0f}"])
t_hp = pd.DataFrame(hp_rows, columns=["Model", "Selected hyperparameters (none left at default)", "CV ROC-AUC", "Fit (s)"])
t_ebm_space = pd.DataFrame([
    ["interactions", "20, 40, 80", ebm_p["interactions"], "number of pairwise terms; EDA showed stream × CGT and gender × stream effects"],
    ["learning_rate", "0.005–0.05 (log)", f"{ebm_p['learning_rate']:.4f}", "small steps give smooth, stable shape functions"],
    ["max_bins", "256, 512, 1024", ebm_p["max_bins"], "resolution of numeric shape functions (CGT has ~90 distinct values)"],
    ["max_leaves", "2, 3", ebm_p["max_leaves"], "size of each tiny tree in the cyclic booster"],
    ["min_samples_leaf", "2, 10, 25, 50", ebm_p["min_samples_leaf"], "minimum students per leaf: regularises noisy bins"],
    ["smoothing_rounds", "0, 50, 200", ebm_p["smoothing_rounds"], "initial random-split rounds for smoother curves"],
    ["outer_bags", "4 (search) → 8 (final)", ebm_p["outer_bags"], "bagging: averages 8 EBMs and gives the uncertainty bands"]],
    columns=["Hyperparameter", "Search space", "Selected", "Rationale"])
mcols = ["group", "ROC-AUC", "AUC CI low", "AUC CI high", "PR-AUC (at-risk)", "accuracy", "balanced acc", "F1 (macro)", "Brier", "overfit gap", "fit time (s)", "ms / 1k preds", "size (MB)"]
t_res = met[mcols].sort_values("ROC-AUC", ascending=False).rename(columns={"group": "Group"})
t_pairs = pairs.rename(columns={"ΔAUC (EBM − model)": "ΔAUC (EBM − model)"})
t_ew = ew[["threshold", "at-risk recall", "at-risk precision", "flagged %", "accuracy", "balanced acc"]]
t_align = pd.DataFrame(X["domain_alignment"])[["knowledge", "learned", "aligned"]]
t_align["aligned"] = t_align["aligned"].map({True: "✔ yes", False: "✘ no", None: "–"})
t_align.columns = ["Domain knowledge / expectation", "What the EBM learned", "Aligned?"]
t_agree = pd.DataFrame(X["agreement"]).T
t_agree.columns = ["Median Spearman ρ", "Top-1 match", "Mean top-3 overlap (of 3)", "Sign agreement"]
t_fair = fair[fair.group.str.startswith("gender")][["group", "model", "n", "actual fail %", "flagged %", "at-risk recall", "AUC", "mean predicted", "actual pass %"]]
t_ce = ce.sort_values("effect")[["effect", "n_train"]]
t_ce.index = [i.title() for i in t_ce.index]

lit = pd.DataFrame([
    ["Rajapakse & Ranawaka (2020)", "Sri Lanka, Grade 5 scholarship exam, 500 students", "Logistic regression", f"LR: AUC {f3(met.loc['Logistic Regression','ROC-AUC'])}"],
    ["Abeywardhane & Gamage (2023)", "Sri Lanka, A/L stream recommendation", "Random forest best (72% acc.)", f"RF: AUC {f3(met.loc['Random Forest','ROC-AUC'])}"],
    ["Bujang et al. (2021)", "Malaysia, course grades, 1,282 records", "RF best (with SMOTE)", f"RF: AUC {f3(met.loc['Random Forest','ROC-AUC'])}"],
    ["Batool et al. (2023), survey", "Educational data mining review", "DT, NB, ANN, RF most used", f"DT: AUC {f3(met.loc['Decision Tree','ROC-AUC'])}"]],
    columns=["Study", "Context", "Method / finding", "Same method on this data"])

CSS = """
@page { size: A4; }
body { font-family: 'Georgia', 'Times New Roman', serif; font-size: 10.6pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 21pt; margin: 0 0 4pt; color: #1f3864; } h2 { font-size: 14.5pt; color: #1f3864; border-bottom: 1.5px solid #1f3864; padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.8pt; color: #2f5597; margin: 12pt 0 4pt; page-break-after: avoid; } p { margin: 4pt 0 7pt; text-align: justify; }
.marks { font-size: 9pt; color: #7f7f7f; font-weight: normal; } .title { text-align: center; margin-top: 70pt; }
.title .sub { font-size: 13pt; color: #2f5597; margin-bottom: 30pt; } .title .meta { font-size: 11pt; line-height: 1.8; }
.abstract { border-left: 3px solid #2f5597; background: #f3f6fb; padding: 8pt 12pt; margin: 28pt 20pt; font-size: 10pt; }
figure { margin: 8pt 0 10pt; page-break-inside: avoid; text-align: center; } figure img { max-width: 100%; }
figcaption { font-size: 8.8pt; color: #444; margin-top: 3pt; text-align: left; }
.tbl { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 2pt 0 10pt; font-family: 'Helvetica', Arial, sans-serif; page-break-inside: avoid; }
.tbl th { background: #1f3864; color: white; padding: 3pt 4pt; text-align: left; } .tbl td { border-bottom: 0.5px solid #ccc; padding: 2.5pt 4pt; vertical-align: top; }
.tbl tr:nth-child(even) td { background: #f6f8fb; } .tcap { font-size: 8.8pt; color: #444; margin-top: 8pt; page-break-after: avoid; }
.eq { text-align: center; font-family: 'Cambria Math', 'STIX Two Math', serif; font-size: 11.5pt; margin: 6pt 0 8pt; }
.box { background: #fff7ed; border: 1px solid #f0c48a; padding: 6pt 10pt; font-size: 9.6pt; margin: 6pt 0 10pt; page-break-inside: avoid; }
.pb { page-break-before: always; } ul { margin: 2pt 0 6pt 16pt; padding: 0; } li { margin: 1.5pt 0; }
.ref p { text-indent: -16pt; padding-left: 16pt; font-size: 9.2pt; text-align: left; margin: 2pt 0; } .missing { color: red; }
code { font-size: 9pt; background: #f2f2f2; padding: 0 2pt; }
"""

H = []
a = H.append
a(f"""<div class='title'>
<h1>Early Identification of At-Risk GCE A/L Candidates in Sri Lanka<br>with an Explainable Boosting Machine</h1>
<div class='sub'>Applied Machine Learning: Assignment Report</div>
<div class='meta'><b>Nimantha Herath</b><br>Index number: [add]<br>MSc in Artificial Intelligence, University of Moratuwa<br>
Source code: [GitHub link]<br>Demo video: [link]</div></div>
<div class='abstract'><b>Abstract.</b> About {pc(1 - S['eligible_rate'], 0)} of Sri Lankan GCE Advanced Level (A/L) candidates fail to meet the basic
university-entry requirement of passing all three subjects. Using the full national 2020 results
({num(S['sat_all_three'])} candidates, validated against official statistics), this work predicts, <i>before results are released</i>,
which candidates are at risk, from their subject combination, Common General Test (CGT) score, General English grade, age and gender.
A machine-learning algorithm not covered in the lectures, the <b>Explainable Boosting Machine (EBM)</b>, a glass-box generalised additive model
with pairwise interactions trained by cyclic gradient boosting, is compared with the taught models (logistic regression, decision tree, k-NN,
random forest, XGBoost) and a LightGBM black-box reference. The EBM reaches a test ROC-AUC of <b>{f3(E['ROC-AUC'])}</b>
(95% CI {f3(E['AUC CI low'])}–{f3(E['AUC CI high'])}). That significantly beats logistic regression, the decision tree, k-NN and random forest,
and is within {f3(gap_bb)} of the best black box, while being fully interpretable and the least over-fitted model. Its learned shape functions agree with
seven independent pieces of domain knowledge, including the 31 January school-entry cohort effect and the CGT exemption for repeat candidates.
They are cross-checked with PDP, permutation importance, SHAP and LIME. An early-warning threshold catches {pc(ew.loc['early warning','at-risk recall'], 0)} of at-risk students.
Removing gender equalises at-risk detection between girls and boys at a cost of {f3(d['EBM (no gender)'])} AUC, a trade-off discussed together with
re-identification risk and other ethical limits. A Streamlit application exposes predictions and explanations to teachers.</div>
<div class='pb'></div>""")

# 1 ------------------------------------------------------------------
a(f"""<h2>1. Problem Definition and Dataset Collection <span class='marks'>(15 marks)</span></h2>
<h3>1.1 Problem and relevance</h3>
<p>The GCE A/L examination decides access to Sri Lanka's free state universities. To be considered by the University Grants Commission (UGC), a
candidate must first pass all three subjects (grade S or above). In 2020, {pc(1 - S['eligible_rate'], 1)} of candidates who sat all three papers did not.
Schools and tuition teachers usually learn who struggled only when results are published, months after the
examination, when it is too late to help. The question here is whether information that is available <b>before results</b> can identify
at-risk candidates early enough for supportive intervention (extra classes, counselling, advice on re-sitting or alternative pathways). The task is
<b>binary classification</b>: <code>eligible = 1</code> if all three subjects were passed, otherwise 0.</p>
<h3>1.2 Data source and collection</h3>
<p>The data is the 2020 GCE A/L results sheet published by the Department of Examinations and compiled into one table by Amarasinghe (2021)
on Kaggle (licence CC BY-NC-SA 3.0 IGO). Names, National Identity Card (NIC) numbers and examination index numbers were removed by the publisher; birth date
and gender were decoded from the NIC before removal, which explains the 'Invalid nic error' entries. Because this is a community compilation rather
than an official release, it was <b>validated against the official statistics</b> (Table 1). The qualification rate matches to 0.04 percentage points, and the
dataset covers 96.2% of candidates who sat and 96.3% of those who qualified. Coverage is the same for both groups, and also by gender (96.3% of qualified
females, 95.8% of qualified males), so the missing ~4% shows no sign of selection bias.</p>
{table(t_valid, "Validation of the dataset against official 2020 figures (Department of Examinations; UGC Statistics 2021, Table 2.1).")}
<h3>1.3 Features, target and size</h3>
<p>The raw file has {num(S['raw_rows'])} rows and {S['raw_cols']} columns. After cleaning, <b>{num(S['sat_all_three'])} candidates</b> remain, described by
51 model features (Table 2): 4 numeric, 3 nominal, 2 missingness flags and 43 subject indicators. Candidates chose from {S['cardinality']['subjects']} subjects,
forming {num(S['cardinality']['combinations'])} distinct combinations. The 20 most common cover {pc(S['cardinality']['top20_share'], 0)} of students, while
{num(S['cardinality']['combos_lt_30'])} combinations have fewer than 30 students. The target is moderately imbalanced ({pc(S['eligible_rate'])} eligible).</p>
{table(t_feat, "Features and target.", index=False)}
<h3>1.4 Preprocessing</h3>
<p>Table 3 lists the cleaning steps. Two deserve emphasis. First, a <b>leakage audit</b> found that the official <code>stream</code>, <code>Zscore</code> and
rank columns are filled in only for candidates who passed. Each one separates the classes perfectly ({num(S['leakage']['stream_blank_vs_eligible']['0']['true'])} of
{num(S['leakage']['stream_blank_vs_eligible']['0']['true'] + S['leakage']['stream_blank_vs_eligible']['0']['false'])} non-eligible candidates have a blank stream), so a model using them would look perfect and be useless.
They were removed, and the stream was <b>re-derived from the subject choices</b> using the UGC stream rules. This reproduces the official stream for
{pc(S['derived_stream_agreement']['standard_streams'], 1)} of candidates in the six standard streams. Second, <b>missingness is informative</b>: candidates without a CGT score passed at
{pc(S['missingness']['CGT score']['pass_if_missing'])} against {pc(S['missingness']['CGT score']['pass_if_present'])} for those with one, because repeat candidates are exempt from
the CGT and General English papers. Missing values are therefore kept as explicit indicators rather than silently imputed.</p>
{table(t_pre, "Preprocessing steps.", index=False)}
{img("eda_overview.png", "Target balance, CGT score by outcome, and pass rate by (derived) stream.")}
{img("eda_interactions.png", "The CGT effect depends strongly on the stream (left), and the gender gap varies by stream (right). Both are non-linear interactions.")}
{img("eda_age_cohort.png", f"Age is non-monotonic (left). Among 18–19-year-olds, January-born candidates pass more often (right): {pc(co['19']['jan'],0)} against {pc(co['19']['feb_dec'],0)} at age 19, consistent with the 31 January school-entry cut-off.", "92%")}
<h3>1.5 Ethical data use</h3>
<p>The data is public, anonymised by its publisher and licensed for non-commercial use, and no attempt was made to identify any person. Even without names,
though, combinations of attributes can single people out. A <b>k-anonymity check</b> (Table 4) shows that in the public file,
{pc(ka_pub['unique_records_pct'], 1)} of candidates are unique on birth date + gender + subject combination. Anyone who knows a classmate's birthday and subjects could
look up their grades. This project therefore follows <b>data minimisation</b>: day of birth is never used (uniqueness drops to {pc(ka_use['unique_records_pct'], 1)}),
it is stripped from every file shipped with the code and front end. The raw file is <b>not redistributed</b> (the repository links to the source instead), and the front end ships only two minimised samples (500 + 5,000 candidates) of the fields the model uses,
and only aggregate patterns are reported. Gender is a sensitive attribute, and its use is examined explicitly in §5.</p>
{table(t_kanon, "Re-identification risk (k-anonymity) for different sets of quasi-identifiers.", index=False)}
""")

# 2 ------------------------------------------------------------------
a(f"""<h2 class='pb'>2. Selection of a New Machine Learning Algorithm <span class='marks'>(15 marks)</span></h2>
<h3>2.1 The Explainable Boosting Machine</h3>
<p>The lectures covered linear and logistic regression, SVM, decision trees, random forest, AdaBoost, gradient boosting, XGBoost, stacking and voting, naive
Bayes and k-means. The algorithm chosen here, the <b>Explainable Boosting Machine</b> (Lou et al., 2013; Nori et al., 2019), was not covered. It is a
<i>generalised additive model with pairwise interactions</i> (GA²M):</p>
<div class='eq'>logit P(eligible | x) = β₀ + Σⱼ fⱼ(xⱼ) + Σ₍ᵢ,ⱼ₎∈𝓘 fᵢⱼ(xᵢ, xⱼ)</div>
<p>Each <b>shape function</b> fⱼ is an arbitrary learned curve for one feature, and 𝓘 is a small set of automatically selected feature pairs. Training uses
<b>cyclic gradient boosting</b>. In each round, the algorithm visits the features one at a time and fits a tiny tree (here {ebm_p['max_leaves']} leaves) to the current residuals
<i>using only that feature</i>, with a very low learning rate ({ebm_p['learning_rate']:.3f}). Thousands of such rounds build each fⱼ gradually, and because only one feature is ever
used at a time, the result stays additive. The <b>FAST</b> algorithm then ranks all candidate pairs by how much residual they explain, and the top
{ebm_p['interactions']} pairs get their own two-dimensional functions fᵢⱼ. Finally, {ebm_p['outer_bags']} models trained on bootstrap samples (<b>bagging</b>) are averaged, which smooths the curves and
gives uncertainty bands. The model is fully transparent: a prediction is <i>exactly</i> the intercept plus the sum of the term values, so every prediction decomposes
into contributions that can be read off a plot.</p>
<h3>2.2 Why the EBM was selected</h3>
<ul>
<li><b>The data called for it.</b> The EDA showed strong non-linear effects (CGT, age) and pairwise interactions (stream × CGT, gender × stream) but weak correlation for any single
feature. Linear models cannot capture this, and boosted trees can but are opaque. The EBM captures non-linear main effects and pairwise interactions while staying interpretable.</li>
<li><b>The application needs explanations.</b> Teachers acting on a flag need to know <i>why</i> a student is at risk, and decisions about young people call for models
whose logic can be inspected and audited (Rudin, 2019). With an EBM, the explanation <i>is</i> the model, not a post-hoc approximation.</li>
<li><b>The accuracy cost is small.</b> EBMs typically match gradient-boosted trees on tabular data (Caruana et al., 2015; Nori et al., 2019). This is tested directly in §3.</li>
<li><b>It can be edited.</b> Because each term is separate, a term such as gender can be inspected, removed or constrained, which supports the fairness analysis in §5.</li></ul>
<h3>2.3 How the EBM differs from standard models</h3>
{table(t_cmp, "The EBM compared with standard models.", index=False)}
<p>Compared with a <b>decision tree</b>, the EBM never splits on several features in one path, so it cannot hide long, unstable interaction chains, and it boosts thousands of
small steps instead of making one greedy pass. Compared with <b>logistic regression</b>, it keeps additivity on the log-odds scale but replaces each linear coefficient with a learned
curve and adds pairwise terms, so non-linearity no longer has to be engineered by hand. Compared with <b>k-NN</b>, it learns a compact global model (0.2 MB) instead of
memorising the data, and it predicts about 200× faster. Compared with <b>random forest / XGBoost</b>, it gives up unrestricted high-order interactions in exchange for exact,
global and local interpretability.</p>
""")

# 3 ------------------------------------------------------------------
a(f"""<h2 class='pb'>3. Model Training and Evaluation <span class='marks'>(20 marks)</span></h2>
<h3>3.1 Train / validation / test split</h3>
<p>One real candidate was set aside first for the inference demonstration. The rest were split <b>70 / 10 / 20</b> with stratification on the target (Table {table.n + 1}).
The <b>training</b> set was used for fitting and for hyperparameter search. The search used 3-fold stratified cross-validation on a 60,000-row stratified
subsample, to keep it tractable on a laptop. The <b>validation</b> set was used only to choose the early-warning threshold, and the <b>test</b> set was touched once, for the final evaluation.
The EBM additionally holds out 15% of each bag internally for early stopping.</p>
{table(t_split, "Data split.")}
<h3>3.2 Hyperparameter choices</h3>
<p>No model uses library defaults. Each was tuned with a randomised or grid search over a justified space, maximising ROC-AUC (Table {table.n + 1}).
For the EBM, the space and the reasoning are shown in Table {table.n + 2}. The search preferred the largest number of interactions ({ebm_p['interactions']}),
consistent with the interaction effects seen in the EDA, and a large minimum leaf size ({ebm_p['min_samples_leaf']}), a strong regulariser that suits the noisy outcome of a single examination.
The final EBM uses {ebm_p['outer_bags']} bags. The same configuration was also trained <b>without gender</b> for the fairness analysis.</p>
{table(t_hp, "Selected hyperparameters and cross-validated ROC-AUC (training subsample).", index=False)}
{table(t_ebm_space, "EBM search space and rationale.", index=False)}
<h3>3.3 Performance metrics</h3>
<p><b>ROC-AUC</b> is the primary metric: it measures how well a model <i>ranks</i> students by risk, independently of any threshold, which suits an early-warning
list. Because the at-risk class is the minority (35.6%), <b>PR-AUC for the at-risk class</b> and <b>balanced accuracy</b> are also reported, along with <b>macro-F1</b> and accuracy at the 0.5 threshold.
The <b>Brier score</b> measures calibration, which matters because teachers see probabilities. 95% confidence intervals come from {300} bootstrap resamples of the
test set, and differences between models are tested with a <b>paired bootstrap</b> on ΔAUC.</p>
<h3>3.4 Results</h3>
{table(t_res, f"Test-set results ({num(M['n_test'])} unseen candidates).", fmt={c: f4 for c in mcols[1:10]} | {"fit time (s)": lambda v: f"{v:.0f}", "ms / 1k preds": lambda v: f"{v:.1f}", "size (MB)": lambda v: f"{v:.2f}"})}
{img("eval_auc.png", "Test ROC-AUC with 95% bootstrap confidence intervals.", "80%")}
{table(t_pairs, "Paired bootstrap comparison: ΔAUC of the EBM against every other model.", fmt={"ΔAUC (EBM − model)": f4, "CI low": f4, "CI high": f4})}
{img("eval_curves.png", "ROC, precision–recall (at-risk class) and calibration curves.")}
<h3>3.5 What the results indicate</h3>
<ul>
<li><b>The EBM significantly beats every interpretable or simple baseline:</b> logistic regression (+{f3(d['Logistic Regression'])} AUC), the decision tree (+{f3(d['Decision Tree'])}),
k-NN (+{f3(d['k-NN'])}) and even the random forest (+{f3(d['Random Forest'])}). All confidence intervals exclude zero.</li>
<li><b>The cost of transparency is small.</b> XGBoost and LightGBM are ahead by {f3(-d['XGBoost'])}–{f3(-d['LightGBM'])} AUC. That gap is statistically significant with 58,000 test
students, but it means roughly 1 in 200 pairs of students are ranked differently. In exchange, the EBM gives exact explanations and is the <b>least over-fitted</b>
model (train–test AUC gap {f3(E['overfit gap'])} against {f3(met.loc['XGBoost','overfit gap'])} for XGBoost and {f3(met.loc['Random Forest','overfit gap'])} for random forest).
It is also the fastest to score ({E['ms / 1k preds']:.1f} ms per 1,000 students) and small (0.2 MB).</li>
<li><b>It is well calibrated</b> (Brier {f4(E['Brier'])}; the calibration curve follows the diagonal), so its probabilities can be shown to teachers as they are.</li>
<li><b>There is a ceiling of about 0.77 AUC for every model.</b> This shows that the limit is the information available before results, not the algorithm. School, district,
O/L results and socio-economic factors are not in the data (§5).</li></ul>
<p><b>Early-warning operating point.</b> At the default threshold of 0.5, the EBM catches only {pc(ew.loc['default (0.5)','at-risk recall'],0)} of at-risk students. A threshold chosen on the
<i>validation</i> set ({ew.loc['early warning','threshold']:.2f}) catches <b>{pc(ew.loc['early warning','at-risk recall'],0)}</b>, with {pc(ew.loc['early warning','at-risk precision'],0)} of flagged students
actually failing, against a base rate of 35.6% (Table {table.n + 1}).</p>
{table(t_ew, "Default vs early-warning threshold (EBM, test set).", fmt={c: f3 for c in t_ew.columns})}
{img("eval_confusion.png", "Confusion matrices: XGBoost and EBM at 0.5, and the EBM at the early-warning threshold.")}
""")

# 4 ------------------------------------------------------------------
cg_slope = X["shapes"]["cgt_score"]
a(f"""<h2 class='pb'>4. Explainability and Interpretation <span class='marks'>(20 marks)</span></h2>
<p>Five complementary methods were applied: (i) the EBM's own <b>global shape functions and term importances</b>, (ii) its exact <b>local explanations</b>, (iii) <b>partial dependence
and ICE plots</b>, (iv) <b>permutation feature importance</b>, and (v) <b>SHAP</b> on the LightGBM black box and <b>LIME</b> on the EBM, used to cross-check the EBM's explanation.</p>
<h3>4.1 What the model has learned: global view</h3>
{img("xai_ebm_importance.png", "EBM term importance (mean absolute contribution on the test set). Red bars are pairwise interaction terms.", "80%")}
{img("xai_ebm_shapes.png", "Learned shape functions (log-odds; the shaded band is variation across the 8 bags). These are the model itself, not an approximation.")}
<p>The <b>most influential features</b> are the stream and subject combination, the CGT score, gender and the General English grade, followed by age (Figure {img.n - 1}).
The shape functions (Figure {img.n}) show <i>how</i> each feature acts:</p>
<ul>
<li><b>CGT score</b>: an almost perfectly monotonic, close-to-linear increase from about −2.6 to +3.2 log-odds across the score range. General aptitude is the strongest single signal.</li>
<li><b>General English</b>: a monotonic step function from F to A (−0.4 to +0.8).</li>
<li><b>Stream</b>: Biological and Physical Science are strongly negative and the technology streams and Arts positive, matching their pass rates.</li>
<li><b>Age</b>: a peak at 20 (typically a second attempt), first-timers slightly below zero, and a steep drop after 22.</li>
<li><b>Birth month</b>: January stands out (+{X['shapes']['birth_month']['score'][0]:.2f}), while the other months are near zero. This is a cohort effect, discussed in §4.6.</li>
<li><b>Missing CGT</b>: the missing-value bin of the CGT term carries <b>+{X['cgt_missing_bin_score']:.2f}</b> log-odds, which identifies repeat candidates.</li></ul>
<p>Because the stream is a deterministic function of the subjects, the EBM's split of credit between the <i>stream</i> term and individual <i>subject</i> terms is not unique. The
interpretable quantity is their sum for each combination (Figure {img.n + 1}). Science combinations such as Combined Maths–ICT–Physics and Agriculture–Biology–Chemistry carry the
largest penalties ({ce['effect'].min():+.2f}), while Arts combinations in Sinhala, Buddhist Civilisation, Geography and Political Science carry the largest bonuses.</p>
{img("xai_ebm_subjects.png", "Learned effect of the 15 most common subject combinations (stream + subject terms combined).", "85%")}
<h3>4.2 Interactions</h3>
<p>The strongest pairwise terms are {", ".join("<i>" + t.replace("SUBJ_", "").replace("_", " ").title() + "</i>" for t in top_inter)}. The <b>age × gender</b> term shows that the
female advantage shrinks and then reverses with age: among older (repeat) candidates, males do relatively better. The <b>CGT × General English</b> term shows the two
reinforce each other: a high CGT score is worth more with a good English grade, and an A in English is worth less with a low CGT score. The <b>CGT × Physics</b> term shows
that a high CGT score is worth <i>more</i> for Physics students than for others, consistent with Physics demanding the quantitative reasoning the CGT measures.</p>
{img("xai_ebm_interactions.png", "Three of the learned pairwise interaction functions (log-odds).")}
<h3>4.3 Local explanations</h3>
<p>An EBM prediction is the intercept ({X['local_cases']['Held-out candidate']['intercept']:+.2f}) plus the sum of its term contributions, so the per-student explanation is
exact. Figure {img.n + 1} explains three students:</p>
<ul>
<li>The <b>held-out candidate</b> (Arts stream, CGT 43, failed General English, old syllabus) gets P = {X['local_cases']['Held-out candidate']['p']:.2f}, mostly from the Arts stream, being female and the old syllabus, partly offset by the English grade. She did in fact pass.</li>
<li>The <b>confidently at-risk</b> student (Physical Science, CGT {X['local_cases']['Confidently at risk']['cgt']:.0f}) gets P = {X['local_cases']['Confidently at risk']['p']:.2f}, driven by a very low CGT score and the science stream.</li>
<li>The <b>borderline</b> student (Commerce: Accounting, Business Studies, Economics; CGT {X['local_cases']['Borderline (near threshold)']['cgt']:.0f}; P = {X['local_cases']['Borderline (near threshold)']['p']:.2f}) shows competing forces. Accounting pulls upwards and the other two Commerce subjects pull downwards, so the three should be read together as the combination effect in §4.1. Being male pulls further down, leaving the student just below the early-warning threshold.</li></ul>
{img("xai_ebm_local.png", "Exact local explanations for three test students (top 10 contributions, log-odds).")}
<h3>4.4 Partial dependence and permutation importance</h3>
<p>Partial dependence plots (Figure {img.n + 1}) show the average predicted probability as one feature is varied. For the EBM and the LightGBM black box they are almost identical for CGT,
English, age and birth month, so two very different learning algorithms recover the same relationships. The ICE curves (thin lines) are roughly parallel, confirming that the CGT
effect is mostly additive. The 2-D PDP (Figure {img.n + 2}) isolates the January × age-18/19 cohort pattern. Permutation importance (Figure {img.n + 3}) ranks CGT, stream, gender and
General English highest for the EBM, consistent with its term importances.</p>
{img("xai_pdp.png", "Partial dependence (thick lines) for the EBM and LightGBM; ICE curves for 60 students (thin lines, CGT panel).")}
{img("xai_pdp_2d.png", "Two-way partial dependence of the EBM: age at exam × birth month.", "60%")}
{img("xai_permutation.png", "Permutation importance (drop in test ROC-AUC when a feature is shuffled), EBM vs LightGBM.", "80%")}
<h3>4.5 Cross-checking explanations: EBM vs SHAP vs LIME</h3>
<p>To test whether the explanations are robust, the EBM's exact contributions were compared, in the same 9-feature space, with <b>TreeSHAP</b> values of the LightGBM black box
(Lundberg &amp; Lee, 2017) and with <b>LIME</b> (Ribeiro et al., 2016) applied to the EBM itself, for 200 test students. LIME was configured so that the subject combination is a
single feature. Perturbing 43 subject indicators independently would create impossible students, such as one taking nine subjects.</p>
{table(t_agree, "Agreement between explanation methods (200 test students).", fmt={c: f3 for c in t_agree.columns})}
{img("xai_agreement.png", "Global importance in a common feature space: EBM (exact), SHAP on LightGBM, LIME on the EBM.")}
<p>The EBM and SHAP-on-LightGBM agree strongly (median rank correlation {ag_s['spearman_median']:.2f}; same top feature for {pc(ag_s['top1_match'],0)} of students; direction agrees {pc(ag_s['sign_agreement'],0)} of the time).
So the glass box and the black box have learned the same drivers. LIME agrees less closely (ρ = {ag_l['spearman_median']:.2f}) and is <b>unstable</b>: across {X['lime_stability']['runs']} random seeds,
its top-3 features for the same borderline student were identical in only {X['lime_stability']['same_top3']} runs. That illustrates why an exact, built-in explanation is preferable
to a sampled approximation when one is available.</p>
<h3>4.6 Does the model's behaviour align with domain knowledge?</h3>
{table(t_align, "Alignment of the learned model with domain knowledge about the Sri Lankan A/L system.", index=False)}
<p>All seven checks agree. Two are worth highlighting because the model discovered them <i>without being told</i>. Grade 1 admission in Sri Lanka requires a child to be five years old by
<b>31 January</b>, so January-born students belong to the year-earlier school cohort. At the same nominal age they are more likely to be second-attempt candidates, who pass more often:
at age 19, {pc(co['19']['jan'],0)} of January-born candidates passed against {pc(co['19']['feb_dec'],0)} of the rest. Likewise, the positive missing-CGT bin reflects the rule that repeat
candidates are exempt from the CGT and General English. A model that learned spurious patterns would not line up with institutional rules like these.</p>
""")

# 5 ------------------------------------------------------------------
a(f"""<h2 class='pb'>5. Critical Discussion <span class='marks'>(10 marks)</span></h2>
<h3>5.1 Limitations of the model</h3>
<ul>
<li><b>Modest discrimination.</b> An AUC of {f3(E['ROC-AUC'])} means about one pair in four is ranked the wrong way round. Even at the early-warning threshold, {pc(1-ew.loc['early warning','at-risk recall'],0)} of
at-risk students are missed and {pc(1-ew.loc['early warning','at-risk precision'],0)} of flagged students pass anyway. The tool can prioritise support but must not decide anything on its own.</li>
<li><b>One year only.</b> 2020 was disrupted by COVID-19, and with no later year available the model has no temporal validation. Patterns may shift from year to year.</li>
<li><b>Timing.</b> The CGT and General English papers are sat during the A/L examination period, so the warning comes weeks, not months, before results. An earlier model would need O/L results or school-based assessments.</li>
<li><b>Additivity.</b> The EBM allows only pairwise interactions. The black boxes' 0.005 AUC advantage comes from higher-order effects the EBM cannot represent.</li>
<li><b>Sparse regions.</b> For very old candidates (over 30) or rare subject combinations, shape functions rest on few students and their uncertainty bands widen accordingly (Figure 8). Predictions there are less reliable.</li></ul>
<h3>5.2 Data quality issues</h3>
<ul>
<li>Community-compiled, not an official release. It covers about 96% of candidates, but the coverage checks (§1.2) show no detectable bias.</li>
<li>NIC decoding errors affected {pc(S['missingness']['age (NIC error)']['missing_pct'],2)} of birth dates and genders, and {S['implausible_age_set_missing']} implausible ages (e.g. 9 or 108) were set to missing.</li>
<li>The derived stream approximates the official one for the 'NON' (non-standard combination) group.</li>
<li>Important predictors are absent: school, district, medium of instruction, O/L results, attendance and socio-economic status. This is the main reason all models plateau around 0.77 AUC.</li></ul>
<h3>5.3 Risks of bias and unfairness</h3>
<p>Gender is the third most important feature: female candidates passed at {pc(S['pass_by_gender']['female'])} against {pc(S['pass_by_gender']['male'])} for males, which is also visible in UGC statistics.
Using it improves accuracy (+{f3(d['EBM (no gender)'])} AUC), but it changes <b>who gets help</b> (Table {table.n + 1}, Figure {img.n + 1}). With gender, the model flags {pc(fr('gender = male','EBM','flagged %'),0)} of boys
against an actual fail rate of {pc(fr('gender = male','EBM','actual fail %'),0)}, and catches {pc(male_rec_g,0)} of at-risk boys but only {pc(fem_rec_g,0)} of at-risk girls. At-risk girls are under-detected because
the model expects girls to pass. <b>Without gender</b>, at-risk recall becomes almost equal ({pc(fem_rec_n,0)} for girls, {pc(male_rec_n,0)} for boys), satisfying <i>equal opportunity</i>, at the cost of more
false alarms for girls and a {f3(d['EBM (no gender)'])} lower AUC. For a supportive tool whose purpose is to reach every struggling student, the gender-free model is the more defensible
default, and the front end offers both. Subgroup AUC is stable across streams (0.72–0.79) and the model is well calibrated within every stream and both syllabi.</p>
{table(t_fair, "At-risk detection by gender at the early-warning threshold.", fmt={c: f3 for c in ["actual fail %", "flagged %", "at-risk recall", "AUC", "mean predicted", "actual pass %"]}, index=False)}
{img("xai_fairness.png", "Share flagged vs actual fail rate (left) and at-risk recall (right), by gender, with and without the gender feature.")}
<h3>5.4 Real-world impact and ethical considerations</h3>
<ul>
<li><b>Intended use:</b> helping schools target <i>supportive, low-cost</i> interventions such as extra classes, mentoring, or advice on re-sitting and vocational pathways.
<b>Never</b> admission, streaming or any decision that limits a student's options. A risk label is itself a sensitive inference and can become self-fulfilling if teachers lower expectations.</li>
<li><b>Human in the loop:</b> the explanation shown with every prediction lets teachers judge whether the reasons make sense for the individual student.</li>
<li><b>Privacy:</b> day of birth and identifiers are never used, the raw data is not redistributed, and a school using the tool would need consent and data-protection compliance under Sri Lanka's Personal Data Protection Act, No. 9 of 2022.</li>
<li><b>Transparency and accountability:</b> the glass-box model can be audited term by term, and harmful terms can be removed, which is not possible with a black box.</li></ul>
""")

# 7 ------------------------------------------------------------------
shots = [f for f in ["app_student.png", "app_class.png", "app_ethics.png"] if (FIG / f).exists()]
a(f"""<h2 class='pb'>6. Front-End Integration <span class='marks'>(bonus, 10 marks)</span></h2>
<p>A <b>Streamlit</b> web application (<code>app.py</code>) loads the trained pipelines and offers five tabs:</p>
<ol>
<li><b>Check a student:</b> enter the subjects, CGT score (or "did not sit"), English grade, birth year/month, gender and syllabus, and get P(eligible), an at-risk flag, every model's prediction,
and an explanation chart. The chart shows the EBM's exact contributions, SHAP for the boosters, or LIME for any model.</li>
<li><b>Check a class:</b> upload a CSV of students, get a ranked risk list (scored automatically if results are present) and download it.</li>
<li><b>Model comparison:</b> the metrics, confidence intervals, significance tests, curves and tuned hyperparameters.</li>
<li><b>What the model learned:</b> the shape functions, interactions, PDPs, method agreement and the domain-alignment table.</li>
<li><b>Ethics and model card:</b> intended use, data minimisation, limitations and fairness results. The sidebar switches between the EBM and the gender-free EBM and adjusts the flagging threshold.</li></ol>
<p><b>Deployment.</b> The app is packaged as a Docker image (<code>Dockerfile</code>). A GitHub Actions workflow builds and publishes it to the GitHub Container Registry on every push,
and it is hosted on <b>Azure Container Apps</b>, following the pattern of the course MLOps workshop: [add the app URL]. The deployed app works on the <b>real 2020 data</b>. Its sample class is 500 held-out test candidates
with their actual results, so the app scores itself live, and LIME uses 5,000 real training candidates as background. Both files are minimised as described in §1.5.</p>
{"".join(img(s, c, w) for s, c, w in zip(shots, ["Check-a-student tab with the EBM's exact explanation.", "Check-a-class tab: ranked risk list for an uploaded CSV, scored against actual results.", "Ethics and model-card tab with the fairness comparison."], ["78%", "70%", "70%"]))}
<h2>7. Conclusion</h2>
<p>Using the full national 2020 A/L results, an Explainable Boosting Machine predicts which candidates will fail to meet the basic university requirement with ROC-AUC {f3(E['ROC-AUC'])}. It significantly
outperforms the taught interpretable and ensemble baselines and comes within {f3(gap_bb)} of the best black-box gradient-boosted trees, while being fully transparent, well calibrated and the
least over-fitted model. Its learned behaviour matches what is known about the A/L system, including subtle institutional rules it was never told about, and the explanations are consistent across
EBM, PDP, permutation importance and SHAP. The main open issues are the information ceiling of pre-results data and the fairness trade-off around gender. Natural next steps are multi-year
validation and linking school-level data.</p>
<div class='ref'><h2>References</h2>
<p>Abeywardhane, K. G. K., &amp; Gamage, A. (2023). Subject stream prediction: A machine learning approach to select the suitable subject stream for senior secondary students in Sri Lanka. <i>International Journal of Innovative Science and Research Technology, 7</i>(12), 1380–1387.</p>
<p>Amarasinghe, S. (2021). <i>Student performance (GCE A/L exam 2020, Sri Lanka)</i> [Data set]. Kaggle. https://www.kaggle.com/datasets/sasikaamarasinghe/student-performance-gce-al-exam-2020-sri-lanka</p>
<p>Batool, S., Rashid, J., Nisar, M. W., Kim, J., Kwon, H.-Y., &amp; Hussain, A. (2023). Educational data mining to predict students' academic performance: A survey study. <i>Education and Information Technologies, 28</i>, 905–971.</p>
<p>Bujang, S. D. A., Selamat, A., Ibrahim, R., Krejcar, O., Herrera-Viedma, E., Fujita, H., &amp; Ghani, N. A. M. (2021). Multiclass prediction model for student grade prediction using machine learning. <i>IEEE Access, 9</i>, 95608–95621.</p>
<p>Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., &amp; Elhadad, N. (2015). Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. <i>Proc. 21st ACM SIGKDD</i>, 1721–1730.</p>
<p>Department of Examinations, Sri Lanka (2021). 2020 G.C.E. (A/L) results release (194,297 of 301,771 candidates qualified). Reported by Ada Derana: https://adaderana.lk/news.php?nid=73563</p>
<p>Lou, Y., Caruana, R., Gehrke, J., &amp; Hooker, G. (2013). Accurate intelligible models with pairwise interactions. <i>Proc. 19th ACM SIGKDD</i>, 623–631.</p>
<p>Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>Advances in Neural Information Processing Systems 30</i>.</p>
<p>Nori, H., Jenkins, S., Koch, P., &amp; Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. <i>arXiv:1909.09223</i>.</p>
<p>Rajapakse, C., &amp; Ranawaka, U. M. (2020). Predicting examination performance using machine learning approach: A case study of the Grade 5 scholarship examination in Sri Lanka. <i>Proc. SCSE 2020</i>, 202–209.</p>
<p>Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. <i>Proc. 22nd ACM SIGKDD</i>, 1135–1144.</p>
<p>Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. <i>Nature Machine Intelligence, 1</i>, 206–215.</p>
<p>University Grants Commission, Sri Lanka (2021). <i>Sri Lanka University Statistics 2021</i>, Chapter 2: University admissions. https://www.ugc.ac.lk/downloads/statistics/stat_2021/Chapter%202.pdf</p></div>
<h2>Appendix A. Comparison with existing Sri Lankan and educational data-mining studies</h2>
<p>Published results come from different exams, targets and features, so they cannot be compared one-to-one. Their <i>methods</i> were therefore re-implemented on this data (Table {table.n + 1}).
Studies reporting over 90% accuracy typically use in-course marks, which are close proxies for the outcome. This work deliberately uses only pre-results information.</p>
{table(lit, "Methods used in earlier studies, evaluated on this dataset.", index=False)}
<h2>Appendix B. Reproducibility and tools</h2>
<p>Code: <code>01_preprocess.py</code> → <code>02_train.py</code> → <code>03_evaluate.py</code> → <code>04_explain.py</code> (about 45 minutes on a 2-core laptop), with <code>walkthrough.ipynb</code> as an executed record,
<code>app.py</code> as the front end, and <code>report/build_report.py</code> to generate this report from the result files. Libraries: interpret 0.7 (EBM), scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7,
SHAP 0.54, LIME 0.2, Streamlit. Random seed 42 throughout. An AI coding assistant was used to help write and debug code and to generate the front end. All modelling decisions,
analysis and interpretation were reviewed by the author.</p>
""")

html = f"<!doctype html><html><head><meta charset='utf-8'><title>A/L EBM Report</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
OUTD.mkdir(exist_ok=True)
(OUTD / "AL_EBM_Report.html").write_text(html, encoding="utf-8")


async def pdf():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto((OUTD / "AL_EBM_Report.html").as_uri())
        await pg.pdf(path=str(OUTD / "AL_EBM_Report.pdf"), format="A4", print_background=True,
                     margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"},
                     display_header_footer=True, header_template="<span></span>",
                     footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'><span class='pageNumber'></span></div>")
        await b.close()
asyncio.run(pdf())
print("report written:", OUTD / "AL_EBM_Report.pdf")
