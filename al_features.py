"""
Feature engineering for the Sri Lankan GCE A/L 2020 results dataset (final assignment, v2).

Shared by the notebook (training) and the Streamlit front end (inference).

Target  : eligible = 1 if the candidate passed (A/B/C/S) all three subjects,
          i.e. met the basic UGC requirement to be considered for university entry.
Leakage : the official `stream`, `Zscore`, `district_rank`, `island_rank` are only
          populated for candidates who passed all three subjects -> never used.
          Instead the stream is *re-derived from the subject choices*, which are
          known before results (98.4% agreement with the official stream).

Two feature views are produced from the same raw row:
  * build_features()      -> numeric/binary view (multi-hot subjects) for sklearn,
                             XGBoost and LightGBM
  * build_cat_features()  -> native categorical view for CatBoost (stream, subject
                             combination, sorted subjects as categories)
"""
import numpy as np
import pandas as pd

EXAM_YEAR = 2020
PASS_GRADES = {"A", "B", "C", "S"}
GE_ORDINAL = {"A": 4, "B": 3, "C": 2, "S": 1, "F": 0}
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

# Subjects taken by >= 1,000 candidates get an indicator column; the long tail -> SUBJ_OTHER
SUBJECTS = [
    "CHEMISTRY", "PHYSICS", "SINHALA", "ECONOMICS", "ACCOUNTING", "BUSINESS STUDIES",
    "GEOGRAPHY", "POLITICAL SCIENCE", "BIOLOGY", "COMBINED MATHEMATICS",
    "INFORMATION & COMMUNICATION TECHNOLOGY", "SCIENCE FOR TECHNOLOGY",
    "COMMUNICATION & MEDIA STUDIES", "BUDDHIST CIVILIZATION", "AGRICULTURAL SCIENCE",
    "ENGINEERING TECHNOLOGY", "TAMIL", "HISTORY OF SRI LANKA & INDIA",
    "BIO SYSTEMS TECHNOLOGY", "LOGIC & SCIENTIFIC METHOD", "ISLAM", "HOME ECONOMICS",
    "ART", "DANCING(INDIGENOUS)", "HISTORY OF SRI LANKA & EUROPE", "HINDU CIVILIZATION",
    "ISLAMIC CIVILIZATION", "DRAMA AND THEATRE (SINHALA)", "ARABIC", "ORIENTAL MUSIC",
    "CHRISTIANITY", "DRAMA AND THEATRE (TAMIL)", "ENGLISH", "JAPANESE", "BUDDHISM",
    "CHRISTIAN CIVILIZATION", "CARNATIC MUSIC", "PALI", "FRENCH", "AGRO TECHNOLOGY",
    "HISTORY OF INDIA", "BUSINESS STATISTICS",
]
STREAMS = ["PHYSICAL SCIENCE", "BIOLOGICAL SCIENCE", "COMMERCE", "ARTS",
           "ENGINEERING TECHNOLOGY", "BIOSYSTEMS TECHNOLOGY"]

NUMERIC_FEATURES = ["cgt_score", "ge_grade", "age_at_exam", "birth_month"]
CATEGORICAL_FEATURES = ["gender", "syllabus", "stream"]
FLAG_FEATURES = ["cgt_absent", "ge_absent"]
CAT_VIEW_CATS = ["gender", "syllabus", "stream", "combo", "subj_a", "subj_b", "subj_c"]


def subject_col(s: str) -> str:
    return "SUBJ_" + "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


SUBJECT_FEATURES = [subject_col(s) for s in SUBJECTS] + ["SUBJ_OTHER"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES + SUBJECT_FEATURES
CAT_VIEW_FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + CAT_VIEW_CATS


# ---------------------------------------------------------------- helpers
def filter_sat_exam(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep candidates who actually sat all three subject papers."""
    res = raw[["sub1_r", "sub2_r", "sub3_r"]]
    return raw.loc[~res.isin(["Absent", "Withheld"]).any(axis=1)].copy()


def make_target(raw: pd.DataFrame) -> pd.Series:
    return raw[["sub1_r", "sub2_r", "sub3_r"]].isin(PASS_GRADES).all(axis=1).astype(int)


def derive_stream(raw: pd.DataFrame) -> pd.Series:
    """Re-create the A/L stream from the subject combination (UGC stream rules)."""
    subs = raw[["sub1", "sub2", "sub3"]]
    has = lambda s: subs.eq(s).any(axis=1)
    return pd.Series(np.select(
        [has("ENGINEERING TECHNOLOGY"), has("BIO SYSTEMS TECHNOLOGY"),
         has("COMBINED MATHEMATICS") | has("HIGHER MATHEMATICS"), has("BIOLOGY"),
         has("ACCOUNTING") | has("BUSINESS STUDIES") | has("BUSINESS STATISTICS")],
        ["ENGINEERING TECHNOLOGY", "BIOSYSTEMS TECHNOLOGY", "PHYSICAL SCIENCE",
         "BIOLOGICAL SCIENCE", "COMMERCE"], "ARTS"), index=raw.index)


def _base(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    cgt = pd.to_numeric(raw["cgt_r"], errors="coerce")            # 'Absent' -> NaN
    df["cgt_score"], df["cgt_absent"] = cgt, cgt.isna().astype(int)
    ge = raw["ge_r"].map(GE_ORDINAL)                              # 'Absent'/'Withheld' -> NaN
    df["ge_grade"], df["ge_absent"] = ge, ge.isna().astype(int)
    age = EXAM_YEAR - pd.to_numeric(raw["birth_year"], errors="coerce")            # NIC errors -> NaN
    df["age_at_exam"] = age.where(age.between(15, 70))                              # implausible ages -> NaN
    df["birth_month"] = raw["birth_month"].map(MONTHS)
    df["gender"] = raw["gender"].where(raw["gender"].isin(["male", "female"]))       # 'Major error' -> NaN
    df["syllabus"] = raw["syllabus"].where(raw["syllabus"].isin(["new", "old"]))
    df["stream"] = derive_stream(raw)
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Numeric/multi-hot view. Missing values stay NaN; imputation happens in the pipeline."""
    df = _base(raw)
    subs = raw[["sub1", "sub2", "sub3"]]
    for s in SUBJECTS:
        df[subject_col(s)] = subs.eq(s).any(axis=1).astype(int)
    df["SUBJ_OTHER"] = subs.apply(lambda c: ~c.isin(SUBJECTS)).sum(axis=1)
    return df[ALL_FEATURES]


def build_cat_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Native categorical view for CatBoost: no one-hot, the subject combination is one category."""
    df = _base(raw)
    srt = np.sort(raw[["sub1", "sub2", "sub3"]].fillna("-").values.astype(str), axis=1)
    df["subj_a"], df["subj_b"], df["subj_c"] = srt[:, 0], srt[:, 1], srt[:, 2]
    df["combo"] = pd.Series([" | ".join(r) for r in srt], index=raw.index)
    for c in CAT_VIEW_CATS:
        df[c] = df[c].fillna("unknown").astype(str)
    return df[CAT_VIEW_FEATURES]


def raw_record(subjects, cgt_score, ge_grade, birth_year, birth_month, gender, syllabus) -> pd.DataFrame:
    """One raw-format row from form inputs (used by the front end)."""
    s = list(subjects) + ["-"] * (3 - len(subjects))
    return pd.DataFrame([{
        "sub1": s[0], "sub2": s[1], "sub3": s[2], "sub1_r": "-", "sub2_r": "-", "sub3_r": "-",
        "cgt_r": "Absent" if cgt_score is None else f"{int(cgt_score):03d}",
        "ge_r": ge_grade, "birth_year": str(birth_year), "birth_month": birth_month,
        "gender": gender, "syllabus": syllabus}])


def unified_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Both views side by side; each model pipeline selects the columns it needs."""
    a = build_features(raw)
    b = build_cat_features(raw)[["combo", "subj_a", "subj_b", "subj_c"]]
    return pd.concat([a, b], axis=1)


class SoftVote:
    """Soft-voting ensemble over already-fitted pipelines (average of predicted probabilities).
    Equivalent to sklearn VotingClassifier(voting='soft') but without refitting the members."""
    def __init__(self, members: dict, weights=None):
        self.members = members
        self.weights = weights or {k: 1.0 for k in members}
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        w = np.array([self.weights[k] for k in self.members])
        p = np.stack([m.predict_proba(X)[:, 1] for m in self.members.values()], axis=1) @ w / w.sum()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def cat_view(X: pd.DataFrame) -> pd.DataFrame:
    """Select/convert the unified frame into CatBoost's native categorical view
    (used inside a FunctionTransformer, so it must live in this importable module)."""
    df = X[NUMERIC_FEATURES + FLAG_FEATURES].copy()
    for c in ["gender", "syllabus", "stream"]:
        df[c] = X[c].fillna("unknown").astype(str)
    for c in ["combo", "subj_a", "subj_b", "subj_c"]:
        df[c] = X[c].astype(str)
    return df[CAT_VIEW_FEATURES]


from sklearn.base import BaseEstimator, ClassifierMixin


class CatBoostCat(ClassifierMixin, BaseEstimator):
    """Thin sklearn-compatible wrapper around CatBoostClassifier with the native categorical
    columns fixed (CatBoost's own `cat_features` list breaks sklearn.clone in recent versions)."""
    def __init__(self, iterations=500, learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
                 random_strength=1.0, bagging_temperature=1.0, thread_count=2, random_seed=42):
        self.iterations, self.learning_rate, self.depth, self.l2_leaf_reg = iterations, learning_rate, depth, l2_leaf_reg
        self.random_strength, self.bagging_temperature = random_strength, bagging_temperature
        self.thread_count, self.random_seed = thread_count, random_seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature, thread_count=self.thread_count,
            random_seed=self.random_seed, cat_features=CAT_VIEW_CATS, eval_metric="AUC", verbose=0)
        self.model_.fit(X, y)
        self.classes_ = np.array(self.model_.classes_)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ------------------------------------------------------------------ LIME support
# LIME perturbs features independently. If each of the 43 subject indicators were perturbed on its
# own, LIME would invent impossible students (e.g. taking 9 subjects). So LIME works in a compact,
# human-readable space where the *subject combination is one categorical feature*; perturbing it
# swaps in another real combination (sampled by frequency), from which the stream and subject
# indicators are re-derived exactly as for real data. Every perturbed student is therefore valid.
LIME_NUMERIC = ["cgt_score", "ge_grade", "age_at_exam", "birth_month"]
LIME_CATEG = ["gender", "syllabus", "cgt_absent", "ge_absent", "combo"]
LIME_FEATURES = LIME_NUMERIC + LIME_CATEG
_GE_LETTER = {v: k for k, v in GE_ORDINAL.items()}
_MONTH_NAME = {v: k for k, v in MONTHS.items()}


class LimeAL:
    """LIME tabular explainer for the A/L models, defined on the compact feature space above."""

    def __init__(self, background: pd.DataFrame, top_combos: int = 400, seed: int = 42):
        self.seed = seed
        self.bg = background.copy()
        self.medians = {c: float(self.bg[c].median()) for c in LIME_NUMERIC}
        self.codes = {"gender": ["female", "male"], "syllabus": ["new", "old"],
                      "cgt_absent": [0, 1], "ge_absent": [0, 1],
                      "combo": list(self.bg["combo"].value_counts().index[:top_combos])}
        self.bg = self.bg[self.bg["combo"].isin(self.codes["combo"])]
        self._build()

    def _build(self):
        from lime.lime_tabular import LimeTabularExplainer
        n = len(LIME_NUMERIC)
        self.explainer = LimeTabularExplainer(
            self.encode(self.bg), feature_names=LIME_FEATURES, class_names=["not eligible", "eligible"],
            categorical_features=list(range(n, len(LIME_FEATURES))),
            categorical_names={n + i: [str(k) for k in self.codes[c]] for i, c in enumerate(LIME_CATEG)},
            discretize_continuous=True, mode="classification", random_state=self.seed)

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        M = np.zeros((len(X), len(LIME_FEATURES)))
        for j, c in enumerate(LIME_NUMERIC):
            M[:, j] = X[c].fillna(self.medians[c]).values
        for j, c in enumerate(LIME_CATEG, start=len(LIME_NUMERIC)):
            codes = self.codes[c]
            M[:, j] = X[c].map({k: i for i, k in enumerate(codes)}).fillna(0).values
        return M

    def decode(self, M: np.ndarray) -> pd.DataFrame:
        """LIME-space rows -> raw A/L rows -> unified frame (same feature code as real data)."""
        n = len(LIME_NUMERIC)
        cat = lambda c, j: [self.codes[c][int(round(min(max(v, 0), len(self.codes[c]) - 1)))] for v in M[:, j]]
        combos = [c.split(" | ") for c in cat("combo", n + 4)]
        cgt_abs, ge_abs = cat("cgt_absent", n + 2), cat("ge_absent", n + 3)
        raw = pd.DataFrame({
            "sub1": [c[0] for c in combos], "sub2": [c[1] for c in combos], "sub3": [c[2] for c in combos],
            "sub1_r": "-", "sub2_r": "-", "sub3_r": "-",
            "cgt_r": ["Absent" if a else f"{int(round(np.clip(v, 0, 100))):03d}" for a, v in zip(cgt_abs, M[:, 0])],
            "ge_r": ["Absent" if a else _GE_LETTER[int(round(np.clip(v, 0, 4)))] for a, v in zip(ge_abs, M[:, 1])],
            "birth_year": [str(int(round(EXAM_YEAR - v))) for v in M[:, 2]],
            "birth_month": [_MONTH_NAME[int(round(np.clip(v, 1, 12)))] for v in M[:, 3]],
            "gender": cat("gender", n), "syllabus": cat("syllabus", n + 1)})
        return unified_frame(raw)

    def explain(self, model, x_row: pd.DataFrame, num_features: int = 9, num_samples: int = 3000):
        """LIME explanation of P(eligible) for one student (a unified-frame row)."""
        combo = x_row["combo"].iloc[0]
        if combo not in self.codes["combo"]:          # make the student's own combination representable
            self.codes["combo"].append(combo); self._build()
        x = x_row.copy()
        for c in LIME_NUMERIC:                         # absent scores are represented by the *_absent flags
            x[c] = x[c].fillna(self.medians[c])
        return self.explainer.explain_instance(self.encode(x)[0], lambda M: model.predict_proba(self.decode(M)),
                                               num_features=num_features, num_samples=num_samples, labels=(1,))

    @staticmethod
    def weights(exp) -> pd.Series:
        return pd.Series({LIME_FEATURES[j]: w for j, w in exp.as_map()[1]})


def shap_by_feature(lgbm_pipe, X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM TreeSHAP contributions folded into the LIME feature space:
    one-hot gender/syllabus -> parent; stream + all subject indicators -> 'combo'."""
    Xt = lgbm_pipe.named_steps["prep"].transform(X)
    names = lgbm_pipe.named_steps["prep"].get_feature_names_out()
    sv = lgbm_pipe.named_steps["model"].booster_.predict(Xt, pred_contrib=True)[:, :-1]
    def parent(n):
        if n.startswith("SUBJ_") or n.startswith("stream_"):
            return "combo"
        return next((c for c in ["gender", "syllabus"] if n.startswith(c + "_")), n)
    df = pd.DataFrame(sv, columns=[parent(n) for n in names])
    return df.T.groupby(level=0).sum().T.reindex(columns=LIME_FEATURES, fill_value=0.0)


# ------------------------------------------------------------------ EBM view
EBM_COLS = NUMERIC_FEATURES + ["gender", "syllabus", "stream"] + FLAG_FEATURES + SUBJECT_FEATURES


def ebm_view(X: pd.DataFrame) -> pd.DataFrame:
    """Columns for the Explainable Boosting Machine. Numerics keep NaN (EBM bins missing values
    separately); categoricals are strings so EBM treats them as nominal."""
    d = X[EBM_COLS].copy()
    for c in ["gender", "syllabus", "stream"]:
        d[c] = d[c].fillna("unknown").astype(str)
    return d


def ebm_view_no_gender(X: pd.DataFrame) -> pd.DataFrame:
    return ebm_view(X).drop(columns=["gender"])
