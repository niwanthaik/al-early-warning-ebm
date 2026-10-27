"""Model pipelines (preprocessing + estimator). Kept in an importable module so pickles load anywhere."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

import al_features as F


def onehot_prep(scale: bool = True) -> ColumnTransformer:
    """For LR / k-NN (scaled) and DT / RF (unscaled): median imputation, one-hot categoricals."""
    num = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num), F.NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), F.CATEGORICAL_FEATURES),
        ("bin", "passthrough", F.FLAG_FEATURES + F.SUBJECT_FEATURES)], verbose_feature_names_out=False)


def boost_prep() -> ColumnTransformer:
    """For XGBoost / LightGBM: no scaling, NaNs kept so the booster learns a missing-value branch."""
    return ColumnTransformer([
        ("num", "passthrough", F.NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), F.CATEGORICAL_FEATURES),
        ("bin", "passthrough", F.FLAG_FEATURES + F.SUBJECT_FEATURES)], verbose_feature_names_out=False)


def ebm_prep(gender: bool = True) -> FunctionTransformer:
    """For the EBM: raw-ish columns, categoricals as strings, NaNs kept (EBM bins missing values itself)."""
    return FunctionTransformer(F.ebm_view if gender else F.ebm_view_no_gender)


def make(prep, model) -> Pipeline:
    return Pipeline([("prep", prep), ("model", model)])
