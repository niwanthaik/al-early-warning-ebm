"""Shared settings for the pipeline scripts."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA, PROC, MODELS, RESULTS, FIGS = (ROOT / "data", ROOT / "data" / "processed", ROOT / "models",
                                    ROOT / "results", ROOT / "figures")
RAW_FILE = DATA / "al_2020_results.zip"
SEED = 42
N_JOBS = 2
SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}
TUNE_ROWS = 60_000            # stratified subsample of train used for hyperparameter search (3-fold CV)
KNN_ROWS = 80_000             # k-NN reference set (memory/latency bound)
BOOT = 300                    # bootstrap resamples for confidence intervals
OFFICIAL = {"sat": 301_771, "qualified": 194_297, "rate": 0.6439}   # Department of Examinations, 2020
import os
QUICK = os.environ.get("QUICK") == "1"     # smoke-test mode: tiny samples
if QUICK:
    TUNE_ROWS, KNN_ROWS, BOOT = 6_000, 8_000, 30
    MODELS, RESULTS, FIGS = ROOT / "models_quick", ROOT / "results_quick", ROOT / "figures_quick"
for d in (PROC, MODELS, RESULTS, FIGS):
    d.mkdir(parents=True, exist_ok=True)
