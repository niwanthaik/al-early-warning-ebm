# Early Identification of At-Risk GCE A/L Candidates with an Explainable Boosting Machine

Applied Machine Learning assignment (MSc in Artificial Intelligence, University of Moratuwa), Nimantha Herath

**Problem:** predict, *before results are released*, which Sri Lankan GCE A/L candidates will fail to pass all three subjects
(the basic university-entry requirement), so that schools can offer support early.
**New algorithm:** the **Explainable Boosting Machine (EBM)**, a glass-box GA²M trained by cyclic gradient boosting
(InterpretML). It is compared with the taught models (LR, DT, k-NN, RF, XGBoost) and a LightGBM black-box reference.
**Data:** national GCE A/L 2020 results, 290,285 candidates, validated against Department of Examinations and UGC statistics.

| Deliverable | File |
|---|---|
| Written report | `report/AL_EBM_Report.pdf` |
| Preprocessing script | `01_preprocess.py` |
| Training script | `02_train.py` (+ `pipelines.py`, `al_features.py`, `config.py`) |
| Evaluation script | `03_evaluate.py` |
| Explainability script | `04_explain.py` |
| Executed walkthrough | `walkthrough.ipynb` |
| Front end (bonus) | `app.py` (Streamlit) |
| Demo video script | `DEMO_SCRIPT.md` |
| Deployment (Docker → GHCR → Azure Container Apps) | `Dockerfile`, `.github/workflows/docker-publish.yml`, `DEPLOY_AZURE.md` |

## Dataset
The raw data is **not included in this repository**. It is public but contains quasi-identifiers (birth date, gender and subjects
together single out 24.6% of candidates). Download it yourself:

1. Download `2020_al_data_kaggle_upload_new_old_syllabi.csv` from
   https://www.kaggle.com/datasets/sasikaamarasinghe/student-performance-gce-al-exam-2020-sri-lanka
2. Zip it as `data/al_2020_results.zip`, or edit `RAW_FILE` in `config.py`.

## Run
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate     # or python3.12 -m venv .venv
uv pip install -r requirements.txt
python 01_preprocess.py      # ~30 s: cleaning, validation, ethics checks, EDA, 70/10/20 split
python 02_train.py           # ~40 min: tuning + training of 8 models
python 03_evaluate.py        # ~2 min: test metrics, bootstrap CIs, curves, threshold
python 04_explain.py         # ~2 min: EBM shapes, local, PDP, permutation, SHAP, LIME, fairness
streamlit run app.py         # front end at http://localhost:8501
python report/build_report.py  # regenerates the PDF report from results/ and figures/
```
macOS: if XGBoost or LightGBM can't find `libomp.dylib`, copy the copy bundled with scikit-learn:
`cp .venv/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib "$(python -c 'import sys;print(sys.base_prefix)')/lib/"`

## Deploy (Azure)
**Live app:** https://al-early-warning.redwave-d9ae0d05.eastasia.azurecontainerapps.io

Every push to `main` builds `ghcr.io/niwanthaik/al-early-warning-ebm:latest` with GitHub Actions. See `DEPLOY_AZURE.md` for the Azure Container Apps setup and updates.
The image contains the models plus two minimised real samples: 500 test candidates (sample class) and 5,000 training candidates (LIME background).
