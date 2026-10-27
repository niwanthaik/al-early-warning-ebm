# Streamlit front end for the A/L early-warning EBM (trained models + real 2020 sample data)
FROM python:3.12-slim

# libgomp1: OpenMP runtime needed by LightGBM / XGBoost on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app.py al_features.py pipelines.py config.py ./
COPY models/ models/
COPY results/ results/
COPY figures/ figures/
COPY data/sample_upload.csv data/lime_background.csv data/

# run as a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--client.toolbarMode=viewer"]
