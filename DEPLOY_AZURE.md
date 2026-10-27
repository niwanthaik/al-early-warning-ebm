# Deploying the app on Azure Container Apps

The same pattern as the course MLOps workshop (Sections 5–6): **Docker image → GitHub Container Registry → Azure Container Apps**.
GitHub builds the image for you, so you don't need Docker on your Mac. Total time is about 20 minutes.

The image contains the trained models, the results/figures and two minimised real samples from the public 2020 dataset:
500 test candidates for the "Check a class" tab and 5,000 training candidates for LIME (no names, NIC, index numbers or day of birth).

---

## Step 1: Push the project to GitHub (5 min)

1. On github.com, click **New repository**. Name it e.g. `al-early-warning-ebm`. Leave it **empty** (no README or .gitignore).
2. In Terminal:
   ```bash
   cd ~/personal/Studies/MSC/Sem2/Applied\ ML/ML_Assignment_EBM
   git init
   git add .
   git commit -m "A/L early-warning EBM: code, models, report, Docker + Azure deployment"
   git branch -M main
   git remote add origin https://github.com/<your-username>/al-early-warning-ebm.git
   git push -u origin main
   ```
   If git asks for a password, use a **Personal Access Token**: GitHub → Settings → Developer settings → Tokens (classic) → scopes `repo` and `workflow`.
   `.gitignore` already keeps the real data (`data/*.zip`, `data/processed/`, `data/private/`) out of the repository.

## Step 2: Let GitHub build the image (5 min, automatic)

1. Open the repository's **Actions** tab. The workflow **build-and-publish-image** starts on its own.
   It builds the Docker image, pushes it to `ghcr.io/<your-username>/al-early-warning-ebm`, and smoke-tests `/_stcore/health`.
2. When it turns green, go to your GitHub profile → **Packages** → `al-early-warning-ebm` → **Package settings** →
   **Change visibility → Public**. Azure can only pull a public image without extra credentials, as in the workshop.

## Step 3: Create the app on Azure (5–10 min)

1. Go to https://portal.azure.com and open **Cloud Shell** (the `>_` icon at the top). Choose **Bash**.
2. Paste the following, replacing `<your-username>` (lower case):
   ```bash
   IMAGE=ghcr.io/<your-username>/al-early-warning-ebm:latest
   RG=rg-al-early-warning
   APP=al-early-warning
   LOC=southeastasia        # if your subscription blocks this region, try australiaeast, eastasia or centralindia

   az extension add --name containerapp --upgrade -y
   az provider register --namespace Microsoft.App --wait
   az provider register --namespace Microsoft.OperationalInsights --wait
   az group create -n $RG -l $LOC

   az containerapp up -n $APP -g $RG -l $LOC --image $IMAGE --ingress external --target-port 8501

   # 1 vCPU / 2 GB RAM (the models need ~1 GB); scale to zero when nobody is using it, to save credit
   az containerapp update -n $APP -g $RG --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1

   echo "https://$(az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv)"
   ```
3. Open the printed URL. **The first load takes 30–60 seconds** (a cold start from zero replicas, then the models load). After that it's fast.
   Put this URL on the report's title page and in §6, and show it in your demo video.

## Updating the app later
Push to `main`. GitHub rebuilds the image, then tell Azure to use it:
```bash
az containerapp update -n al-early-warning -g rg-al-early-warning \
  --image ghcr.io/<your-username>/al-early-warning-ebm:latest --revision-suffix v$(date +%m%d%H%M)
```

## Cost and clean-up
With `--min-replicas 0` you only pay while someone is using the app, which is cents per day on the free/student credit.
After marking, delete everything with:
```bash
az group delete -n rg-al-early-warning --yes
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `RequestDisallowedByPolicy` / region not allowed | Student subscriptions restrict regions. Change `LOC` and rerun from `az group create`. |
| App shows "Unauthorized" or fails to pull the image | The GHCR package is still private. Make it Public (Step 2.2). |
| Page loads forever on first visit | Cold start. Wait about 60 s and refresh. Set `--min-replicas 1` during the demo or review (uses more credit). |
| Container restarts / out of memory | Make sure `--memory 2.0Gi` was applied (Step 3). |
| Logs | `az containerapp logs show -n al-early-warning -g rg-al-early-warning --follow` |
