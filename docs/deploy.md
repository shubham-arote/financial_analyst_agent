# Deploy to GCP Cloud Run

Two stages. **Silver** gets a live public URL fast (single warm instance, keys in Secret
Manager). **Gold** makes it truly stateless/multi-instance (external stores + CI/CD).

> Division of labor: the repo ships a Cloud-Run-ready container + this runbook. The `gcloud`
> steps below run against **your** GCP project — Claude can't touch your cloud account.

---

## Prerequisites

- A GCP project with **billing enabled**, and the `gcloud` CLI authenticated (`gcloud auth login`).
- A Groq key and a Cohere key (the app's two secrets).

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
                       secretmanager.googleapis.com cloudbuild.googleapis.com
```

---

## Silver — live URL on Cloud Run (single instance)

State (parsed docs in SQLite, the in-memory index, conversation checkpoints) lives **on the
instance**. With `--min-instances 1 --max-instances 1` one warm instance holds it — fine for a
demo; not durable across redeploys. (Durable/multi-instance = Gold, below.)

### 1. Artifact Registry repo

```bash
gcloud artifacts repositories create srr --repository-format=docker --location="$REGION"
```

### 2. Secrets (note `printf` — no trailing newline, which breaks API auth)

```bash
printf '%s' "gsk_your_groq_key"   | gcloud secrets create GROQ_API_KEY   --data-file=-
printf '%s' "your_cohere_key"     | gcloud secrets create COHERE_API_KEY --data-file=-
```

### 3. Build the image with Cloud Build (no local Docker needed)

```bash
gcloud builds submit \
  --tag "$REGION-docker.pkg.dev/$PROJECT_ID/srr/agent:latest"
```

### 4. Deploy

```bash
gcloud run deploy srr-agent \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/srr/agent:latest" \
  --region "$REGION" --allow-unauthenticated \
  --port 8080 --memory 1Gi --cpu 1 \
  --timeout 3600 \                       # long timeout so WebSockets stay open
  --min-instances 1 --max-instances 1 \  # one warm instance holds in-memory + SQLite state
  --set-env-vars SRR_PDF_PARSER=auto \
  --set-secrets GROQ_API_KEY=GROQ_API_KEY:latest,COHERE_API_KEY=COHERE_API_KEY:latest
```

### 5. Let the runtime service account read the secrets

```bash
PROJ_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
for S in GROQ_API_KEY COHERE_API_KEY; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:$PROJ_NUM-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```

The deploy prints a `https://srr-agent-….run.app` URL. Open it, **Load sample**, ask a question,
then a bare follow-up ("what about the prior year?") — hybrid retrieval + memory run in the cloud.

Verify: `curl https://srr-agent-….run.app/healthz` → `{"status":"ok","cloud":true,...}`.

---

## Gold — stateless, multi-instance, CI/CD

Each piece swaps one local store for a managed one; all are behind interfaces already in the code.

| Local (Silver) | Managed (Gold) | Code touchpoint |
|---|---|---|
| in-memory hybrid index | **Qdrant Cloud** (named dense+sparse, native RRF + filter lookup) | new `QdrantRetriever` behind `Retriever` (week2 pattern; `table_lookup` → `query_filter`) |
| `SqliteSaver` checkpointer | **Cloud SQL Postgres** | `langgraph-checkpoint-postgres`, swap in `_get_checkpointer` |
| `SqliteDocStore` (`data/docs.db`) | **GCS** blobs + **Firestore/Cloud SQL** metadata | new `DocStore` backend (interface already exists) |
| long PDF parse in-process | **Cloud Run Job / Cloud Tasks** | move `_parse_pdf_bg` to a worker |

Then: drop `--min-instances 1 --max-instances 1`, enable **scale-to-zero**, add the Cloud SQL
connection (`--add-cloudsql-instances`), and split the MCP retrieval sidecar (multi-container
`service.yaml`).

### CI/CD (after `git init` + push to GitHub)

Use **Workload Identity Federation** (no JSON keys) + a GitHub Actions job:
`build → push to Artifact Registry → gcloud run deploy` on push to `main`. (Workflow file lands
in `.github/workflows/deploy.yml` once the repo is on GitHub.)

---

## Notes / gotchas

- **`$PORT`** — Cloud Run sets it (8080); the image binds `${PORT:-8080}`. Hardcoding a port is the
  #1 cause of "container failed to start."
- **WebSockets** work on Cloud Run; keep `--timeout` high (≤3600s).
- **Cold starts** — the lean image (no Docling/torch) keeps these small; `--min-instances 1` avoids
  them entirely at a small cost.
- **Secrets never in the image** — `.env` is in `.dockerignore`; keys come only via `--set-secrets`.
