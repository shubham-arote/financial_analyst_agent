# Document-QA agent — lean, Cloud-Run-ready image (cloud-first: no Docling/torch).
# Parsing uses the text layer (born-digital) and cloud-VLM OCR via Groq (scanned), so the
# image stays small (~300 MB) and cold-starts fast on Cloud Run / scale-to-zero.
# Want the local Docling layout model too? add `docling` to the pip line below — the `auto`
# parser uses it automatically when installed, and falls back to the text layer when not.
FROM python:3.12-slim

WORKDIR /app

# PyMuPDF / Pillow rendering need a couple of system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python samples/make_sample.py && python samples/make_sample_pdf.py

# Cloud Run injects $PORT (usually 8080) and the app MUST bind to it. Secrets (GROQ_API_KEY,
# COHERE_API_KEY) are provided at deploy time via --set-secrets, never baked into the image.
ENV SRR_PDF_PARSER=auto \
    PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/healthz' % os.getenv('PORT','8080'))"
CMD ["sh", "-c", "python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
