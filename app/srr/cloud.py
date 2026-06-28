"""
cloud.py — free, OpenAI-compatible cloud providers (Groq / OpenRouter / custom).

Why cloud, not local: a 2 GB Maxwell GPU can't host a VLM, and CPU OCR adds a heavy
torch dependency. Free OpenAI-compatible endpoints give a real *small VLM* for the
Recognition stage (vision) and a real LLM for the agent loop (text), with zero local
model weights. Configure with ONE env var:

    GROQ_API_KEY=...          -> https://api.groq.com/openai/v1     (fast, free tier)
    OPENROUTER_API_KEY=...    -> https://openrouter.ai/api/v1       (free :free models)
    # or a fully custom OpenAI-compatible endpoint:
    VLM_BASE_URL=... VLM_API_KEY=... [VISION_MODEL=...] [TEXT_MODEL=...]

Keys can live in a project-root .env file (loaded here). If no key is present,
has_cloud() is False and callers fall back (stub recognizer / extractive answers).
"""

from __future__ import annotations

import base64
import functools
import io

from PIL import Image

from ..config import settings


def resolve_provider() -> dict | None:
    """Return {name, base_url, api_key, vision_model, text_model} or None."""
    vision = settings.vision_model
    text = settings.text_model

    if settings.vlm_base_url and settings.vlm_api_key:
        generic = settings.vlm_model
        return {"name": "custom", "base_url": settings.vlm_base_url, "api_key": settings.vlm_api_key,
                "vision_model": vision or generic, "text_model": text or generic}

    if settings.groq_api_key:
        return {"name": "Groq", "base_url": "https://api.groq.com/openai/v1",
                "api_key": settings.groq_api_key,
                "vision_model": vision or "meta-llama/llama-4-scout-17b-16e-instruct",
                "text_model": text or "llama-3.3-70b-versatile"}

    if settings.openrouter_api_key:
        return {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
                "api_key": settings.openrouter_api_key,
                "vision_model": vision or "meta-llama/llama-3.2-11b-vision-instruct:free",
                "text_model": text or "meta-llama/llama-3.3-70b-instruct:free"}

    return None


def has_cloud() -> bool:
    return resolve_provider() is not None


def provider_label() -> str:
    p = resolve_provider()
    return f"{p['name']} · {p['vision_model']}" if p else "none (offline)"


@functools.lru_cache(maxsize=1)
def get_client():
    p = resolve_provider()
    if not p:
        return None
    from openai import OpenAI
    headers = None
    if p["name"] == "OpenRouter":
        headers = {"HTTP-Referer": "http://localhost:8000", "X-Title": "SRR-OCR-MVP"}
    return OpenAI(base_url=p["base_url"], api_key=p["api_key"],
                  timeout=90.0, max_retries=2, default_headers=headers)


def _data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat_vision(image: Image.Image, prompt: str, max_tokens: int = 1024) -> str:
    p, client = resolve_provider(), get_client()
    if not client:
        return "[no-cloud]"
    try:
        r = client.chat.completions.create(
            model=p["vision_model"], temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
            ]}],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:  # never let one block kill the stream
        return f"[vlm-error: {type(e).__name__}: {e}]"


def chat_text(prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
    p, client = resolve_provider(), get_client()
    if not client:
        return "[no-cloud]"
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    try:
        r = client.chat.completions.create(
            model=p["text_model"], temperature=0.0, max_tokens=max_tokens, messages=msgs)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[llm-error: {type(e).__name__}: {e}]"
