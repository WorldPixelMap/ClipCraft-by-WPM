"""Optional strictly opt-in local AI clients (Ollama & LM Studio) for topic scoring and social copy polish."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234"


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object cleanly, stripping optional markdown fences from local LLMs."""
    txt = (raw_text or "").strip()
    if not txt:
        return None
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
        txt = re.sub(r"\s*```$", "", txt)
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", txt)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def check_ollama_status(timeout: float = 1.2, base_url: str | None = None) -> dict[str, Any]:
    """Detect whether a local Ollama server is running and list installed model names."""
    target = (base_url or OLLAMA_BASE_URL).rstrip("/")
    req = urllib.request.Request(f"{target}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = [
            str(m.get("name"))
            for m in (data.get("models") or [])
            if isinstance(m, dict) and m.get("name")
        ]
        return {"available": True, "models": models}
    except Exception:
        return {"available": False, "models": []}


def check_lmstudio_status(timeout: float = 1.2, base_url: str | None = None) -> dict[str, Any]:
    """Detect whether LM Studio's local OpenAI-compatible server is running and list loaded models."""
    target = (base_url or LMSTUDIO_BASE_URL).rstrip("/")
    req = urllib.request.Request(f"{target}/v1/models", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = [
            str(m.get("id"))
            for m in (data.get("data") or [])
            if isinstance(m, dict) and m.get("id")
        ]
        return {"available": True, "models": models}
    except Exception:
        return {"available": False, "models": []}


def check_local_ai_status(timeout: float = 1.2) -> dict[str, Any]:
    """Probe Ollama and LM Studio concurrently and return unified status + model entries."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_ol = pool.submit(check_ollama_status, timeout)
        f_lm = pool.submit(check_lmstudio_status, timeout)
        ol = f_ol.result()
        lm = f_lm.result()

    entries: list[dict[str, str]] = []
    for m in ol.get("models") or []:
        entries.append({"provider": "ollama", "id": m, "value": f"ollama::{m}", "label": f"Ollama • {m}"})
    for m in lm.get("models") or []:
        entries.append({"provider": "lmstudio", "id": m, "value": f"lmstudio::{m}", "label": f"LM Studio • {m}"})

    return {
        "available": bool(ol.get("available") or lm.get("available")),
        "ollama": ol,
        "lmstudio": lm,
        "models": entries,
    }


def parse_provider_and_model(model: str, provider: str = "auto") -> tuple[str, str]:
    """Resolve explicit (provider, model_id) from prefixed UI value or provider selector."""
    clean = (model or "").strip()
    prov = (provider or "auto").strip().lower()
    if clean.startswith("lmstudio::"):
        return "lmstudio", clean[len("lmstudio::") :].strip()
    if clean.startswith("ollama::"):
        return "ollama", clean[len("ollama::") :].strip()
    if prov in {"ollama", "lmstudio"}:
        return prov, clean
    # Auto-detect when unprefixed model ID is supplied directly
    lm = check_lmstudio_status(timeout=0.6)
    if clean and clean in (lm.get("models") or []):
        return "lmstudio", clean
    ol = check_ollama_status(timeout=0.6)
    if clean and clean in (ol.get("models") or []):
        return "ollama", clean
    if lm.get("available") and not ol.get("available"):
        return "lmstudio", clean
    return "ollama", clean


def format_ai_label(model: str, provider: str = "auto") -> str:
    """Return a user-friendly provider + model label for job status notes."""
    prov, mod = parse_provider_and_model(model, provider)
    pname = "LM Studio" if prov == "lmstudio" else "Ollama"
    return f"{pname} ({mod})" if mod else pname


def _ollama_generate_json(
    model: str,
    prompt: str,
    timeout: float = 20.0,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    """Invoke local Ollama /api/generate with JSON output and strict fallback."""
    if not model.strip():
        return None
    target = (base_url or OLLAMA_BASE_URL).rstrip("/")
    payload = json.dumps(
        {
            "model": model.strip(),
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 280},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{target}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        return _extract_json_object(str(body.get("response") or ""))
    except Exception:
        return None


def _lmstudio_generate_json(
    model: str,
    prompt: str,
    timeout: float = 22.0,
    base_url: str | None = None,
) -> dict[str, Any] | None:
    """Invoke LM Studio /v1/chat/completions OpenAI-compatible local endpoint."""
    if not model.strip():
        return None
    target = (base_url or LMSTUDIO_BASE_URL).rstrip("/")
    payload = json.dumps(
        {
            "model": model.strip(),
            "messages": [
                {
                    "role": "system",
                    "content": "You respond strictly with a single valid JSON object and no extra commentary.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 300,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{target}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        choices = body.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return None
        msg = choices[0].get("message") or {}
        return _extract_json_object(str(msg.get("content") or ""))
    except Exception:
        return None


def _local_ai_generate_json(
    model: str,
    prompt: str,
    provider: str = "auto",
    timeout: float = 20.0,
) -> dict[str, Any] | None:
    prov, model_id = parse_provider_and_model(model, provider)
    if not model_id:
        return None
    if prov == "lmstudio":
        return _lmstudio_generate_json(model_id, prompt, timeout=timeout)
    return _ollama_generate_json(model_id, prompt, timeout=timeout)


def score_moments_with_local_ai(
    model: str,
    candidates: list[dict[str, Any]],
    topic: str = "",
    provider: str = "auto",
) -> dict[int, float]:
    """Ask local Ollama or LM Studio to score up to 8 excerpts by topic relevance or viral hook strength (0..1)."""
    if not model.strip() or not candidates:
        return {}

    trimmed = candidates[:8]
    lines = []
    for idx, c in enumerate(trimmed):
        snippet = str(c.get("preview") or "").strip()[:260]
        if snippet:
            lines.append(f"{idx}: {snippet}")
    if not lines:
        return {}

    clean_topic = topic.strip()
    if clean_topic:
        goal_line = f"Rate how relevant each video excerpt is to the topic: '{clean_topic}'."
    else:
        goal_line = (
            "Rate each video excerpt from 0.0 to 1.0 on viral short-form clip potential "
            "(strong opening hook, curiosity, emotional punch, complete thought)."
        )

    prompt = (
        f"{goal_line}\n"
        f"Return JSON with key 'scores' mapping string index to number 0.0-1.0.\n"
        f"Excerpts:\n" + "\n".join(lines)
    )
    res = _local_ai_generate_json(model, prompt, provider=provider, timeout=18.0)
    if not res or not isinstance(res.get("scores"), dict):
        return {}
    out: dict[int, float] = {}
    for k, v in res["scores"].items():
        try:
            idx = int(k)
            val = max(0.0, min(1.0, float(v)))
            out[idx] = val
        except (TypeError, ValueError):
            continue
    return out


def score_topic_relevance(
    model: str,
    topic: str,
    candidates: list[dict[str, Any]],
    provider: str = "auto",
) -> dict[int, float]:
    """Ask local Ollama or LM Studio to score up to 8 candidate excerpts against a user topic (0..1)."""
    if not topic.strip():
        return {}
    return score_moments_with_local_ai(model=model, candidates=candidates, topic=topic, provider=provider)


def enhance_social_hook_with_ollama(
    model: str,
    clip_text: str,
    video_title: str,
    platform: str,
    language_hint: str = "auto",
    provider: str = "auto",
) -> dict[str, str] | None:
    """Optionally polish on-screen hook banner, short viral title, and caption via local Ollama or LM Studio."""
    snippet = clip_text.strip()[:700]
    if not model.strip() or len(snippet) < 15:
        return None

    prompt = (
        f"You write punchy viral short-video hooks, titles, and 2-sentence captions for {platform} Shorts/Reels.\n"
        f"Language preference: {language_hint}. Source video: {video_title[:120]}.\n"
        f"Clip transcript: {snippet}\n"
        f"Return JSON with keys: 'hook_banner' (<= 44 chars, 3-6 punchy words for top video banner, no quotes), "
        f"'title' (<= 65 chars, no quotes), 'hook_caption' (1-2 engaging sentences)."
    )
    res = _local_ai_generate_json(model, prompt, provider=provider, timeout=18.0)
    if not res:
        return None
    title = str(res.get("title") or "").strip().strip('"')
    hook_caption = str(res.get("hook_caption") or "").strip()
    hook_banner = str(res.get("hook_banner") or "").strip().strip('"')
    if not title and not hook_banner:
        return None
    return {
        "hook_banner": hook_banner[:56],
        "title": title[:70],
        "hook_caption": hook_caption[:320],
    }


enhance_social_hook_with_local_ai = enhance_social_hook_with_ollama
