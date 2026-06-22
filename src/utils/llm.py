import json
import os

import requests

from src.utils.logger import get_logger

log = get_logger(__name__)

# Overridable via OLLAMA_BASE_URL env var.
# Defaults to localhost — works both locally and inside the Docker container
# (since Ollama runs in the same container via entrypoint.sh).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

_LLM_SYSTEM = """\
You are a precise spelling and grammar checker with deep knowledge of computer networking \
and telecommunications terminology. This includes but is not limited to: protocols (TCP/IP, \
UDP, HTTP/S, DNS, BGP, OSPF, MPLS, VXLAN, etc.), network devices (routers, switches, \
firewalls, load balancers, proxies), concepts (subnetting, VLANs, NAT, QoS, SDN, NFV, \
latency, throughput, packet loss), cloud networking (VPC, peering, transit gateways), \
and security (TLS, IPsec, zero-trust, ACLs). Treat established networking acronyms and \
vendor-specific terms (e.g. Cisco, Juniper, Palo Alto) as correct spelling.
When given a text, respond ONLY with a JSON object using this exact schema:
{
  "has_issues": true | false,
  "issues": [
    {
      "type": "spelling" | "grammar" | "style",
      "original": "the wrong word or phrase",
      "suggestion": "corrected word or phrase",
      "explanation": "brief reason"
    }
  ],
  "corrected_text": "the fully corrected sentence"
}
If the text is correct, set has_issues to false and issues to [].
Output ONLY the JSON object — no extra text, no markdown fences."""


def ollama_list_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Return names of models available in the running Ollama server."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        log.debug("Ollama models available: %s", models)
        return models
    except Exception as exc:
        log.warning("Could not reach Ollama at %s: %s", base_url, exc)
        return []


def check_ollama(
    model: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
) -> dict:
    """
    Run a pre-flight health check against Ollama.

    Returns a dict with:
      reachable  : bool   — whether the Ollama server responded
      models     : list   — all pulled model names (empty if unreachable)
      model_ready: bool   — True when `model` is in the pulled list (or model is None)
      hint       : str    — human-readable action to take on failure
    """
    models = ollama_list_models(base_url)
    reachable = len(models) > 0 or _ollama_ping(base_url)

    if not reachable:
        return {
            "reachable":   False,
            "models":      [],
            "model_ready": False,
            "hint":        "Ollama is not running. Start it with:  ollama serve",
        }

    model_ready = True
    hint        = ""
    if model:
        model_ready = any(m == model or m.startswith(model.split(":")[0]) for m in models)
        if not model_ready:
            hint = f"Model not pulled. Run:  ollama pull {model}"

    return {
        "reachable":   True,
        "models":      models,
        "model_ready": model_ready,
        "hint":        hint,
    }


def _ollama_ping(base_url: str) -> bool:
    """Return True if the Ollama server root endpoint responds."""
    try:
        r = requests.get(base_url, timeout=3)
        return r.status_code < 500
    except Exception:
        return False


def ollama_check(
    model: str,
    text: str,
    base_url: str = OLLAMA_BASE_URL,
    ignore: set[str] | None = None,
) -> dict:
    """
    Send text to Ollama for spell/grammar review.
    Returns a parsed result dict or {"error": "..."} on failure.
    """
    system = _LLM_SYSTEM
    if ignore:
        word_list = ", ".join(sorted(ignore))
        system += (
            f"\nAdditionally, treat these words as correctly spelled "
            f"and do NOT flag them: {word_list}."
        )

    payload = {
        "model":  model,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": f'Text: """{text}"""'},
        ],
    }
    log.debug("LLM check: model=%s text_len=%d", model, len(text))
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        result = json.loads(raw)
        log.debug("LLM result: has_issues=%s issues=%d", result.get("has_issues"), len(result.get("issues", [])))
        return result
    except requests.exceptions.ConnectionError:
        log.error("Cannot connect to Ollama at %s", base_url)
        return {"error": f"Cannot connect to Ollama at {base_url}. Is it running? Try: ollama serve"}
    except requests.exceptions.Timeout:
        log.error("Ollama request timed out for model %s", model)
        return {"error": "Ollama request timed out (>120s). Try a smaller/faster model."}
    except requests.exceptions.HTTPError as exc:
        log.error("HTTP %s from Ollama: %s", exc.response.status_code, exc.response.text[:200])
        if exc.response.status_code == 404:
            return {"error": f"Model '{model}' not found. Pull it with: ollama pull {model}"}
        return {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"}
    except json.JSONDecodeError as exc:
        log.error("LLM returned non-JSON: %s", exc)
        return {"error": "LLM returned non-JSON. Try a different model or add a stricter prompt."}
    except Exception as exc:
        log.exception("Unexpected error in ollama_check")
        return {"error": str(exc)}


def _filter_llm_result(result: dict, ignore: set[str]) -> dict:
    """Remove issues whose original text matches an ignored word."""
    if "error" in result or not ignore:
        return result
    filtered = [
        i for i in result.get("issues", [])
        if i.get("original", "").lower() not in ignore
    ]
    if len(filtered) == len(result.get("issues", [])):
        return result
    updated = dict(result)
    updated["issues"]     = filtered
    updated["has_issues"] = bool(filtered)
    return updated
