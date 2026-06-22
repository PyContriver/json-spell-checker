import json
import os

import requests

from src.utils.logger import get_logger

log = get_logger(__name__)

# Overridable via OLLAMA_BASE_URL env var.
# Defaults to localhost — works both locally and inside the Docker container
# (since Ollama runs in the same container via entrypoint.sh).
OLLAMA_BASE_URL     = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT      = int(os.environ.get("OLLAMA_TIMEOUT", "240"))   # seconds per request
OLLAMA_MAX_RETRIES  = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))  # retry on timeout
OLLAMA_NUM_CTX      = int(os.environ.get("OLLAMA_NUM_CTX",     "1024")) # context window — smaller = faster
OLLAMA_NUM_THREAD   = int(os.environ.get("OLLAMA_NUM_THREAD",  "0"))    # 0 = Ollama picks

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
    timeout: int | None = None,
    max_retries: int | None = None,
    num_ctx: int | None = None,
    num_thread: int | None = None,
) -> dict:
    """
    Send text to Ollama for spell/grammar review with retry on timeout.
    Returns a parsed result dict or {"error": "..."} on failure.

    timeout     : per-request timeout in seconds (default: OLLAMA_TIMEOUT env / 180s)
    max_retries : number of retry attempts on timeout (default: OLLAMA_MAX_RETRIES env / 2)
    """
    _timeout     = timeout     if timeout     is not None else OLLAMA_TIMEOUT
    _max_retries = max_retries if max_retries is not None else OLLAMA_MAX_RETRIES
    _num_ctx     = num_ctx     if num_ctx     is not None else OLLAMA_NUM_CTX
    _num_thread  = num_thread  if num_thread  is not None else OLLAMA_NUM_THREAD

    system = _LLM_SYSTEM
    if ignore:
        word_list = ", ".join(sorted(ignore))
        system += (
            f"\nAdditionally, treat these words as correctly spelled "
            f"and do NOT flag them: {word_list}."
        )

    options: dict = {"num_ctx": _num_ctx}
    if _num_thread > 0:
        options["num_thread"] = _num_thread

    payload = {
        "model":   model,
        "format":  "json",
        "stream":  False,
        "options": options,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": f'Text: """{text}"""'},
        ],
    }

    log.debug("LLM check: model=%s text_len=%d timeout=%ds retries=%d",
              model, len(text), _timeout, _max_retries)

    for attempt in range(1, _max_retries + 2):   # +2: first try + retries
        try:
            r = requests.post(f"{base_url}/api/chat", json=payload, timeout=_timeout)
            r.raise_for_status()
            raw    = r.json()["message"]["content"].strip()
            result = json.loads(raw)
            log.debug("LLM result (attempt %d): has_issues=%s issues=%d",
                      attempt, result.get("has_issues"), len(result.get("issues", [])))
            return result

        except requests.exceptions.Timeout:
            if attempt <= _max_retries:
                wait = attempt * 2   # simple backoff: 2s, 4s, …
                log.warning("Ollama timeout on attempt %d/%d — retrying in %ds",
                            attempt, _max_retries + 1, wait)
                import time
                time.sleep(wait)
                continue
            log.error("Ollama timed out after %d attempt(s) (timeout=%ds)", attempt, _timeout)
            return {
                "skipped": True,
                "reason":  f"Timed out after {attempt} attempt(s) ({_timeout}s each).",
            }

        except requests.exceptions.ConnectionError:
            log.error("Cannot connect to Ollama at %s", base_url)
            return {"error": f"Cannot connect to Ollama at {base_url}. Is it running? Try: ollama serve"}
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

    return {"skipped": True, "reason": "Ollama check failed after all retries."}


def warmup_model(
    model: str,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int | None = None,
) -> tuple[bool, str]:
    """
    Send a minimal request to Ollama to load the model into memory.
    Call this once before starting a batch to avoid cold-start timeouts.

    Returns (success, error_message).
    """
    # Warm-up gets extra budget on top of the regular timeout — loading a
    # 7B model from disk can take 60-120s on a CPU-only machine.
    _timeout = (timeout if timeout is not None else OLLAMA_TIMEOUT) + 120
    log.info("Warming up model '%s' (loading into memory)…", model)
    payload = {
        "model":  model,
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "user", "content": "Reply with exactly: {\"ok\": true}"},
        ],
    }
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=_timeout)
        r.raise_for_status()
        log.info("Model '%s' warm-up complete.", model)
        return True, ""
    except requests.exceptions.Timeout:
        msg = f"Model warm-up timed out after {_timeout}s — the model may be too large or the server is busy."
        log.error(msg)
        return False, msg
    except Exception as exc:
        msg = str(exc)
        log.error("Model warm-up failed: %s", msg)
        return False, msg


def _filter_llm_result(result: dict, ignore: set[str]) -> dict:
    """Remove issues whose original text matches an ignored word."""
    if "error" in result or "skipped" in result or not ignore:
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
