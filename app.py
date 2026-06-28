#!/usr/bin/env python3
"""
Streamlit UI for the JSON Spell & Grammar Agent.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from spellchecker import SpellChecker

from src.utils.spell import spell_check
from src.utils.llm import (ollama_check, _filter_llm_result, check_ollama,
                           warmup_model, OLLAMA_BASE_URL, OLLAMA_TIMEOUT, OLLAMA_MAX_RETRIES,
                           OLLAMA_NUM_CTX, OLLAMA_NUM_THREAD)
from src.utils.io import extract_strings, load_ignore_words, load_json_files_from_dir
from src.utils.git import fetch_json_files, list_branches
from src.utils.logger import get_logger
from src.utils import settings as _settings
from src.utils import run_manager as _rm

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="JSON Spell Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_SS_DEFAULTS: dict = {
    "is_running":       False,
    "run_phase":        "idle",    # "idle" | "spell" | "llm" | "done"
    "stop_event":       None,
    "worker_thread":    None,
    "progress_counter": [0],
    "llm_total":        0,
    "llm_results":      {},
    "run_context":      None,
    "was_stopped":      False,
    # Saved widget values so they survive the two-rerun pattern
    "_pending":         None,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Load persisted user preferences once per session
if "prefs" not in st.session_state:
    st.session_state.prefs = _settings.load()

# Reconnect to an active run if this is a fresh session after a refresh
if not st.session_state.is_running and not st.session_state.run_context:
    active = _rm.get()
    if active and active["thread"].is_alive():
        log.info("Reconnecting refreshed session to active run.")
        st.session_state.worker_thread    = active["thread"]
        st.session_state.stop_event       = active["stop_event"]
        st.session_state.progress_counter = active["progress_counter"]
        st.session_state.llm_results      = active["llm_results"]
        st.session_state.llm_total        = active["llm_total"]
        st.session_state.was_stopped      = active.get("was_stopped", False)
        st.session_state.run_context      = active["ctx"]
        st.session_state.is_running       = True
        st.session_state.run_phase        = "llm"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #0f172a; }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stTextArea label,
  section[data-testid="stSidebar"] .stCheckbox label {
    color: #94a3b8 !important; font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
  }
  .hero-title { font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
  .hero-sub { color: #64748b; font-size: 1rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }
  .stat-grid { display: flex; gap: 1rem; margin-bottom: 1.8rem; }
  .stat-card { flex: 1; border-radius: 12px; padding: 1.1rem 1.4rem; }
  .stat-card.total   { background: #1e293b; border: 1px solid #334155; }
  .stat-card.flagged { background: #2d1515; border: 1px solid #7f1d1d; }
  .stat-card.clean   { background: #0d2a1a; border: 1px solid #14532d; }
  .stat-card .label  { font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: #94a3b8; }
  .stat-card.flagged .label { color: #fca5a5; }
  .stat-card.clean   .label { color: #86efac; }
  .stat-card .value  { font-size: 2.4rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
  .stat-card.flagged .value { color: #f87171; }
  .stat-card.clean   .value { color: #4ade80; }
  .field-header { display: flex; align-items: center; gap: 0.5rem; font-family: monospace;
    font-size: 0.9rem; font-weight: 700; color: #cbd5e1; }
  .badge { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
    padding: 2px 8px; border-radius: 999px; text-transform: uppercase; }
  .badge-spell { background: #422006; color: #fed7aa; }
  .badge-llm   { background: #2e1065; color: #e9d5ff; }
  .badge-ok    { background: #052e16; color: #bbf7d0; }
  .badge-err   { background: #450a0a; color: #fecaca; }
  .issue-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }
  .issue-table th { background: #1e293b; color: #94a3b8; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em; padding: 6px 10px; text-align: left; }
  .issue-table td { padding: 6px 10px; border-bottom: 1px solid #1e293b; color: #e2e8f0; vertical-align: top; }
  .issue-table tr:last-child td { border-bottom: none; }
  .type-spelling { color: #f87171; font-weight: 600; }
  .type-grammar  { color: #fb923c; font-weight: 600; }
  .type-style    { color: #a78bfa; font-weight: 600; }
  .original   { color: #f87171; font-family: monospace; }
  .suggestion { color: #4ade80; font-family: monospace; }
  .fixed-box  { background: #0d2a1a; border: 1px solid #166534; border-radius: 8px;
    padding: 0.6rem 1rem; font-size: 0.88rem; color: #86efac; font-style: italic; margin-top: 0.5rem; }
  .fixed-label { font-size: 0.7rem; font-weight: 700; color: #4ade80;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px; }
  .preview { font-size: 0.82rem; color: #64748b; font-style: italic; margin: 0.2rem 0 0.6rem 1.4rem; }
  hr.field-div { border: none; border-top: 1px solid #1e293b; margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    health = check_ollama(base_url=OLLAMA_BASE_URL)

    if health["reachable"]:
        st.markdown(
            "<div style='background:#0d2a1a;border:1px solid #166534;border-radius:8px;"
            "padding:8px 12px;font-size:0.8rem;color:#86efac;margin-bottom:8px'>"
            "🟢 <b>Ollama is running</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#2d1515;border:1px solid #7f1d1d;border-radius:8px;"
            "padding:8px 12px;font-size:0.8rem;color:#fca5a5;margin-bottom:8px'>"
            "🔴 <b>Ollama not reachable</b><br>"
            "<code style='font-size:0.75rem'>ollama serve</code></div>",
            unsafe_allow_html=True,
        )

    prefs = st.session_state.prefs

    st.markdown("**LLM Model**")
    available_models  = health["models"]
    model_options     = ["(none — spell check only)"] + available_models
    saved_model       = prefs.get("selected_model", "mistral:7b")
    default_index     = next(
        (i + 1 for i, m in enumerate(available_models) if m == saved_model), 0
    )
    selected_model_label = st.selectbox(
        "Ollama model", model_options, index=default_index,
        label_visibility="collapsed", disabled=not health["reachable"],
    )
    selected_model = None if selected_model_label.startswith("(none") else selected_model_label

    if selected_model:
        mh = check_ollama(selected_model, OLLAMA_BASE_URL)
        if mh["model_ready"]:
            st.markdown(f"<span style='font-size:0.78rem;color:#4ade80'>✔ {selected_model} is ready</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:#2d1515;border:1px solid #7f1d1d;border-radius:6px;"
                f"padding:6px 10px;font-size:0.78rem;color:#fca5a5;margin-top:4px'>"
                f"⚠ Model not pulled<br><code>ollama pull {selected_model}</code></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("**Spell-check Language**")
    lang_opts = ["en", "es", "de", "fr", "pt"]
    lang = st.selectbox("Language", lang_opts,
                        index=lang_opts.index(prefs.get("lang", "en")),
                        label_visibility="collapsed")
    no_spellcheck = st.checkbox("Skip spell checker (LLM only)", value=prefs.get("no_spellcheck", True))
    st.markdown("---")
    st.markdown("**Ignore Words**")
    ignore_raw = st.text_area(
        "One word per line or space-separated",
        value=prefs.get("ignore_raw", ""),
        placeholder="BGP\nOSPF\nMPLS\ndatacenter",
        height=110, label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Parallel Workers**")
    workers = st.slider("LLM workers", min_value=1, max_value=12,
                        value=prefs.get("workers", 1), label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**LLM Timeout (seconds)**")
    llm_timeout = st.number_input(
        "Timeout per request",
        min_value=30, max_value=600, value=prefs.get("llm_timeout", OLLAMA_TIMEOUT), step=30,
        label_visibility="collapsed",
        help="Increase if you see timeout errors. Each retry uses the same timeout.",
    )
    st.markdown("**Retries on Timeout**")
    llm_retries = st.number_input(
        "Retries",
        min_value=0, max_value=5, value=prefs.get("llm_retries", OLLAMA_MAX_RETRIES), step=1,
        label_visibility="collapsed",
        help="Number of times to retry a timed-out request before marking it as failed.",
    )

    st.markdown("---")
    st.markdown("**Context Window** *(smaller = faster)*")
    ctx_opts = [512, 1024, 2048, 4096]
    saved_ctx = prefs.get("llm_num_ctx", OLLAMA_NUM_CTX)
    if saved_ctx not in ctx_opts:
        saved_ctx = 1024
    llm_num_ctx = st.select_slider(
        "num_ctx", options=ctx_opts, value=saved_ctx,
        label_visibility="collapsed",
        help="Tokens Ollama allocates per request. 1024 is plenty for spell-check.",
    )
    st.markdown("**CPU Threads for Ollama**")
    llm_num_thread = st.number_input(
        "num_thread (0 = auto)",
        min_value=0, max_value=32, value=prefs.get("llm_num_thread", OLLAMA_NUM_THREAD), step=1,
        label_visibility="collapsed",
        help="Limit Ollama's CPU threads so the app has headroom.",
    )

    st.markdown("---")
    if st.button("💾  Save Settings", use_container_width=True):
        new_prefs = {
            "selected_model": selected_model or "mistral:7b",
            "lang":           lang,
            "no_spellcheck":  no_spellcheck,
            "ignore_raw":     ignore_raw,
            "workers":        workers,
            "llm_timeout":    int(llm_timeout),
            "llm_retries":    int(llm_retries),
            "llm_num_ctx":    llm_num_ctx,
            "llm_num_thread": int(llm_num_thread),
        }
        _settings.save(new_prefs)
        st.session_state.prefs = new_prefs
        st.success("✔ Saved")
    st.markdown("---")

    # Run status indicator — always visible in sidebar
    if st.session_state.is_running:
        done  = st.session_state.progress_counter[0]
        total = st.session_state.llm_total
        pct   = int(done / total * 100) if total else 0
        st.markdown(
            f"<div style='background:#1e1b4b;border:1px solid #4338ca;border-radius:8px;"
            f"padding:8px 12px;font-size:0.8rem;color:#a5b4fc'>"
            f"🔄 <b>Running…</b> {done}/{total} fields ({pct}%)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span style='font-size:0.75rem;color:#475569'>Powered by pyspellchecker + Ollama</span>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper: LLM background worker
# ---------------------------------------------------------------------------

def _llm_worker(tasks, model, base_url, ignore_words, results, stop_event, n_workers, counter,
                timeout=240, max_retries=2, num_ctx=1024, num_thread=0):
    """Runs in a background thread. Writes results and checkpoints to disk."""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(ollama_check, model, text, base_url, ignore_words,
                            timeout, max_retries, num_ctx, num_thread): (name, field)
            for name, field, text in tasks
        }
        for future in as_completed(futures):
            if stop_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                _rm.mark_stopped()
                break
            name, field = futures[future]
            results[f"{name}||{field}"] = _filter_llm_result(future.result(), ignore_words)
            counter[0] += 1
            _rm.maybe_checkpoint(counter[0])
    _rm.finish()


# ---------------------------------------------------------------------------
# Helper: render field-by-field report (reused by live results + Reports tab)
# ---------------------------------------------------------------------------

def _render_field_rows(per_file_rows: dict, file_map: dict, key_prefix: str = "live") -> None:
    tab_labels = list(file_map.keys())
    tabs = st.tabs(tab_labels) if len(file_map) > 1 else [st.container()]

    for tab_idx, (tab, name) in enumerate(zip(tabs, file_map.keys())):
        all_rows  = per_file_rows.get(name, [])
        n_flagged = sum(1 for r in all_rows if r["has_issues"])
        tab_key   = f"{key_prefix}_{tab_idx}"

        with tab:
            st.markdown(
                f"**{name}** — "
                f"<span style='color:#f87171'>{n_flagged} flagged</span> / "
                f"<span style='color:#4ade80'>{len(all_rows) - n_flagged} clean</span> "
                f"out of {len(all_rows)} fields",
                unsafe_allow_html=True,
            )

            # ── Filter + search bar ────────────────────────────────────────
            col_f, col_s, col_p = st.columns([2, 3, 1])
            with col_f:
                view_filter = st.radio(
                    "Show",
                    ["All", "Flagged", "Clean"],
                    horizontal=True,
                    key=f"filter_{tab_key}",
                    label_visibility="collapsed",
                )
            with col_s:
                search = st.text_input(
                    "Search field path",
                    placeholder="🔍  Filter by field path…",
                    key=f"search_{tab_key}",
                    label_visibility="collapsed",
                )
            with col_p:
                page_size = st.selectbox(
                    "Per page",
                    [25, 50, 100, 200],
                    index=1,
                    key=f"pagesize_{tab_key}",
                    label_visibility="collapsed",
                )

            # Apply filters
            filtered = all_rows
            if view_filter == "Flagged":
                filtered = [r for r in filtered if r["has_issues"]]
            elif view_filter == "Clean":
                filtered = [r for r in filtered if not r["has_issues"]]
            if search.strip():
                q = search.strip().lower()
                filtered = [r for r in filtered if q in r["path"].lower()]

            total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
            page_key    = f"page_{tab_key}"
            if page_key not in st.session_state:
                st.session_state[page_key] = 1
            # Reset to page 1 when filters change
            current_page = st.session_state[page_key]
            current_page = max(1, min(current_page, total_pages))

            st.caption(f"{len(filtered)} field(s) shown · page {current_page} of {total_pages}")
            st.markdown("---")

            # Paginated rows
            start = (current_page - 1) * page_size
            for row in filtered[start : start + page_size]:
                _render_single_row(row)

            # ── Pagination controls ────────────────────────────────────────
            if total_pages > 1:
                col_prev, col_info, col_next = st.columns([1, 3, 1])
                with col_prev:
                    if st.button("← Prev", key=f"prev_{tab_key}",
                                 disabled=current_page <= 1, use_container_width=True):
                        st.session_state[page_key] = current_page - 1
                        st.rerun()
                with col_info:
                    st.markdown(
                        f"<div style='text-align:center;color:#64748b;padding-top:6px'>"
                        f"Page {current_page} of {total_pages}</div>",
                        unsafe_allow_html=True,
                    )
                with col_next:
                    if st.button("Next →", key=f"next_{tab_key}",
                                 disabled=current_page >= total_pages, use_container_width=True):
                        st.session_state[page_key] = current_page + 1
                        st.rerun()


def _render_single_row(row: dict) -> None:
    field      = row["path"]
    text       = row.get("text") or row.get("value", "")
    s_issues   = row.get("s_issues") or row.get("spell_issues", [])
    l_result   = row.get("l_result") or row.get("llm_result", {})
    has_issues = row["has_issues"]

    icon = "🔴" if has_issues else "🟢"
    badges = ""
    if s_issues:                        badges += '<span class="badge badge-spell">Spell</span> '
    if l_result.get("has_issues"):      badges += '<span class="badge badge-llm">LLM</span> '
    if l_result.get("skipped"):         badges += '<span class="badge" style="background:#451a03;color:#fcd34d">Skipped</span> '
    elif "error" in l_result:           badges += '<span class="badge badge-err">LLM Error</span> '
    if not has_issues and not l_result.get("skipped"): badges += '<span class="badge badge-ok">OK</span>'

    header = f"""<div class="field-header">
      <span>{icon}</span>
      <code style="color:#a5b4fc">{field}</code>
      {badges}
    </div>"""

    if has_issues:
        with st.expander(field, expanded=False):
            st.markdown(header, unsafe_allow_html=True)
            preview = text[:120] + ("…" if len(text) > 120 else "")
            st.markdown(f'<div class="preview">"{preview}"</div>', unsafe_allow_html=True)

            if s_issues:
                st.markdown("**🔡 Spell Check**")
                rows_html = "".join(
                    f"<tr><td class='original'>{i['word']}</td>"
                    f"<td class='suggestion'>{', '.join(i['suggestions'][:4]) or '—'}</td></tr>"
                    for i in s_issues
                )
                st.markdown(f"""<table class="issue-table">
                  <thead><tr><th>Misspelled</th><th>Suggestions</th></tr></thead>
                  <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)

            if l_result.get("skipped"):
                st.caption(f"⏭ LLM skipped: {l_result.get('reason', 'timeout')}")
            elif "error" in l_result:
                st.error(f"LLM error: {l_result['error']}")
            elif l_result.get("has_issues") and l_result.get("issues"):
                st.markdown("**🤖 LLM Analysis**")
                rows_html = "".join(
                    f"<tr>"
                    f"<td class='type-{i.get('type','spelling')}'>{i.get('type','').title()}</td>"
                    f"<td class='original'>{i.get('original','')}</td>"
                    f"<td class='suggestion'>{i.get('suggestion','')}</td>"
                    f"<td style='color:#94a3b8'>{i.get('explanation','')}</td>"
                    f"</tr>"
                    for i in l_result["issues"]
                )
                st.markdown(f"""<table class="issue-table">
                  <thead><tr><th>Type</th><th>Original</th><th>Suggestion</th><th>Explanation</th></tr></thead>
                  <tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
                corrected = l_result.get("corrected_text", "")
                if corrected:
                    st.markdown(
                        f'<div class="fixed-label">Suggested Fix</div>'
                        f'<div class="fixed-box">{corrected}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(header, unsafe_allow_html=True)

    st.markdown('<hr class="field-div">', unsafe_allow_html=True)


def _render_results(ctx: dict) -> None:
    """Render summary cards + per-file tabs from a stored run context."""
    file_map      = ctx["file_map"]
    per_file_rows = ctx["per_file_rows"]
    all_rows      = ctx["all_rows"]
    grand_flagged  = ctx["grand_flagged"]
    grand_skipped  = ctx.get("grand_skipped", 0)
    grand_total    = ctx["grand_total"]
    ignore_words   = ctx.get("ignore_words", set())

    skipped_card = (
        f'<div class="stat-card total" style="border-color:#854d0e">'
        f'<div class="label" style="color:#fbbf24">⏭ LLM Skipped</div>'
        f'<div class="value" style="color:#fbbf24">{grand_skipped}</div></div>'
        if grand_skipped else ""
    )

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card total"><div class="label">Files Checked</div>
        <div class="value">{len(file_map)}</div></div>
      <div class="stat-card total"><div class="label">Total Fields</div>
        <div class="value">{grand_total}</div></div>
      <div class="stat-card flagged"><div class="label">⚠ Flagged</div>
        <div class="value">{grand_flagged}</div></div>
      <div class="stat-card clean"><div class="label">✓ Clean</div>
        <div class="value">{grand_total - grand_flagged - grand_skipped}</div></div>
      {skipped_card}
    </div>""", unsafe_allow_html=True)
    if grand_skipped:
        st.caption(f"⏭ {grand_skipped} field(s) skipped by LLM due to timeout — spell check results still shown for those.")

    if ignore_words:
        st.caption(f"Ignoring {len(ignore_words)} word(s): {', '.join(sorted(ignore_words))}")

    st.markdown("---")
    _render_field_rows(per_file_rows, file_map)

    # Auto-save
    report_json = json.dumps(all_rows, indent=2, ensure_ascii=False)
    st.markdown("---")
    try:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        stamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"{stamp}_report.json"
        report_path.write_text(report_json, encoding="utf-8")
        st.success(f"✔ Report auto-saved → `{report_path}`")
    except Exception as e:
        st.warning(f"Could not save report to disk: {e}")
    st.download_button("⬇️  Download JSON Report", data=report_json,
                       file_name="spell_report.json", mime="application/json")


def _build_context(file_map, file_entries, spell_results, llm_results, ignore_words) -> dict:
    """Aggregate spell + LLM results into the context dict used by _render_results."""
    grand_flagged = 0
    per_file_rows: dict[str, list] = {}
    all_rows: list[dict] = []

    grand_skipped = 0
    for name, entries in file_entries.items():
        rows = []
        for field, text in entries:
            s_issues   = spell_results.get(name, {}).get(field, [])
            l_result   = llm_results.get(name, {}).get(field, {})
            llm_skipped = l_result.get("skipped", False)
            llm_bad    = l_result.get("has_issues", False) or ("error" in l_result and not llm_skipped)
            has_issues  = bool(s_issues) or llm_bad
            if has_issues:
                grand_flagged += 1
            if llm_skipped:
                grand_skipped += 1
            rows.append({"path": field, "text": text,
                         "s_issues": s_issues, "l_result": l_result, "has_issues": has_issues})
            all_rows.append({"file": name, "path": field, "value": text,
                             "has_issues": has_issues, "spell_issues": s_issues, "llm_result": l_result})
        per_file_rows[name] = rows

    return {
        "file_map": file_map, "file_entries": file_entries,
        "per_file_rows": per_file_rows, "all_rows": all_rows,
        "grand_flagged": grand_flagged, "grand_skipped": grand_skipped,
        "grand_total": sum(len(e) for e in file_entries.values()),
        "ignore_words": ignore_words,
    }


# ---------------------------------------------------------------------------
# Top-level tabs
# ---------------------------------------------------------------------------

st.markdown('<p class="hero-title">JSON Spell Checker</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Spell & grammar check for JSON files — with optional AI review.</p>',
            unsafe_allow_html=True)

tab_checker, tab_reports = st.tabs(["🔍  Spell Checker", "📋  Reports"])

# ===========================================================================
# TAB 1 — Spell Checker
# ===========================================================================

with tab_checker:

    # ── Show partial results from a previous interrupted run ────────────────
    if not st.session_state.is_running and not st.session_state.run_context:
        ckpt = _rm.load_checkpoint()
        if ckpt and ckpt.get("status") in ("in_progress", "stopped"):
            pct = int(ckpt["completed"] / ckpt["total"] * 100) if ckpt["total"] else 0
            st.warning(
                f"⚠ A previous run was interrupted at **{ckpt['completed']}/{ckpt['total']} fields ({pct}%)** "
                f"— model: `{ckpt.get('model','?')}` · saved: `{ckpt.get('saved_at','?')[:19]}`"
            )
            col_view, col_discard = st.columns([2, 1])
            with col_view:
                if st.button("📋 View partial results", use_container_width=True):
                    # Reconstruct minimal context from checkpoint for rendering
                    file_names    = ckpt.get("file_names", [])
                    file_entries  = ckpt.get("file_entries", {})
                    spell_results = ckpt.get("spell_results", {})
                    llm_flat      = ckpt.get("llm_results", {})
                    ignore_words  = set(ckpt.get("ignore_words", []))
                    file_map      = {n: {} for n in file_names}
                    llm_results   = {n: {} for n in file_names}
                    for key, val in llm_flat.items():
                        if "||" in key:
                            name, field = key.split("||", 1)
                            if name in llm_results:
                                llm_results[name][field] = val
                    ctx = _build_context(file_map, file_entries, spell_results,
                                         llm_results, ignore_words)
                    st.session_state.run_context = ctx
                    st.rerun()
            with col_discard:
                if st.button("🗑 Discard", use_container_width=True):
                    _rm.clear()
                    st.rerun()

    # ── Source selector (hidden while a run is in progress) ─────────────────
    uploaded_files = []
    json_text = local_dir = git_url = git_token = git_branch = git_subdir = ""
    recursive = False

    if not st.session_state.is_running:
        src_upload, src_dir, src_git = st.tabs([
            "⬆️  Upload / Paste", "📁  Local Directory", "🔗  Git Repository"
        ])

        with src_upload:
            uploaded_files = st.file_uploader(
                "Drop one or more JSON files", type=["json"],
                accept_multiple_files=True, label_visibility="collapsed",
            )
            col_paste, _ = st.columns([3, 1])
            with col_paste:
                json_text = st.text_area(
                    "…or paste JSON here", height=120,
                    placeholder='{\n  "message": "Welcom to the dashbord!"\n}',
                )

        with src_dir:
            st.markdown("Scan all `*.json` files inside a local directory.")
            col_d, col_r = st.columns([3, 1])
            with col_d:
                local_dir = st.text_input("Directory path",
                    placeholder="/home/you/project/configs",
                    label_visibility="collapsed",
                    key="local_dir_input")
            with col_r:
                recursive = st.checkbox("Recursive", value=False, key="recursive_input")

        with src_git:
            st.markdown("Fetch `*.json` files from GitHub / GitLab via API. Falls back to `git clone` for other hosts.")
            col_u, col_t = st.columns([3, 2])
            with col_u:
                git_url = st.text_input("Repo URL",
                    placeholder="https://github.com/org/repo", label_visibility="collapsed")
            with col_t:
                git_token = st.text_input("API Key / Token", type="password",
                    placeholder="Leave blank for public repos",
                    help="GitHub: PAT (`ghp_…`) · GitLab: Project token · Bitbucket: App password")
            col_b, col_s = st.columns([2, 3])
            with col_b:
                branches = list_branches(git_url.strip(), git_token.strip()) if git_url.strip() else []
                if branches:
                    git_branch = st.selectbox("Branch", branches,
                        index=next((i for i, b in enumerate(branches) if b in ("main","master")), 0))
                else:
                    git_branch = st.text_input("Branch", value="main")
            with col_s:
                git_subdir = st.text_input("Directory within repo",
                    placeholder="configs/  (leave blank for root)")

        _busy = st.session_state.run_phase != "idle"
        _btn_labels = {"idle": "🔍  Run Check", "spell": "⏳ Spell checking…",
                       "llm": "⏳ LLM reviewing…", "done": "🔍  Run Check"}
        col_run, _ = st.columns([1, 4])
        with col_run:
            run_btn = st.button(
                _btn_labels.get(st.session_state.run_phase, "🔍  Run Check"),
                type="primary",
                use_container_width=True,
                disabled=_busy or st.session_state.is_running,
            )

        # On click: save all widget values and trigger rerun so button
        # renders as disabled BEFORE the heavy work begins
        if run_btn and st.session_state.run_phase == "idle":
            st.session_state._pending = {
                "uploaded_files": uploaded_files,
                "json_text":      json_text,
                "local_dir":      local_dir,
                "recursive":      recursive,
                "git_url":        git_url,
                "git_token":      git_token,
                "git_branch":     git_branch,
                "git_subdir":     git_subdir,
                "selected_model": selected_model,
                "lang":           lang,
                "no_spellcheck":  no_spellcheck,
                "ignore_raw":     ignore_raw,
                "workers":        workers,
                "llm_timeout":    llm_timeout,
                "llm_retries":    llm_retries,
                "llm_num_ctx":    llm_num_ctx,
                "llm_num_thread": llm_num_thread,
            }
            st.session_state.run_phase = "spell"
            st.rerun()
    else:
        run_btn = False

    # ── Execute run (triggered by run_phase transition, not run_btn directly) ─
    if st.session_state.run_phase in ("spell", "llm") and not st.session_state.is_running:
        p = st.session_state._pending or {}
        uploaded_files  = p.get("uploaded_files", [])
        json_text       = p.get("json_text", "")
        local_dir       = p.get("local_dir", "")
        recursive       = p.get("recursive", False)
        git_url         = p.get("git_url", "")
        git_token       = p.get("git_token", "")
        git_branch      = p.get("git_branch", "main")
        git_subdir      = p.get("git_subdir", "")
        selected_model  = p.get("selected_model", None)
        lang            = p.get("lang", "en")
        no_spellcheck   = p.get("no_spellcheck", False)
        ignore_raw      = p.get("ignore_raw", "")
        workers         = p.get("workers", 4)
        llm_timeout     = p.get("llm_timeout",    OLLAMA_TIMEOUT)
        llm_retries     = p.get("llm_retries",    OLLAMA_MAX_RETRIES)
        llm_num_ctx     = p.get("llm_num_ctx",    OLLAMA_NUM_CTX)
        llm_num_thread  = p.get("llm_num_thread", OLLAMA_NUM_THREAD)

        file_map: dict = {}

        try:
            if uploaded_files or json_text.strip():
                for uf in (uploaded_files or []):
                    try:
                        file_map[uf.name] = json.load(uf)
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON in {uf.name}: {e}"); st.stop()
                if not file_map and json_text.strip():
                    try:
                        file_map["pasted_json"] = json.loads(json_text.strip())
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}"); st.stop()

            elif local_dir.strip():
                with st.spinner(f"Scanning `{local_dir.strip()}`…"):
                    p = Path(local_dir.strip())
                    if not p.is_dir():
                        st.error(f"Directory not found: `{p}`\n\nCheck the path is correct and accessible."); st.stop()
                    file_map = load_json_files_from_dir(p, recursive)
                if not file_map:
                    st.warning(f"No JSON files found in `{p}`"
                               + (" (try enabling **Recursive**)" if not recursive else "") + "."); st.stop()
                st.success(f"Found {len(file_map)} JSON file(s) in `{p}`")

            elif git_url.strip():
                if not git_url.strip().startswith("http"):
                    st.error("Please provide a full HTTPS URL."); st.stop()
                with st.spinner("Fetching JSON files from repository…"):
                    ok, file_map, err = fetch_json_files(
                        git_url.strip(), git_token.strip(),
                        git_branch or "main", git_subdir.strip().lstrip("/"),
                    )
                if not ok:
                    st.error(f"Failed:\n```\n{err}\n```"); st.stop()

            else:
                st.warning("Choose a source: upload a file, enter a directory path, or provide a Git URL.")
                st.stop()

        except Exception as _e:
            log.exception("Unexpected error loading files")
            st.session_state.run_phase = "idle"
            st.error(f"Unexpected error loading files: {_e}")
            st.exception(_e)
            st.stop()

        file_entries = {n: extract_strings(d) for n, d in file_map.items()}
        all_tasks    = [(n, f, t) for n, entries in file_entries.items() for f, t in entries]

        if not all_tasks:
            st.session_state.run_phase = "idle"
            st.warning("No string values found."); st.stop()

        words_list = ignore_raw.replace(",", " ").split()
        try:
            ignore_words = load_ignore_words(words_list, None)
        except FileNotFoundError as e:
            st.session_state.run_phase = "idle"
            st.error(str(e)); st.stop()

        # Pass 1 — spell check with live progress bar
        spell_results: dict = {n: {} for n in file_map}
        if not no_spellcheck:
            checker = SpellChecker(language=lang)
            if ignore_words:
                checker.word_frequency.load_words(ignore_words)
            spell_prog = st.progress(0, text=f"🔡 Spell checking 0 / {len(all_tasks)} fields…")
            for i, (n, f, t) in enumerate(all_tasks, 1):
                spell_results[n][f] = spell_check(checker, t, ignore_words)
                spell_prog.progress(i / len(all_tasks),
                                    text=f"🔡 Spell checking {i} / {len(all_tasks)} fields…")
            spell_prog.empty()
            log.info("Spell check complete: %d fields across %d files", len(all_tasks), len(file_map))

        # Pass 2 — LLM (background thread)
        stop_event = threading.Event()
        llm_results_shared: dict = {}
        progress_counter = [0]

        st.session_state.stop_event       = stop_event
        st.session_state.llm_results      = llm_results_shared
        st.session_state.progress_counter = progress_counter
        st.session_state.llm_total        = len(all_tasks) if selected_model else 0
        st.session_state.was_stopped      = False

        st.session_state.run_context = {
            "file_map":      file_map,
            "file_entries":  file_entries,
            "spell_results": spell_results,
            "all_tasks":     all_tasks,
            "ignore_words":  ignore_words,
            "has_llm":       bool(selected_model),
            "model":         selected_model,
        }

        if selected_model:
            with st.spinner(f"⏳ Loading `{selected_model}` into memory — this can take up to 2 min on first load…"):
                ok, warm_err = warmup_model(selected_model, OLLAMA_BASE_URL, llm_timeout)
            if not ok:
                log.warning("Warm-up timed out — proceeding anyway: %s", warm_err)
                st.warning(f"⚠ Model warm-up timed out — proceeding anyway. "
                           f"First few fields may be slow. ({warm_err})")

            thread = threading.Thread(
                target=_llm_worker,
                args=(all_tasks, selected_model, OLLAMA_BASE_URL, ignore_words,
                      llm_results_shared, stop_event, workers, progress_counter,
                      llm_timeout, llm_retries, llm_num_ctx, llm_num_thread),
                daemon=True,
            )
            thread.start()
            st.session_state.worker_thread        = thread
            st.session_state.is_running           = True
            st.session_state.run_phase            = "llm"
            st.session_state._last_render_count   = 0
            _rm.start(
                st.session_state.run_context,
                thread, stop_event, progress_counter,
                llm_results_shared, len(all_tasks),
            )
            st.rerun()
        else:
            ctx = _build_context(file_map, file_entries, spell_results, {}, ignore_words)
            st.session_state.run_phase = "idle"
            _render_results(ctx)

    # ── Polling loop while LLM thread is alive ───────────────────────────────
    if st.session_state.is_running:
        thread  = st.session_state.worker_thread
        total   = st.session_state.llm_total
        done    = st.session_state.progress_counter[0]
        model   = st.session_state.run_context.get("model", "")
        pct     = done / total if total else 0
        ctx_raw = st.session_state.run_context
        n_files = len(ctx_raw.get("file_map", {}))

        # Status banner
        st.markdown(f"""
        <div style="background:#1e1b4b;border:1px solid #4338ca;border-radius:12px;
                    padding:1rem 1.4rem;margin-bottom:1rem">
          <div style="font-size:1rem;font-weight:700;color:#a5b4fc;margin-bottom:0.4rem">
            🔄 Check in progress
          </div>
          <div style="font-size:0.85rem;color:#c7d2fe;display:flex;gap:2rem;flex-wrap:wrap">
            <span>Model <b style="color:#e0e7ff">{model}</b></span>
            <span>Files <b style="color:#e0e7ff">{n_files}</b></span>
            <span>Fields <b style="color:#e0e7ff">{done} / {total}</b> checked</span>
            <span>Progress <b style="color:#e0e7ff">{pct*100:.0f}%</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_bar, col_btn = st.columns([5, 1])
        with col_bar:
            st.progress(pct)
        with col_btn:
            if st.button("⏹ Stop", type="secondary", use_container_width=True):
                st.session_state.stop_event.set()
                st.session_state.was_stopped = True

        # ── Live partial results (re-render every 50 completed fields) ───────
        BATCH_RENDER = 50
        last_render  = st.session_state.get("_last_render_count", 0)
        if done >= last_render + BATCH_RENDER or (done > 0 and last_render == 0):
            st.session_state._last_render_count = done
            llm_flat     = st.session_state.llm_results
            llm_partial  = {n: {} for n in ctx_raw["file_map"]}
            for key, val in llm_flat.items():
                if "||" in key:
                    n, f = key.split("||", 1)
                    if n in llm_partial:
                        llm_partial[n][f] = val
            partial_ctx = _build_context(
                ctx_raw["file_map"], ctx_raw["file_entries"],
                ctx_raw["spell_results"], llm_partial, ctx_raw["ignore_words"],
            )
            st.markdown(f"---\n**Live Results** *(updating every {BATCH_RENDER} fields — {done} done so far)*")
            _render_field_rows(partial_ctx["per_file_rows"], partial_ctx["file_map"],
                               key_prefix="live_partial")

        if thread and thread.is_alive():
            time.sleep(2)   # slower poll — partial results are rendered above
            st.rerun()
        else:
            # Thread finished (or was stopped) — compile and render
            st.session_state.is_running = False
            st.session_state.run_phase  = "idle"
            st.session_state._pending   = None
            _rm.clear()
            ctx_raw   = st.session_state.run_context
            llm_flat  = st.session_state.llm_results

            # Unflatten "name||field" → nested dict
            llm_results: dict = {n: {} for n in ctx_raw["file_map"]}
            for key, val in llm_flat.items():
                if "||" in key:
                    name, field = key.split("||", 1)
                    if name in llm_results:
                        llm_results[name][field] = val

            ctx = _build_context(
                ctx_raw["file_map"], ctx_raw["file_entries"],
                ctx_raw["spell_results"], llm_results, ctx_raw["ignore_words"],
            )

            if st.session_state.was_stopped:
                st.warning("⏹ Run stopped early — results below are partial.")

            _render_results(ctx)

    # ── Show last results if nothing is running and no button pressed ─────────
    elif not run_btn and st.session_state.run_context and not st.session_state.is_running:
        # Results already rendered above; nothing extra needed on a clean rerun
        pass


# ===========================================================================
# TAB 2 — Reports
# ===========================================================================

with tab_reports:
    reports_dir = Path("reports")
    report_files = sorted(reports_dir.glob("*.json"), reverse=True) if reports_dir.exists() else []

    if not report_files:
        st.info("No reports yet — run a check first.")
    else:
        col_sel, col_del = st.columns([4, 1])
        with col_sel:
            selected_report = st.selectbox(
                "Select a saved report",
                report_files,
                format_func=lambda p: f"{p.stem}  ({p.stat().st_size // 1024} KB)",
                label_visibility="collapsed",
            )
        with col_del:
            if st.button("🗑 Delete", use_container_width=True):
                selected_report.unlink(missing_ok=True)
                st.success(f"Deleted `{selected_report.name}`")
                st.rerun()

        if selected_report and selected_report.exists():
            data = json.loads(selected_report.read_text(encoding="utf-8"))

            # Reconstruct per_file_rows and file_map from saved report rows
            file_map_r:      dict = {}
            per_file_rows_r: dict = {}
            for row in data:
                name = row.get("file", "unknown")
                if name not in file_map_r:
                    file_map_r[name] = {}
                    per_file_rows_r[name] = []
                per_file_rows_r[name].append(row)

            flagged = sum(1 for r in data if r.get("has_issues"))
            total   = len(data)

            st.markdown(f"**`{selected_report.name}`** — "
                        f"<span style='color:#f87171'>{flagged} flagged</span> / "
                        f"<span style='color:#4ade80'>{total - flagged} clean</span> "
                        f"across {len(file_map_r)} file(s)",
                        unsafe_allow_html=True)
            st.markdown("---")

            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card total"><div class="label">Files</div>
                <div class="value">{len(file_map_r)}</div></div>
              <div class="stat-card total"><div class="label">Total Fields</div>
                <div class="value">{total}</div></div>
              <div class="stat-card flagged"><div class="label">⚠ Flagged</div>
                <div class="value">{flagged}</div></div>
              <div class="stat-card clean"><div class="label">✓ Clean</div>
                <div class="value">{total - flagged}</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            _render_field_rows(per_file_rows_r, file_map_r, key_prefix="report")

            st.markdown("---")
            st.download_button(
                "⬇️  Download this report",
                data=selected_report.read_text(encoding="utf-8"),
                file_name=selected_report.name,
                mime="application/json",
            )
