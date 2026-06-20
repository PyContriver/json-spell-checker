#!/usr/bin/env python3
"""
Streamlit UI for the JSON Spell & Grammar Agent.
"""

import json
from pathlib import Path
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from spellchecker import SpellChecker

from src.utils.spell import spell_check
from src.utils.llm import ollama_list_models, ollama_check, _filter_llm_result, check_ollama, OLLAMA_BASE_URL
from src.utils.io import extract_strings, load_ignore_words, load_json_files_from_dir
from src.utils.git import fetch_json_files, list_branches

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
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0f172a; }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stTextArea label,
  section[data-testid="stSidebar"] .stCheckbox label { color: #94a3b8 !important; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }

  /* Main heading */
  .hero-title { font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; }
  .hero-sub   { color: #64748b; font-size: 1rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }

  /* Stat cards */
  .stat-grid { display: flex; gap: 1rem; margin-bottom: 1.8rem; }
  .stat-card { flex: 1; border-radius: 12px; padding: 1.1rem 1.4rem; }
  .stat-card.total   { background: #1e293b; border: 1px solid #334155; }
  .stat-card.flagged { background: #2d1515; border: 1px solid #7f1d1d; }
  .stat-card.clean   { background: #0d2a1a; border: 1px solid #14532d; }
  .stat-card .label  { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #94a3b8; }
  .stat-card.flagged .label { color: #fca5a5; }
  .stat-card.clean   .label { color: #86efac; }
  .stat-card .value  { font-size: 2.4rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
  .stat-card.flagged .value { color: #f87171; }
  .stat-card.clean   .value { color: #4ade80; }

  /* Field rows */
  .field-header { display: flex; align-items: center; gap: 0.5rem; font-family: monospace; font-size: 0.9rem; font-weight: 700; color: #cbd5e1; }
  .badge { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; padding: 2px 8px; border-radius: 999px; text-transform: uppercase; }
  .badge-spell  { background: #422006; color: #fed7aa; }
  .badge-llm    { background: #2e1065; color: #e9d5ff; }
  .badge-ok     { background: #052e16; color: #bbf7d0; }
  .badge-err    { background: #450a0a; color: #fecaca; }

  /* Issue table */
  .issue-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }
  .issue-table th { background: #1e293b; color: #94a3b8; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; padding: 6px 10px; text-align: left; }
  .issue-table td { padding: 6px 10px; border-bottom: 1px solid #1e293b; color: #e2e8f0; vertical-align: top; }
  .issue-table tr:last-child td { border-bottom: none; }
  .type-spelling { color: #f87171; font-weight: 600; }
  .type-grammar  { color: #fb923c; font-weight: 600; }
  .type-style    { color: #a78bfa; font-weight: 600; }
  .original      { color: #f87171; font-family: monospace; }
  .suggestion    { color: #4ade80; font-family: monospace; }

  /* Fixed text */
  .fixed-box { background: #0d2a1a; border: 1px solid #166534; border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.88rem; color: #86efac; font-style: italic; margin-top: 0.5rem; }
  .fixed-label { font-size: 0.7rem; font-weight: 700; color: #4ade80; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px; }

  /* Preview text */
  .preview { font-size: 0.82rem; color: #64748b; font-style: italic; margin: 0.2rem 0 0.6rem 1.4rem; }

  /* Dividers */
  hr.field-div { border: none; border-top: 1px solid #1e293b; margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # --- Ollama health check (runs on every page load) ---
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

    # Ollama model selector — default to mistral:7b if available
    DEFAULT_MODEL = "mistral:7b"
    st.markdown("**LLM Model**")
    available_models = health["models"]
    model_options    = ["(none — spell check only)"] + available_models
    default_index    = next(
        (i + 1 for i, m in enumerate(available_models) if m == DEFAULT_MODEL),
        0,  # fall back to "(none)" if mistral:7b isn't pulled yet
    )
    selected_model_label = st.selectbox(
        "Ollama model",
        model_options,
        index=default_index,
        label_visibility="collapsed",
        disabled=not health["reachable"],
    )
    selected_model = None if selected_model_label.startswith("(none") else selected_model_label

    # Per-model readiness badge
    if selected_model:
        model_health = check_ollama(selected_model, OLLAMA_BASE_URL)
        if model_health["model_ready"]:
            st.markdown(
                f"<span style='font-size:0.78rem;color:#4ade80'>✔ {selected_model} is ready</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#2d1515;border:1px solid #7f1d1d;border-radius:6px;"
                f"padding:6px 10px;font-size:0.78rem;color:#fca5a5;margin-top:4px'>"
                f"⚠ Model not pulled<br>"
                f"<code>ollama pull {selected_model}</code></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Spell check language
    st.markdown("**Spell-check Language**")
    lang = st.selectbox(
        "Language",
        ["en", "es", "de", "fr", "pt"],
        label_visibility="collapsed",
    )
    no_spellcheck = st.checkbox("Skip spell checker (LLM only)", value=False)

    st.markdown("---")

    # Ignore words
    st.markdown("**Ignore Words**")
    ignore_raw = st.text_area(
        "One word per line or space-separated",
        placeholder="BGP\nOSPF\nMPLS\ndatacenter",
        height=110,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Parallel Workers**")
    workers = st.slider("LLM workers", min_value=1, max_value=12, value=4, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.75rem;color:#475569'>"
        "Powered by pyspellchecker + Ollama"
        "</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<p class="hero-title">JSON Spell Checker</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a JSON file and get a beautiful spell & grammar report — with optional AI review.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Source selector
# ---------------------------------------------------------------------------

src_tab_upload, src_tab_dir, src_tab_git = st.tabs([
    "⬆️  Upload / Paste",
    "📁  Local Directory",
    "🔗  Git Repository",
])

with src_tab_upload:
    uploaded_files = st.file_uploader(
        "Drop one or more JSON files",
        type=["json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    col_paste, _ = st.columns([3, 1])
    with col_paste:
        json_text = st.text_area(
            "…or paste JSON here (single file)",
            height=120,
            placeholder='{\n  "message": "Welcom to the dashbord!"\n}',
        )

with src_tab_dir:
    st.markdown("Scan all `*.json` files inside a local directory.")
    col_dir, col_rec = st.columns([3, 1])
    with col_dir:
        local_dir = st.text_input(
            "Directory path",
            placeholder="/Users/you/project/configs",
            label_visibility="collapsed",
        )
    with col_rec:
        recursive = st.checkbox("Recursive", value=False, help="Also scan subdirectories")

with src_tab_git:
    st.markdown("Fetch `*.json` files directly from GitHub or GitLab via API — no full clone needed. Falls back to `git clone` for Bitbucket and other hosts.")

    col_url, col_token = st.columns([3, 2])
    with col_url:
        git_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/org/repo  or  https://gitlab.com/org/repo",
            label_visibility="collapsed",
        )
    with col_token:
        git_token = st.text_input(
            "API Key / Token",
            type="password",
            placeholder="Leave blank for public repos",
            help=(
                "**GitHub** — Personal Access Token (`ghp_…`)\n\n"
                "**GitLab** — Project / Group token\n\n"
                "**Bitbucket** — App password\n\n"
                "**Self-hosted GitLab** — same as GitLab cloud"
            ),
        )

    # Dynamically load branches when URL + token are provided
    col_branch, col_subdir = st.columns([2, 3])
    with col_branch:
        branches = []
        if git_url.strip():
            with st.spinner("Loading branches…"):
                branches = list_branches(git_url.strip(), git_token.strip())
        if branches:
            git_branch = st.selectbox("Branch", branches,
                                       index=next((i for i, b in enumerate(branches) if b in ("main","master")), 0))
        else:
            git_branch = st.text_input("Branch", value="main")
    with col_subdir:
        git_subdir = st.text_input(
            "Directory within repo",
            placeholder="configs/  (leave blank for root)",
            help="Relative path inside the repo to scan for JSON files",
        )

run_btn = st.button("🔍  Run Check", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------------

if run_btn:
    file_map: dict[str, any] = {}

    # --- Mode: Upload / Paste ---
    if uploaded_files or json_text.strip():
        for uf in (uploaded_files or []):
            try:
                file_map[uf.name] = json.load(uf)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in {uf.name}: {e}")
                st.stop()
        if not file_map and json_text.strip():
            try:
                file_map["pasted_json"] = json.loads(json_text.strip())
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

    # --- Mode: Local Directory ---
    elif local_dir.strip():
        scan_path = Path(local_dir.strip())
        if not scan_path.is_dir():
            st.error(f"Directory not found: `{scan_path}`")
            st.stop()
        file_map = load_json_files_from_dir(scan_path, recursive)
        if not file_map:
            st.warning(f"No JSON files found in `{scan_path}`.")
            st.stop()

    # --- Mode: Git Repository ---
    elif git_url.strip():
        if not git_url.strip().startswith("http"):
            st.error("Please provide a full HTTPS repository URL.")
            st.stop()

        with st.spinner("Fetching JSON files from repository…"):
            success, file_map, err = fetch_json_files(
                url    = git_url.strip(),
                token  = git_token.strip(),
                branch = git_branch if isinstance(git_branch, str) else git_branch,
                subdir = git_subdir.strip().lstrip("/"),
            )
            if not success:
                st.error(f"Failed to fetch repository:\n```\n{err}\n```")
                st.stop()

    else:
        st.warning("Choose a source: upload files, enter a directory path, or provide a Git repo URL.")
        st.stop()

    # --- Build entries per file ---
    file_entries: dict[str, list] = {
        name: extract_strings(data) for name, data in file_map.items()
    }
    all_tasks = [
        (name, field, text)
        for name, entries in file_entries.items()
        for field, text in entries
    ]
    total_fields = len(all_tasks)

    if total_fields == 0:
        st.warning("No string values found in any of the uploaded files.")
        st.stop()

    # --- Build ignore set ---
    words_list = ignore_raw.replace(",", " ").split()
    try:
        ignore_words = load_ignore_words(words_list, None)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # --- Pass 1: spell check ---
    spell_results: dict[str, dict] = {name: {} for name in file_map}
    with st.spinner("Running spell check…"):
        if not no_spellcheck:
            checker = SpellChecker(language=lang)
            if ignore_words:
                checker.word_frequency.load_words(ignore_words)
            for name, field, text in all_tasks:
                spell_results[name][field] = spell_check(checker, text, ignore_words)

    # --- Pass 2: LLM — all tasks in parallel ---
    llm_results: dict[str, dict] = {name: {} for name in file_map}
    if selected_model:
        progress_bar = st.progress(0, text=f"LLM [{selected_model}] ×{workers} workers…")
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    ollama_check, selected_model, text, OLLAMA_BASE_URL, ignore_words
                ): (name, field)
                for name, field, text in all_tasks
            }
            for future in as_completed(futures):
                name, field = futures[future]
                llm_results[name][field] = _filter_llm_result(future.result(), ignore_words)
                completed += 1
                progress_bar.progress(completed / total_fields,
                                       text=f"LLM [{selected_model}] — {completed}/{total_fields} done")
        progress_bar.empty()

    # --- Aggregate totals ---
    grand_flagged = 0
    grand_total   = total_fields
    all_report_rows: list[dict] = []

    per_file_rows: dict[str, list] = {name: [] for name in file_map}
    for name, entries in file_entries.items():
        for field, text in entries:
            s_issues  = spell_results[name].get(field, [])
            l_result  = llm_results[name].get(field, {})
            llm_error = "error" in l_result
            llm_bad   = l_result.get("has_issues", False) or llm_error
            has_issues = bool(s_issues) or llm_bad
            if has_issues:
                grand_flagged += 1
            per_file_rows[name].append({
                "path": field, "text": text,
                "s_issues": s_issues, "l_result": l_result,
                "has_issues": has_issues,
            })
            all_report_rows.append({
                "file": name, "path": field, "value": text,
                "has_issues": has_issues,
                "spell_issues": s_issues, "llm_result": l_result,
            })

    # ---------------------------------------------------------------------------
    # Summary cards (grand totals)
    # ---------------------------------------------------------------------------

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card total">
        <div class="label">Files Checked</div>
        <div class="value">{len(file_map)}</div>
      </div>
      <div class="stat-card total">
        <div class="label">Total Fields</div>
        <div class="value">{grand_total}</div>
      </div>
      <div class="stat-card flagged">
        <div class="label">⚠ Flagged</div>
        <div class="value">{grand_flagged}</div>
      </div>
      <div class="stat-card clean">
        <div class="label">✓ Clean</div>
        <div class="value">{grand_total - grand_flagged}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if ignore_words:
        st.caption(f"Ignoring {len(ignore_words)} word(s): {', '.join(sorted(ignore_words))}")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Per-file tabs
    # ---------------------------------------------------------------------------

    tab_labels = list(file_map.keys())
    tabs = st.tabs(tab_labels) if len(file_map) > 1 else [st.container()]

    for tab, name in zip(tabs, file_map.keys()):
        rows      = per_file_rows[name]
        n_flagged = sum(1 for r in rows if r["has_issues"])
        with tab:
            st.markdown(
                f"**{name}** — "
                f"<span style='color:#f87171'>{n_flagged} flagged</span> / "
                f"<span style='color:#4ade80'>{len(rows) - n_flagged} clean</span> "
                f"out of {len(rows)} fields",
                unsafe_allow_html=True,
            )
            st.markdown("---")

            for row in rows:
                field     = row["path"]
                text      = row["text"]
                s_issues  = row["s_issues"]
                l_result  = row["l_result"]
                has_issues = row["has_issues"]

                icon = "🔴" if has_issues else "🟢"
                badges_html = ""
                if s_issues:
                    badges_html += '<span class="badge badge-spell">Spell</span> '
                if l_result.get("has_issues"):
                    badges_html += '<span class="badge badge-llm">LLM</span> '
                if "error" in l_result:
                    badges_html += '<span class="badge badge-err">LLM Error</span> '
                if not has_issues:
                    badges_html += '<span class="badge badge-ok">OK</span>'

                header_html = f"""
                <div class="field-header">
                  <span>{icon}</span>
                  <code style="color:#a5b4fc">{field}</code>
                  {badges_html}
                </div>
                """

                if has_issues:
                    with st.expander(field, expanded=False):
                        st.markdown(header_html, unsafe_allow_html=True)
                        preview = text[:120] + ("…" if len(text) > 120 else "")
                        st.markdown(f'<div class="preview">"{preview}"</div>', unsafe_allow_html=True)

                        if s_issues:
                            st.markdown("**🔡 Spell Check**")
                            rows_html = "".join(
                                f"<tr><td class='original'>{i['word']}</td>"
                                f"<td class='suggestion'>{', '.join(i['suggestions'][:4]) or '—'}</td></tr>"
                                for i in s_issues
                            )
                            st.markdown(f"""
                            <table class="issue-table">
                              <thead><tr><th>Misspelled</th><th>Suggestions</th></tr></thead>
                              <tbody>{rows_html}</tbody>
                            </table>
                            """, unsafe_allow_html=True)

                        if "error" in l_result:
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
                            st.markdown(f"""
                            <table class="issue-table">
                              <thead><tr><th>Type</th><th>Original</th><th>Suggestion</th><th>Explanation</th></tr></thead>
                              <tbody>{rows_html}</tbody>
                            </table>
                            """, unsafe_allow_html=True)
                            corrected = l_result.get("corrected_text", "")
                            if corrected:
                                st.markdown(f"""
                                <div class="fixed-label">Suggested Fix</div>
                                <div class="fixed-box">{corrected}</div>
                                """, unsafe_allow_html=True)
                else:
                    st.markdown(header_html, unsafe_allow_html=True)

                st.markdown('<hr class="field-div">', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Download report
    # ---------------------------------------------------------------------------

    st.markdown("---")
    st.download_button(
        label="⬇️  Download JSON Report",
        data=json.dumps(all_report_rows, indent=2, ensure_ascii=False),
        file_name="spell_report.json",
        mime="application/json",
    )

