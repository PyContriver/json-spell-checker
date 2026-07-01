"""
HTML report generator — produces a self-contained, human-readable report
with a top-affected-files summary and expandable field details.
"""

from datetime import datetime
from typing import Any


def generate_html(report_rows: list[dict], meta: dict | None = None) -> str:
    """
    Build a self-contained HTML report from a list of report row dicts.

    Parameters
    ----------
    report_rows : list of dicts with keys:
        file, path, value, has_issues, spell_issues, llm_result
    meta : optional dict with keys: model, total, flagged, skipped, generated_at
    """
    meta = meta or {}
    generated_at = meta.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model         = meta.get("model", "—")
    total         = meta.get("total", len(report_rows))
    flagged       = meta.get("flagged", sum(1 for r in report_rows if r.get("has_issues")))
    skipped       = meta.get("skipped", sum(
        1 for r in report_rows
        if r.get("llm_result", {}).get("skipped")
    ))
    clean         = total - flagged - skipped

    # ── Per-file stats ──────────────────────────────────────────────────────
    file_stats: dict[str, dict] = {}
    for row in report_rows:
        fname = row.get("file", "unknown")
        if fname not in file_stats:
            file_stats[fname] = {"total": 0, "flagged": 0, "skipped": 0, "issues": []}
        file_stats[fname]["total"] += 1
        if row.get("has_issues"):
            file_stats[fname]["flagged"] += 1
        if row.get("llm_result", {}).get("skipped"):
            file_stats[fname]["skipped"] += 1
        if row.get("has_issues"):
            file_stats[fname]["issues"].append(row)

    top_files = sorted(file_stats.items(), key=lambda x: x[1]["flagged"], reverse=True)

    # ── CSS ─────────────────────────────────────────────────────────────────
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f172a; color: #e2e8f0; padding: 2rem; }
    h1   { font-size: 1.8rem; font-weight: 800;
           background: linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2   { font-size: 1.1rem; font-weight: 700; color: #94a3b8;
           text-transform: uppercase; letter-spacing: .06em; margin: 2rem 0 .8rem; }
    h3   { font-size: .95rem; font-weight: 700; color: #cbd5e1; margin-bottom: .4rem; }
    .meta { color: #64748b; font-size: .85rem; margin-top: .3rem; }
    /* stat cards */
    .cards { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
    .card  { flex: 1; min-width: 120px; border-radius: 10px; padding: .9rem 1.2rem; }
    .card.total   { background:#1e293b; border:1px solid #334155; }
    .card.flagged { background:#2d1515; border:1px solid #7f1d1d; }
    .card.clean   { background:#0d2a1a; border:1px solid #14532d; }
    .card.skipped { background:#1c1400; border:1px solid #854d0e; }
    .card .label  { font-size:.7rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:.06em; color:#94a3b8; }
    .card.flagged .label { color:#fca5a5; }
    .card.clean   .label { color:#86efac; }
    .card.skipped .label { color:#fcd34d; }
    .card .value  { font-size:2rem; font-weight:800; color:#f1f5f9; line-height:1.1; }
    .card.flagged .value { color:#f87171; }
    .card.clean   .value { color:#4ade80; }
    .card.skipped .value { color:#fbbf24; }
    /* top files table */
    table { width:100%; border-collapse:collapse; font-size:.85rem; }
    thead th { background:#1e293b; color:#94a3b8; font-size:.7rem; font-weight:700;
               text-transform:uppercase; letter-spacing:.05em; padding:8px 12px; text-align:left; }
    tbody td { padding:8px 12px; border-bottom:1px solid #1e293b; vertical-align:middle; }
    tbody tr:hover td { background:#1e293b; }
    .bar-wrap { background:#1e293b; border-radius:4px; height:8px; width:120px; display:inline-block; }
    .bar-fill { background:#f87171; border-radius:4px; height:8px; display:block; }
    .pct { color:#94a3b8; font-size:.78rem; margin-left:.4rem; }
    /* file sections */
    .file-section { margin-bottom:2rem; }
    .file-header  { background:#1e293b; border:1px solid #334155; border-radius:8px;
                    padding:.7rem 1rem; margin-bottom:.5rem; display:flex;
                    justify-content:space-between; align-items:center; }
    .file-name { font-family:monospace; color:#a5b4fc; font-size:.9rem; font-weight:700; }
    .file-badge { font-size:.72rem; font-weight:700; padding:2px 8px; border-radius:999px;
                  background:#2d1515; color:#fca5a5; }
    /* field rows */
    details { margin-bottom:.4rem; }
    details summary { cursor:pointer; list-style:none; padding:.5rem .8rem;
                      background:#1e293b; border-radius:6px; display:flex;
                      align-items:center; gap:.5rem; font-size:.85rem;
                      user-select:none; }
    details summary::-webkit-details-marker { display:none; }
    details summary:hover { background:#243047; }
    details[open] summary { border-radius:6px 6px 0 0; }
    .detail-body { background:#131f35; border:1px solid #1e293b;
                   border-top:none; border-radius:0 0 6px 6px; padding:.8rem 1rem; }
    .preview { color:#64748b; font-style:italic; font-size:.82rem; margin-bottom:.6rem; }
    /* issue tables */
    .issue-table { width:100%; border-collapse:collapse; font-size:.82rem; margin-top:.4rem; }
    .issue-table th { background:#1a2a42; color:#94a3b8; font-size:.68rem; font-weight:700;
                      text-transform:uppercase; padding:5px 8px; text-align:left; }
    .issue-table td { padding:5px 8px; border-bottom:1px solid #1e293b; }
    .issue-table tr:last-child td { border-bottom:none; }
    .spell  { color:#f87171; font-family:monospace; }
    .sugg   { color:#4ade80; font-family:monospace; }
    .type-spelling { color:#f87171; font-weight:600; }
    .type-grammar  { color:#fb923c; font-weight:600; }
    .type-style    { color:#a78bfa; font-weight:600; }
    .original   { color:#f87171; font-family:monospace; }
    .suggestion { color:#4ade80; font-family:monospace; }
    .fixed-box  { background:#0d2a1a; border:1px solid #166534; border-radius:6px;
                  padding:.5rem .8rem; font-size:.82rem; color:#86efac;
                  font-style:italic; margin-top:.5rem; }
    .section-label { font-size:.7rem; font-weight:700; color:#94a3b8;
                     text-transform:uppercase; letter-spacing:.05em; margin:.6rem 0 .2rem; }
    .badge { display:inline-block; font-size:.68rem; font-weight:700; padding:1px 6px;
             border-radius:999px; text-transform:uppercase; margin-left:.3rem; }
    .badge-spell { background:#422006; color:#fed7aa; }
    .badge-llm   { background:#2e1065; color:#e9d5ff; }
    .badge-skip  { background:#451a03; color:#fcd34d; }
    .ok { color:#4ade80; }
    .flag { color:#f87171; }
    footer { margin-top:3rem; padding-top:1rem; border-top:1px solid #1e293b;
             color:#475569; font-size:.78rem; text-align:center; }
    """

    # ── Top files table ──────────────────────────────────────────────────────
    top_rows_html = ""
    for rank, (fname, stats) in enumerate(top_files[:20], 1):
        pct = int(stats["flagged"] / stats["total"] * 100) if stats["total"] else 0
        bar_w = max(1, pct)
        top_rows_html += f"""
        <tr>
          <td style="color:#64748b">{rank}</td>
          <td style="font-family:monospace;color:#a5b4fc">{fname}</td>
          <td style="color:#f87171;font-weight:700">{stats['flagged']}</td>
          <td>{stats['total']}</td>
          <td>
            <span class="bar-wrap"><span class="bar-fill" style="width:{bar_w}%"></span></span>
            <span class="pct">{pct}%</span>
          </td>
        </tr>"""

    # ── Field details per file ───────────────────────────────────────────────
    file_sections_html = ""
    for fname, stats in top_files:
        if not stats["issues"]:
            continue
        fields_html = ""
        for row in stats["issues"]:
            field      = row.get("path", "")
            value      = row.get("value") or row.get("text", "")
            s_issues   = row.get("spell_issues", [])
            l_result   = row.get("llm_result", {})
            llm_bad    = l_result.get("has_issues") and not l_result.get("skipped")
            llm_skip   = l_result.get("skipped")

            badges = ""
            if s_issues:  badges += '<span class="badge badge-spell">Spell</span>'
            if llm_bad:   badges += '<span class="badge badge-llm">LLM</span>'
            if llm_skip:  badges += '<span class="badge badge-skip">Skipped</span>'

            spell_html = ""
            if s_issues:
                rows = "".join(
                    f"<tr><td class='spell'>{i['word']}</td>"
                    f"<td class='sugg'>{', '.join(i['suggestions'][:4]) or '—'}</td></tr>"
                    for i in s_issues
                )
                spell_html = f"""
                <div class="section-label">🔡 Spell Check</div>
                <table class="issue-table">
                  <thead><tr><th>Misspelled</th><th>Suggestions</th></tr></thead>
                  <tbody>{rows}</tbody>
                </table>"""

            llm_html = ""
            if llm_skip:
                llm_html = f"<div class='section-label' style='color:#fbbf24'>⏭ LLM skipped — {l_result.get('reason','timeout')}</div>"
            elif l_result.get("error"):
                llm_html = f"<div class='section-label' style='color:#f87171'>⚠ LLM error: {l_result['error']}</div>"
            elif llm_bad and l_result.get("issues"):
                def _issue_row(i: dict) -> str:
                    t = i.get("type", "spelling")
                    return (
                        f"<tr><td class='type-{t}'>{t.title()}</td>"
                        f"<td class='original'>{i.get('original','')}</td>"
                        f"<td class='suggestion'>{i.get('suggestion','')}</td>"
                        f"<td style='color:#94a3b8;font-size:.8rem'>{i.get('explanation','')}</td></tr>"
                    )
                issue_rows = "".join(_issue_row(i) for i in l_result["issues"])
                corrected = l_result.get("corrected_text", "")
                fix = f'<div class="fixed-box">✔ {corrected}</div>' if corrected else ""
                llm_html = f"""
                <div class="section-label">🤖 LLM Analysis</div>
                <table class="issue-table">
                  <thead><tr><th>Type</th><th>Original</th><th>Suggestion</th><th>Explanation</th></tr></thead>
                  <tbody>{issue_rows}</tbody>
                </table>{fix}"""

            preview = value[:120] + ("…" if len(value) > 120 else "")
            fields_html += f"""
            <details>
              <summary>
                <span class="flag">✘</span>
                <code style="color:#a5b4fc">{field}</code>
                {badges}
              </summary>
              <div class="detail-body">
                <div class="preview">"{preview}"</div>
                {spell_html}
                {llm_html}
              </div>
            </details>"""

        n_flag = stats["flagged"]
        file_sections_html += f"""
        <div class="file-section">
          <div class="file-header">
            <span class="file-name">📄 {fname}</span>
            <span class="file-badge">{n_flag} issue{'s' if n_flag != 1 else ''}</span>
          </div>
          {fields_html}
        </div>"""

    # ── Assemble ─────────────────────────────────────────────────────────────
    skipped_card = (
        f'<div class="card skipped"><div class="label">⏭ LLM Skipped</div>'
        f'<div class="value">{skipped}</div></div>'
    ) if skipped else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Spell Check Report — {generated_at}</title>
  <style>{css}</style>
</head>
<body>
  <h1>JSON Spell Check Report</h1>
  <div class="meta">Generated: {generated_at} &nbsp;·&nbsp; Model: {model}</div>

  <div class="cards">
    <div class="card total"><div class="label">Files</div>
      <div class="value">{len(file_stats)}</div></div>
    <div class="card total"><div class="label">Total Fields</div>
      <div class="value">{total}</div></div>
    <div class="card flagged"><div class="label">⚠ Flagged</div>
      <div class="value">{flagged}</div></div>
    <div class="card clean"><div class="label">✓ Clean</div>
      <div class="value">{clean}</div></div>
    {skipped_card}
  </div>

  <h2>🏆 Top Affected Files</h2>
  <table>
    <thead>
      <tr><th>#</th><th>File</th><th>Flagged</th><th>Total</th><th>Error Rate</th></tr>
    </thead>
    <tbody>{top_rows_html}</tbody>
  </table>

  <h2>📋 Field Details (flagged only)</h2>
  {file_sections_html}

  <footer>
    JSON Spell Checker &nbsp;·&nbsp; {generated_at}
  </footer>
</body>
</html>"""

    return html
