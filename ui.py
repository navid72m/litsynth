# ui.py
import json
import logging
import queue
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from pipeline import CHECKPOINT_DIR, run_literature_synthesis
from schema import LiteratureSynthesis

# ── Thread-safe log capture ────────────────────────────────────────────
_log_queue: queue.Queue = queue.Queue()

class _QueueHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _log_queue.put(self.format(record))

def _setup_logging() -> None:
    logger = logging.getLogger("GemmaSynthesis")
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in [logging.StreamHandler(), _QueueHandler()]:
        h.setFormatter(fmt); logger.addHandler(h)
    logger.setLevel(logging.INFO)

_setup_logging()

def _drain_logs() -> str:
    lines = []
    while not _log_queue.empty():
        try: lines.append(_log_queue.get_nowait())
        except queue.Empty: break
    return "\n".join(lines)


# ── Rendering helpers ──────────────────────────────────────────────────
def _render_agreements(items) -> str:
    if not items: return "No agreements found."
    lines = []
    for a in items:
        bar = "█" * int(a.confidence * 10) + "░" * (10 - int(a.confidence * 10))
        lines.append(f"[{bar}] {a.confidence:.0%}\n{a.statement}")
        if a.evidence: lines.append(f"  Evidence: {', '.join(a.evidence)}")
        lines.append(f"  Papers:   {', '.join(a.supporting_papers)}\n")
    return "\n".join(lines)

def _render_contradictions(items) -> str:
    if not items: return "No contradictions found."
    lines = []
    for c in items:
        lines.append(f"ISSUE: {c.issue}")
        lines.append(f"  {c.paper_a}: {c.claim_a}")
        lines.append(f"  {c.paper_b}: {c.claim_b}")
        lines.append(f"  Why:    {c.reason}")
        if c.resolution: lines.append(f"  Reconcile: {c.resolution}")
        lines.append(f"  Conf:   {c.confidence:.0%}\n")
    return "\n".join(lines)

def _render_gaps(items) -> str:
    if not items: return "No gaps identified."
    emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    lines = []
    for g in items:
        e = emoji.get(g.importance, "⚪")
        lines.append(f"{e} [{g.importance.upper()}] {g.description}")
        if g.caused_by: lines.append(f"  Caused by: {', '.join(g.caused_by)}")
        lines.append(f"  Confidence: {g.confidence:.0%}\n")
    return "\n".join(lines)

def _render_hypotheses(items, label="ACCEPTED") -> str:
    if not items: return f"No {label.lower()} hypotheses."
    lines = []
    for i, h in enumerate(items, 1):
        bar = "█" * int(h.confidence * 10) + "░" * (10 - int(h.confidence * 10))
        lines.append(f"── H{i} [{label}] [{bar}] {h.confidence:.0%} (revision {h.revision}) ──")
        lines.append(f"HYPOTHESIS:    {h.hypothesis}")
        lines.append(f"NULL:          {h.null_hypothesis}")
        lines.append(f"MECHANISM:     [{h.mechanism_type}] {h.mechanism}")
        lines.append(f"EXPERIMENT:    {h.experiment}")
        lines.append(f"EXPECTED:      {h.expected_outcome}")
        lines.append(f"GROUNDED IN:   {', '.join(h.based_on) or 'N/A'}")
        lines.append(f"FILLS GAP:     {h.gap_addressed or 'N/A'}\n")
    return "\n".join(lines)

def _render_summary(s: LiteratureSynthesis) -> str:
    return (
        f"Papers:            {len(s.papers_analyzed)}\n"
        f"Claims extracted:  {s.total_claims}\n"
        f"Agreements:        {len(s.agreements)}\n"
        f"Contradictions:    {len(s.contradictions)}\n"
        f"Research gaps:     {len(s.gaps)}\n"
        f"Hypotheses:        {len(s.hypotheses)} accepted, {len(s.discarded)} discarded\n"
        f"Refinement rounds: {s.refinement_rounds}\n"
        f"Papers analyzed:   {', '.join(s.papers_analyzed)}"
    )


# ── Core runner ────────────────────────────────────────────────────────
def run_with_progress(pdf_files, progress=gr.Progress()):
    if not pdf_files:
        empty = "No files uploaded."
        return empty, empty, empty, empty, empty, empty, "[]"

    _drain_logs()
    temp_dir  = Path(tempfile.mkdtemp())
    pdf_paths = []
    for f in pdf_files:
        src  = Path(f.name) if hasattr(f, "name") else Path(f)
        dest = temp_dir / src.name
        shutil.copy(src, dest); pdf_paths.append(dest)

    try:
        result = run_literature_synthesis(pdf_paths, progress_callback=progress)
        logs   = _drain_logs()
        return (
            logs,
            _render_summary(result),
            _render_agreements(result.agreements),
            _render_contradictions(result.contradictions),
            _render_gaps(result.gaps),
            _render_hypotheses(result.hypotheses, "ACCEPTED")
            + ("\n\n" + _render_hypotheses(result.discarded, "DISCARDED") if result.discarded else ""),
            result.model_dump_json(indent=2),
        )
    except Exception as exc:
        logs = _drain_logs() + f"\n\n❌ FATAL: {exc}"
        return logs, "Error", "Error", "Error", "Error", "Error", "[]"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def clear_checkpoints() -> str:
    deleted = sum(1 for f in CHECKPOINT_DIR.glob("*.json") if not f.unlink())
    return f"✓ Cleared {deleted} checkpoint(s)."


# ── UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Gemma 4 — Research Synthesis Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Gemma 4 — Research Synthesis Engine")
    gr.Markdown(
        "Upload scientific PDFs. Gemma 4 extracts claims, finds agreements and contradictions, "
        "identifies research gaps, generates grounded hypotheses, then runs a multi-round "
        "adversarial refinement loop before presenting accepted results."
    )

    with gr.Row():
        file_input = gr.File(file_count="multiple", file_types=[".pdf"], label="Research papers (PDF)")
        with gr.Column(scale=0, min_width=200):
            start_btn    = gr.Button("▶  Synthesise", variant="primary",  scale=1)
            clear_btn    = gr.Button("🗑  Clear cache", variant="secondary", scale=1)
            clear_status = gr.Textbox(show_label=False, lines=1, interactive=False)

    logs_box = gr.Textbox(label="Pipeline logs", lines=10, interactive=False)

    with gr.Tabs():
        with gr.TabItem("📊 Summary"):
            summary_box = gr.Textbox(label="Run summary", lines=10, interactive=False)
        with gr.TabItem("🤝 Agreements"):
            agr_box = gr.Textbox(label="Agreements", lines=20, interactive=False)
        with gr.TabItem("⚡ Contradictions"):
            con_box = gr.Textbox(label="Contradictions", lines=20, interactive=False)
        with gr.TabItem("🔍 Research Gaps"):
            gap_box = gr.Textbox(label="Gaps (ranked by importance)", lines=20, interactive=False)
        with gr.TabItem("💡 Hypotheses"):
            hyp_box = gr.Textbox(label="Accepted & discarded hypotheses", lines=30, interactive=False)
        with gr.TabItem("📄 Raw JSON"):
            raw_box = gr.Textbox(label="synthesis_output.json", lines=30, interactive=False)

    start_btn.click(
        fn=run_with_progress,
        inputs=file_input,
        outputs=[logs_box, summary_box, agr_box, con_box, gap_box, hyp_box, raw_box],
    )
    clear_btn.click(fn=clear_checkpoints, inputs=[], outputs=[clear_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)