# pipeline.py  — robust research synthesis engine
#
# Core design principles:
#   1. Evidence-grounded   — every output is traceable to specific claim IDs
#   2. Falsifiability-first — hypotheses are validated against forbidden language
#   3. Self-improving      — adversarial review feeds a refinement loop (up to N rounds)
#   4. Calibrated scores   — confidence recalculated from review weakness density
#   5. Speed               — batched extraction, parallel synthesis, thread-local LLMs

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pdfplumber
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from tqdm import tqdm

from schema import (
    Agreement, Claim, Contradiction, Gap,
    Hypothesis, LiteratureSynthesis, Review,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GemmaSynthesis")

# ── Tunable constants ──────────────────────────────────────────────────
CHECKPOINT_DIR      = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE           = 8_000
EXTRACT_BATCH_SIZE   = 3
MAX_EXTRACT_WORKERS  = 6
MAX_SYNTH_WORKERS    = 4
MAX_CLAIM_CHARS      = 80_000
MIN_GROUP_SIZE       = 2
MAX_REFINEMENT_ROUNDS = 2   # adversarial → refine loop iterations
MIN_GROUNDING_REFS   = 2    # min valid claim IDs in hypothesis.based_on

_MODEL_NAME  = "gemma4"
_TEMPERATURE = 0.0

# Forbidden absolute language (validated in schema AND enforced in prompts)
FORBIDDEN_PHRASES = [
    "necessary and sufficient", "proves", "definitive proof",
    "objective metric", "always true", "never fails",
    "impossible to", "certainly", "guaranteed to",
]


# ── Streaming callback ─────────────────────────────────────────────────
class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


# ── Thread-local LLM pool ──────────────────────────────────────────────
_thread_local = threading.local()

def _get_llm(streaming: bool, thinking: bool) -> ChatOllama:
    key = f"llm_{int(streaming)}_{int(thinking)}"
    if not hasattr(_thread_local, key):
        setattr(_thread_local, key, ChatOllama(
            model=_MODEL_NAME,
            temperature=_TEMPERATURE,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else [],
            model_kwargs={"think": thinking},
        ))
    return getattr(_thread_local, key)


# ── Core LLM helper ────────────────────────────────────────────────────
_END_TOKENS = ["<eos>", "<end_of_turn>", "<|eot_id|>", "<|end|>"]
_THINK_RE   = re.compile(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", re.IGNORECASE)
_SYSTEM_MSG = SystemMessage(content=(
    "You are a rigorous scientific reasoning assistant. "
    "Output ONLY valid JSON — no prose, no markdown fences, no special tokens. "
    "Every claim you produce must be falsifiable and scoped. "
    "Never use absolute language such as: " + ", ".join(f'"{p}"' for p in FORBIDDEN_PHRASES)
))

def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    for tok in _END_TOKENS:
        if tok in text:
            text = text.split(tok)[0]
    return re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.DOTALL).strip()

def _parse_json(text: str) -> Any:
    cleaned = _clean(text)
    start = next((i for i, c in enumerate(cleaned) if c in "{["), None)
    if start is None:
        raise ValueError("No JSON start character found")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    return obj

def invoke_json(
    prompt: str,
    *,
    streaming: bool = False,
    thinking: bool  = True,
    retries: int    = 3,
) -> Optional[Any]:
    last: Exception = RuntimeError("no attempts made")
    for attempt in range(retries + 1):
        try:
            llm = _get_llm(streaming=streaming, thinking=thinking)
            raw = llm.invoke([_SYSTEM_MSG, HumanMessage(content=prompt)]).content
            return _parse_json(raw)
        except Exception as exc:
            last = exc
            if attempt < retries:
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt+1}/{retries+1} failed ({exc}). Retry in {wait}s…")
                time.sleep(wait)
    logger.error(f"All retries exhausted: {last}")
    return None


# ── Checkpoint helpers ─────────────────────────────────────────────────
def compute_input_hash(pdf_paths: List[Path]) -> str:
    h = hashlib.md5()
    for p in sorted(pdf_paths, key=lambda x: x.name):
        h.update(p.name.encode())
        s = p.stat()
        h.update(str(s.st_size).encode()); h.update(str(int(s.st_mtime)).encode())
    return h.hexdigest()

def clear_stale_checkpoints(pdf_paths: List[Path]) -> None:
    mp = CHECKPOINT_DIR / "_manifest.json"
    stored  = json.loads(mp.read_text()).get("input_hash") if mp.exists() else None
    current = compute_input_hash(pdf_paths)
    if stored == current:
        return
    logger.info("Input changed — invalidating checkpoints")
    for f in CHECKPOINT_DIR.glob("*.json"):
        if f.name != "_manifest.json":
            f.unlink()
    mp.write_text(json.dumps({"input_hash": current, "cleared_at": datetime.utcnow().isoformat()}))

def save_checkpoint(data: Any, name: str) -> None:
    (CHECKPOINT_DIR / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"✓ checkpoint/{name}.json")

def load_checkpoint(name: str) -> Optional[Any]:
    p = CHECKPOINT_DIR / f"{name}.json"
    if p.exists():
        logger.info(f"↩ checkpoint/{name}.json")
        return json.loads(p.read_text())
    return None


# ── Helpers ────────────────────────────────────────────────────────────
def truncate_to_budget(items: list, max_chars: int = MAX_CLAIM_CHARS) -> list:
    result, total = [], 0
    for item in items:
        blob = json.dumps(item if isinstance(item, dict) else item.model_dump(), default=str)
        if total + len(blob) > max_chars:
            logger.warning(f"Budget cap: {len(result)}/{len(items)} items used")
            break
        result.append(item); total += len(blob)
    return result

def _ids(items) -> str:
    """Compact claim ID list for prompt injection."""
    return json.dumps([getattr(i, "id", i) for i in items])


# ── PDF loading (parallel) ─────────────────────────────────────────────
def _load_one_pdf(args) -> List[Dict]:
    idx, path, chunk_size = args
    pid = f"paper_{idx}"
    try:
        text = ""
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                logger.warning(f"{path.name}: no pages"); return []
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        if not text.strip():
            logger.warning(f"{path.name}: empty"); return []
        chunks = [
            {"paper_id": pid, "chunk_id": i, "text": c}
            for i, c in enumerate(text[j:j+chunk_size] for j in range(0, len(text), chunk_size))
            if c.strip()
        ]
        logger.info(f"  {path.name} → {pid} ({len(chunks)} chunks)")
        return chunks
    except Exception as exc:
        logger.error(f"Load failed {path.name}: {exc}"); return []

def load_pdfs(paths: List[Path], chunk_size: int = CHUNK_SIZE) -> List[Dict]:
    logger.info(f"Loading {len(paths)} PDFs in parallel…")
    with ThreadPoolExecutor(max_workers=min(len(paths), 4)) as ex:
        results = list(ex.map(_load_one_pdf, [(i, p, chunk_size) for i, p in enumerate(paths)]))
    docs = [d for batch in results for d in batch]
    logger.info(f"Total chunks: {len(docs)}")
    return docs


# ── Claim extraction — batched + parallel ──────────────────────────────
def _extract_batch(batch: List[Dict]) -> List[Claim]:
    sections = "\n\n".join(
        f"[paper_id={d['paper_id']} chunk_id={d['chunk_id']}]\n{d['text']}"
        for d in batch
    )
    prompt = f"""Extract up to 4 specific, falsifiable scientific claims from EACH section.

Return a flat JSON list. Each object:
  "id":       unique string e.g. "{{paper_id}}_ck{{chunk_id}}_c0"
  "paper_id": copy the paper_id from the section header exactly
  "claim":    precise, scoped statement with measurable terms where possible
  "method":   experimental method or null
  "dataset":  dataset name or null
  "metric":   metric name or null
  "value":    reported value or null

RULES:
- Claims must be specific to one paper — no synthesis across sections
- Omit vague claims like "results show improvement"
- Preserve numerical values exactly as written

{sections[:12_000]}"""
    data = invoke_json(prompt, streaming=True, thinking=False) or []
    if isinstance(data, dict): data = [data]
    claims = []
    for item in data:
        if not isinstance(item, dict): continue
        pid = item.get("paper_id") or batch[0]["paper_id"]
        item["paper_id"] = pid
        item.setdefault("id", f"{pid}_auto_{len(claims)}")
        for f in ("method", "dataset", "metric", "value"):
            item.setdefault(f, None)
        try:
            claims.append(Claim(**item))
        except Exception as exc:
            logger.warning(f"Bad claim skipped: {exc}")
    return claims

def extract_all_claims(
    docs: List[Dict],
    max_workers: int = MAX_EXTRACT_WORKERS,
    progress_callback: Optional[Callable] = None,
) -> List[Claim]:
    batches = [docs[i:i+EXTRACT_BATCH_SIZE] for i in range(0, len(docs), EXTRACT_BATCH_SIZE)]
    logger.info(f"Extraction: {len(docs)} chunks → {len(batches)} batches (workers={max_workers})")
    all_claims: List[Claim] = []
    done, lock = 0, threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_extract_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            try:
                all_claims.extend(fut.result())
            except Exception as exc:
                logger.error(f"Batch failed: {exc}")
            finally:
                with lock:
                    done += 1
                    if progress_callback:
                        progress_callback(0.10 + 0.20 * done / len(futures),
                                          f"Batch {done}/{len(futures)}")
    logger.info(f"Claims extracted: {len(all_claims)}")
    return all_claims


# ── Agreements ─────────────────────────────────────────────────────────
def identify_agreements(claims: List[Claim]) -> List[Agreement]:
    logger.info(f"Agreements ({len(claims)} claims)…")
    if not claims: return []
    sample = truncate_to_budget(claims)
    data = invoke_json(f"""Identify explicit agreements — findings that two or more papers support
with convergent evidence. Do NOT invent agreements; only report clearly evidenced ones.

Return JSON list. Each object:
  "statement":         the agreed-upon finding (scoped, not absolute)
  "supporting_papers": list of paper_ids
  "evidence":          list of specific claim IDs from the input that support this
  "confidence":        float 0.0–1.0 (reflect actual evidence strength)

If no genuine agreements exist, return [].

CLAIM POOL:
{json.dumps([c.model_dump() for c in sample], indent=2)}""", thinking=True) or []
    if isinstance(data, dict): data = [data]
    out = []
    for item in data:
        item.setdefault("evidence", [])
        try: out.append(Agreement(**item))
        except Exception as exc: logger.warning(f"Bad agreement: {exc}")
    logger.info(f"Agreements: {len(out)}")
    return out


# ── Contradictions — parallel groups, richer output ────────────────────
def _group_by_method(claims: List[Claim]) -> Dict[str, List[Claim]]:
    raw: Dict[str, List[Claim]] = {}
    for c in claims:
        key = (c.method or "general").strip().lower()[:60]
        raw.setdefault(key, []).append(c)
    merged: Dict[str, List[Claim]] = {}
    overflow: List[Claim] = []
    for k, v in raw.items():
        if len(v) >= MIN_GROUP_SIZE: merged[k] = v
        else: overflow.extend(v)
    if len(overflow) >= MIN_GROUP_SIZE:
        merged.setdefault("general", []).extend(overflow)
    return merged

def _check_group(topic: str, group: List[Claim]) -> List[Contradiction]:
    safe = truncate_to_budget(group, max_chars=20_000)
    data = invoke_json(f"""Topic: {topic}

Identify DIRECT contradictions between these claims — only where one claim
explicitly conflicts with another on the same measurement, phenomenon, or conclusion.

Return JSON list. Each object:
  "issue":      what specifically is contradicted (name the variable/metric)
  "paper_a":    paper_id of first claim
  "paper_b":    paper_id of second claim
  "claim_a":    exact text of the contradicting claim from paper_a
  "claim_b":    exact text of the contradicting claim from paper_b
  "reason":     mechanistic explanation of why they conflict
  "resolution": possible reconciliation (different populations, metrics, conditions) or null
  "confidence": float 0.0–1.0

Do NOT manufacture contradictions. If none exist, return [].

CLAIMS:
{json.dumps([c.model_dump() for c in safe], indent=2)}""", thinking=True) or []
    if isinstance(data, dict): data = [data]
    out = []
    for item in data:
        item.setdefault("claim_a", ""); item.setdefault("claim_b", "")
        item.setdefault("resolution", None)
        try: out.append(Contradiction(**item))
        except Exception as exc: logger.warning(f"Bad contradiction in '{topic}': {exc}")
    return out

def identify_contradictions(claims: List[Claim]) -> List[Contradiction]:
    logger.info(f"Contradictions ({len(claims)} claims)…")
    if not claims: return []
    groups = _group_by_method(claims)
    logger.info(f"Clusters: {list(groups.keys())}")
    out: List[Contradiction] = []
    with ThreadPoolExecutor(max_workers=min(len(groups), MAX_SYNTH_WORKERS)) as ex:
        futures = {ex.submit(_check_group, t, g): t for t, g in groups.items()}
        for fut in as_completed(futures):
            try:
                found = fut.result(); out.extend(found)
                logger.debug(f"  '{futures[fut]}': {len(found)}")
            except Exception as exc:
                logger.error(f"Group '{futures[fut]}' failed: {exc}")
    logger.info(f"Contradictions: {len(out)}")
    return out


# ── Gaps — importance-ranked, causally linked ──────────────────────────
def identify_gaps(claims: List[Claim], contradictions: List[Contradiction]) -> List[Gap]:
    logger.info("Gaps…")
    if not claims: return []
    data = invoke_json(f"""Identify the most important *specific* gaps in the literature.
A gap is a question that:
  (a) is directly implied by the evidence or contradictions below, AND
  (b) no paper in the pool answers

For each gap, trace it to the claims or contradictions that reveal it.

Return JSON list ordered by importance (critical first). Each object:
  "description": precise statement of the missing knowledge
  "caused_by":   list of claim IDs or contradiction descriptions that expose this gap
  "importance":  one of "critical" | "high" | "medium" | "low"
  "confidence":  float 0.0–1.0

CLAIMS:
{json.dumps([c.model_dump() for c in truncate_to_budget(claims, 30_000)], indent=2)}

CONTRADICTIONS:
{json.dumps([c.model_dump() for c in truncate_to_budget(contradictions, 10_000)], indent=2)}""",
    thinking=True) or []
    if isinstance(data, dict): data = [data]
    out = []
    for item in data:
        item.setdefault("caused_by", [])
        item.setdefault("importance", "medium")
        try: out.append(Gap(**item))
        except Exception as exc: logger.warning(f"Bad gap: {exc}")
    logger.info(f"Gaps: {len(out)}")
    return out


# ── Hypothesis generation — falsifiable, grounded, scoped ─────────────
_HYP_RULES = f"""MANDATORY RULES — violating any rule means the hypothesis is REJECTED:
1. SCOPE     — state the specific condition, population, or model class this applies to
2. FALSIFY   — include a null hypothesis that experiments should reject
3. MECHANISM — name the specific signal/process, its origin (layer/module/stage),
               and the downstream effect it produces
4. GROUNDING — based_on must list ≥{MIN_GROUNDING_REFS} specific claim IDs from the CLAIM POOL
5. TRACING   — gap_addressed must be a valid gap ID from the GAP LIST
6. LANGUAGE  — forbidden phrases: {', '.join(f'"{p}"' for p in FORBIDDEN_PHRASES)}

BAD EXAMPLE (rejected — vague mechanism, absolute language):
  "Storing knowledge in the KV cache is necessary and sufficient for C1."

GOOD EXAMPLE (accepted — scoped, mechanistic, falsifiable):
  "In decoder-only LLMs with ≥7B parameters, injecting domain-specific embeddings
   into KV cache positions 0–32 will reduce hallucination rate on closed-domain QA
   by ≥15% compared to prompt-only injection, because early-layer cache slots
   function as high-priority retrieval anchors for attention heads in layers 8–16."
"""

def generate_hypotheses(
    claims: List[Claim],
    contradictions: List[Contradiction],
    gaps: List[Gap],
) -> List[Hypothesis]:
    logger.info("Generating hypotheses…")
    valid_ids  = {c.id for c in claims}
    gap_map    = {g.id: g.description for g in gaps}

    data = invoke_json(f"""Generate 3 novel, testable scientific hypotheses from this synthesis.
Each hypothesis must emerge from combining evidence across multiple papers — not restate a single paper.

{_HYP_RULES}

Return ONLY a JSON list of 3 objects, each with these exact fields:
  "hypothesis":       str
  "based_on":         [str]   — claim IDs (≥{MIN_GROUNDING_REFS} from CLAIM POOL)
  "gap_addressed":    str     — gap ID from GAP LIST
  "mechanism":        str     — specific signal → layer/module → downstream effect
  "mechanism_type":   str     — one of: architectural | training | data | emergent | hybrid
  "null_hypothesis":  str     — falsifiable null
  "experiment":       str     — (1) IV, (2) control, (3) measurements, (4) statistical test
  "expected_outcome": str
  "confidence":       float 0.0–1.0

CLAIM POOL (use IDs in based_on):
{json.dumps([{{"id":c.id,"paper":c.paper_id,"claim":c.claim}} for c in claims[:40]], indent=2)}

GAP LIST (use IDs in gap_addressed):
{json.dumps([{{"id":g.id,"description":g.description}} for g in gaps], indent=2)}

CONTRADICTIONS (use for hypothesis inspiration):
{json.dumps([{{"issue":c.issue,"resolution":c.resolution}} for c in contradictions], indent=2)}""",
    thinking=True) or []

    if isinstance(data, dict): data = [data]
    out = []
    for item in data:
        if not isinstance(item, dict): continue
        item.setdefault("based_on", [])
        item.setdefault("gap_addressed", None)
        item.setdefault("mechanism_type", "emergent")
        item.setdefault("null_hypothesis", "No significant difference will be observed.")
        item.setdefault("experiment", "To be designed")
        item.setdefault("expected_outcome", "To be determined")
        item.setdefault("confidence", 0.5)
        item.setdefault("revision", 0)

        # Validate grounding
        grounded = [r for r in item.get("based_on", []) if r in valid_ids]
        if len(grounded) < MIN_GROUNDING_REFS:
            logger.warning(f"Hypothesis under-grounded ({len(grounded)} valid refs) — keeping with warning")
        item["based_on"] = grounded or item["based_on"]

        # Validate gap reference
        if item.get("gap_addressed") and item["gap_addressed"] not in gap_map:
            logger.warning(f"gap_addressed '{item['gap_addressed']}' not in gap list — clearing")
            item["gap_addressed"] = None

        try:
            out.append(Hypothesis(**item))
        except ValueError as exc:
            logger.warning(f"Hypothesis rejected by schema validator: {exc}")
    logger.info(f"Hypotheses generated: {len(out)}")
    return out


# ── Adversarial review — structured, severity-scored ──────────────────
def _review_one(h: Hypothesis, round_num: int = 1) -> Review:
    data = invoke_json(f"""You are a harsh but constructive peer reviewer (round {round_num}).
Your job is to find every flaw, then produce an improved version if salvageable.

EVALUATE these dimensions:
  1. SCOPE     — is the hypothesis too broad or too narrow?
  2. MECHANISM — is the signal, layer, and effect specified concretely?
  3. FALSIFIABILITY — can the null_hypothesis actually be tested?
  4. GROUNDING — does it genuinely follow from the cited evidence?
  5. EXPERIMENT — does it have IV, control, measurements, and a statistical test?
  6. LANGUAGE  — does it avoid absolute claims?

Return JSON:
  "hypothesis_id":      "{h.id}"
  "round":              {round_num}
  "critique":           str (detailed, dimension-by-dimension)
  "weaknesses":         [str]  (one entry per specific flaw)
  "fatal_flaw":         bool   (true only if hypothesis is fundamentally incoherent)
  "improved_hypothesis": str   (rewritten hypothesis text addressing all weaknesses, or null if fatal)
  "confidence_penalty": float  (0.0–1.0, proportional to weakness count and severity)

HYPOTHESIS:
{json.dumps(h.model_dump(), indent=2)}""", thinking=True) or {}

    data.setdefault("hypothesis_id", h.id)
    data.setdefault("round", round_num)
    data.setdefault("critique", "Review unavailable")
    data.setdefault("weaknesses", [])
    data.setdefault("fatal_flaw", False)
    data.setdefault("improved_hypothesis", None)
    data.setdefault("confidence_penalty", min(0.05 * len(data.get("weaknesses", [])), 0.5))
    try:
        return Review(**data)
    except Exception:
        return Review(hypothesis_id=h.id, round=round_num,
                      critique="Review failed", weaknesses=[])

def adversarial_review(hypotheses: List[Hypothesis], round_num: int = 1) -> List[Review]:
    if not hypotheses: return []
    logger.info(f"Adversarial review round {round_num} (×{len(hypotheses)} parallel)…")
    with ThreadPoolExecutor(max_workers=len(hypotheses)) as ex:
        reviews = list(ex.map(lambda h: _review_one(h, round_num), hypotheses))
    return reviews


# ── Confidence recalibration ───────────────────────────────────────────
def recalibrate_confidence(h: Hypothesis, r: Review) -> Hypothesis:
    """Reduce confidence proportional to weakness count + reviewer penalty."""
    density_penalty = 0.06 * len(r.weaknesses)
    total_penalty   = min(density_penalty + r.confidence_penalty, 0.6)
    new_conf        = round(max(0.05, h.confidence - total_penalty), 3)
    logger.debug(f"  {h.id}: conf {h.confidence:.2f} → {new_conf:.2f} "
                 f"({len(r.weaknesses)} weaknesses, penalty={r.confidence_penalty:.2f})")
    return h.model_copy(update={"confidence": new_conf})


# ── Refinement loop ────────────────────────────────────────────────────
def refine_hypotheses(
    hypotheses: List[Hypothesis],
    claims: List[Claim],
    gaps: List[Gap],
    max_rounds: int = MAX_REFINEMENT_ROUNDS,
) -> Tuple[List[Hypothesis], List[Hypothesis], int]:
    """
    Run adversarial review → improve → re-review, up to max_rounds.
    Returns (accepted, discarded, rounds_completed).
    """
    valid_ids = {c.id for c in claims}
    gap_ids   = {g.id for g in gaps}
    current   = list(hypotheses)
    discarded: List[Hypothesis] = []
    rounds_done = 0

    for round_num in range(1, max_rounds + 1):
        if not current:
            break
        rounds_done = round_num
        logger.info(f"── Refinement round {round_num}/{max_rounds} ──")
        reviews = adversarial_review(current, round_num=round_num)
        next_round: List[Hypothesis] = []

        for h, r in zip(current, reviews):
            # 1. Discard fatally flawed hypotheses
            if r.fatal_flaw:
                logger.warning(f"  ✗ {h.id} DISCARDED (fatal flaw): {r.critique[:80]}")
                discarded.append(h)
                continue

            # 2. Recalibrate confidence based on review
            h = recalibrate_confidence(h, r)

            # 3. Attempt to improve if reviewer provided a rewrite
            if r.improved_hypothesis:
                improved_text = r.improved_hypothesis.strip()
                # Reject improvement if it still contains forbidden language
                still_bad = [p for p in FORBIDDEN_PHRASES if p in improved_text.lower()]
                if still_bad:
                    logger.warning(f"  ⚠ {h.id} improvement still contains: {still_bad}")
                    next_round.append(h)
                    continue

                candidate = h.model_copy(update={
                    "hypothesis": improved_text,
                    "revision":   h.revision + 1,
                })
                # Quick re-review to see if it's actually better
                check = _review_one(candidate, round_num=round_num)
                if len(check.weaknesses) < len(r.weaknesses):
                    logger.info(f"  ✓ {h.id} improved "
                                f"({len(r.weaknesses)} → {len(check.weaknesses)} weaknesses, "
                                f"revision {candidate.revision})")
                    candidate = recalibrate_confidence(candidate, check)
                    next_round.append(candidate)
                else:
                    logger.info(f"  ~ {h.id} improvement not better — keeping original")
                    next_round.append(h)
            else:
                next_round.append(h)

        current = next_round
        if all(len(_review_one(h, round_num).weaknesses) == 0 for h in current):
            logger.info(f"All hypotheses passed review at round {round_num} — stopping early")
            break

    return current, discarded, rounds_done


# ── Main pipeline ──────────────────────────────────────────────────────
def run_literature_synthesis(
    pdf_paths: List[Path],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> LiteratureSynthesis:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"PIPELINE START | PDFs: {[p.name for p in pdf_paths]}")
    logger.info("=" * 60)

    def _prog(pct, msg):
        logger.info(f"[{pct*100:.0f}%] {msg}")
        if progress_callback: progress_callback(pct, msg)

    _prog(0.0, "Initializing…")
    clear_stale_checkpoints(pdf_paths)

    # 1. PDFs
    _prog(0.02, "Loading PDFs…")
    docs = load_pdfs(pdf_paths)
    save_checkpoint(docs, "01_docs")
    _prog(0.08, f"Chunks: {len(docs)}")

    # 2. Claims
    raw = load_checkpoint("02_claims")
    if raw is None:
        _prog(0.10, "Extracting claims (batched, parallel)…")
        claims = extract_all_claims(docs, progress_callback=progress_callback)
        save_checkpoint([c.model_dump() for c in claims], "02_claims")
    else:
        claims = [Claim(**c) for c in raw]
    _prog(0.32, f"Claims: {len(claims)}")

    # 3 + 4. Agreements ∥ contradictions (independent, run in parallel)
    raw_agr = load_checkpoint("03_agreements")
    raw_con = load_checkpoint("04_contradictions")
    if raw_agr is None or raw_con is None:
        _prog(0.34, "Agreements ∥ contradictions…")
        with ThreadPoolExecutor(max_workers=2) as ex:
            agr_fut = ex.submit(identify_agreements,     claims) if raw_agr is None else None
            con_fut = ex.submit(identify_contradictions, claims) if raw_con is None else None
            agreements     = agr_fut.result() if agr_fut else [Agreement(**a)     for a in raw_agr]
            contradictions = con_fut.result() if con_fut else [Contradiction(**c) for c in raw_con]
        if raw_agr is None: save_checkpoint([a.model_dump() for a in agreements],     "03_agreements")
        if raw_con is None: save_checkpoint([c.model_dump() for c in contradictions], "04_contradictions")
    else:
        agreements     = [Agreement(**a)     for a in raw_agr]
        contradictions = [Contradiction(**c) for c in raw_con]
    _prog(0.62, f"Agreements: {len(agreements)} | Contradictions: {len(contradictions)}")

    # 5. Gaps
    raw = load_checkpoint("05_gaps")
    if raw is None:
        _prog(0.64, "Identifying gaps…")
        gaps = identify_gaps(claims, contradictions)
        save_checkpoint([g.model_dump() for g in gaps], "05_gaps")
    else:
        gaps = [Gap(**g) for g in raw]
    _prog(0.74, f"Gaps: {len(gaps)}")

    # 6. Hypotheses (initial generation)
    raw = load_checkpoint("06_hypotheses")
    if raw is None:
        _prog(0.76, "Generating hypotheses…")
        hypotheses = generate_hypotheses(claims, contradictions, gaps)
        save_checkpoint([h.model_dump() for h in hypotheses], "06_hypotheses")
    else:
        hypotheses = [Hypothesis(**h) for h in raw]
    _prog(0.82, f"Hypotheses (initial): {len(hypotheses)}")

    # 7. Refinement loop — adversarial review + iterative improvement
    raw_ref = load_checkpoint("07_refined")
    raw_dis = load_checkpoint("07_discarded")
    if raw_ref is None:
        _prog(0.84, f"Refinement loop (up to {MAX_REFINEMENT_ROUNDS} rounds)…")
        accepted, discarded, rounds = refine_hypotheses(hypotheses, claims, gaps)
        save_checkpoint([h.model_dump() for h in accepted],  "07_refined")
        save_checkpoint([h.model_dump() for h in discarded], "07_discarded")
    else:
        accepted   = [Hypothesis(**h) for h in raw_ref]
        discarded  = [Hypothesis(**h) for h in (raw_dis or [])]
        rounds     = MAX_REFINEMENT_ROUNDS  # conservative assumption from cache
    _prog(0.96, f"Accepted: {len(accepted)} | Discarded: {len(discarded)} | Rounds: {rounds}")

    # 8. Assemble and save
    paper_ids = sorted({d["paper_id"] for d in docs})
    synthesis = LiteratureSynthesis(
        papers_analyzed=paper_ids,
        total_claims=len(claims),
        agreements=agreements,
        contradictions=contradictions,
        gaps=gaps,
        hypotheses=accepted,
        discarded=discarded,
        refinement_rounds=rounds,
    )
    Path("synthesis_output.json").write_text(synthesis.model_dump_json(indent=2))

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        f"DONE in {elapsed:.1f}s | papers={len(paper_ids)} claims={len(claims)} "
        f"agreements={len(agreements)} contradictions={len(contradictions)} "
        f"gaps={len(gaps)} hypotheses={len(accepted)} discarded={len(discarded)} "
        f"refinement_rounds={rounds}"
    )
    logger.info("=" * 60)
    _prog(1.0, "Complete ✓")
    return synthesis


if __name__ == "__main__":
    import sys
    paths = list(Path("./papers").glob("*.pdf"))
    if not paths:
        print("No PDFs in ./papers/"); sys.exit(1)
    run_literature_synthesis(paths[:5])