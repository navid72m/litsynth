# LitSynth 📚

**Local, offline research synthesis powered by Gemma 4's 128K context window and thinking mode.**

LitSynth reads up to 15 scientific PDFs simultaneously, reasons across all of them as a single evidence corpus, and produces four structured research outputs — agreements, contradictions, research gaps, and novel falsifiable hypotheses — each one refined through a multi-round adversarial peer review loop before it reaches you.

> Built for the [Gemma 4 Challenge — Build with Gemma 4](https://dev.to/challenges/google-gemma-2026-05-06)

---

## Why this exists

Individual papers answer narrow questions. The breakthroughs live in the gaps *between* them.

A standard RAG pipeline retrieves top-k chunks and reasons over those. The problem: cross-paper relationships fall *between* retrieval buckets. A finding in paper 3 that partially contradicts paper 11 is only visible if both are in context at the same time.

Gemma 4's 128K context window makes it possible to load an entire literature corpus at once. LitSynth is built around that capability.

---

**Example output — 10-paper run on transformer attention research:**
```
Papers:            10
Claims extracted:  214
Agreements:        6
Contradictions:    11  (across 5 method clusters)
Research gaps:     7   (2 critical, 3 high, 2 medium)
Hypotheses:        2 accepted, 1 discarded
Refinement rounds: 2
Runtime:           ~14 min  (MacBook M2 Pro, 32 GB, fully offline)
```

See [`results/example_synthesis.json`](results/example_synthesis.json) for the full output.

---

## How it works

LitSynth runs a seven-stage pipeline, fully locally:

```
PDF files
    │
    ▼
① Parallel PDF loader        (4 workers, pdfplumber)
    │
    ▼
② Batched claim extractor    (6 workers, 3 chunks/call, no thinking mode)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
③ Agreement identification    ④ Contradiction detection
  (single long-context call)    (parallel method clusters)
    │                                  │
    └─────────────────┬────────────────┘
                      ▼
              ⑤ Gap analysis
                (importance-ranked, causally linked to claims)
                      │
                      ▼
              ⑥ Hypothesis generation
                (grounded, falsifiable, schema-validated)
                      │
                      ▼
              ⑦ Adversarial refinement loop
                ┌─────────────────────────────┐
                │  Review all in parallel      │
                │  → Recalibrate confidence    │
                │  → Attempt improvement       │
                │  → Re-review candidate       │
                │  → Accept only if better     │  ← up to 2 rounds
                └─────────────────────────────┘
                      │
                      ▼
              LiteratureSynthesis
              (JSON + Gradio UI)
```

### What each stage produces

| Stage | Output |
|---|---|
| Claim extraction | Namespaced, falsifiable claims with method/dataset/metric/value |
| Agreements | Convergent findings with specific claim IDs as evidence |
| Contradictions | Conflicting claims with exact text, mechanistic reason, and proposed reconciliation |
| Gaps | Ranked open questions traced back to the claims/contradictions that reveal them |
| Hypotheses | Scoped, falsifiable statements with null hypothesis, mechanism, experiment design |
| Refinement | Accepted hypotheses with recalibrated confidence; discarded ones with critique logged |

---

## Quickstart

### Requirements
- macOS or Linux
- [Ollama](https://ollama.com) installed
- 32 GB RAM recommended for the 31B model (16 GB works with the 12B or E4B)
- Python 3.10+

### Install

```bash
git clone https://github.com/navid72m/litsynth.git
cd litsynth
pip install -r requirements.txt
```

### Pull the model

```bash
# Recommended — best synthesis quality
ollama pull gemma4

# Lighter option — fits in 16 GB RAM
ollama pull gemma4:12b
```

### Add your papers

```bash
mkdir papers
# copy your PDFs into papers/
cp ~/Downloads/*.pdf papers/
```

### Run

```bash
# Gradio UI (recommended)
python ui.py

# Headless / CLI
python pipeline.py
```

Open [http://localhost:7860](http://localhost:7860), upload your PDFs, click **Synthesise**.

---

## Configuration

Edit the constants at the top of `pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `8000` | PDF characters per chunk |
| `EXTRACT_BATCH_SIZE` | `3` | Chunks packed per extraction call |
| `MAX_EXTRACT_WORKERS` | `6` | Parallel extraction threads |
| `MAX_SYNTH_WORKERS` | `4` | Parallel synthesis threads |
| `MAX_REFINEMENT_ROUNDS` | `2` | Adversarial review iterations |
| `MIN_GROUNDING_REFS` | `2` | Min valid claim IDs per hypothesis |
| `MAX_CLAIM_CHARS` | `80000` | Token budget for synthesis prompts |

---

## Output schema

```json
{
  "papers_analyzed": ["paper_0", "paper_1", "..."],
  "total_claims": 214,
  "agreements": [
    {
      "statement": "...",
      "supporting_papers": ["paper_0", "paper_3"],
      "evidence": ["paper_0_ck1_c2", "paper_3_ck0_c0"],
      "confidence": 0.82
    }
  ],
  "contradictions": [
    {
      "issue": "...",
      "paper_a": "paper_1",
      "paper_b": "paper_4",
      "claim_a": "exact text...",
      "claim_b": "exact text...",
      "reason": "mechanistic explanation...",
      "resolution": "possible reconciliation...",
      "confidence": 0.74
    }
  ],
  "gaps": [
    {
      "id": "gap_3a8f2c",
      "description": "...",
      "caused_by": ["paper_2_ck1_c3", "paper_7_ck0_c1"],
      "importance": "critical",
      "confidence": 0.88
    }
  ],
  "hypotheses": [
    {
      "id": "H_f2a91b",
      "hypothesis": "scoped, falsifiable statement...",
      "null_hypothesis": "...",
      "mechanism": "specific signal → layer → downstream effect",
      "mechanism_type": "architectural",
      "experiment": "(1) IV, (2) control, (3) measurements, (4) statistical test",
      "expected_outcome": "...",
      "based_on": ["paper_2_ck1_c3", "paper_7_ck0_c1"],
      "gap_addressed": "gap_3a8f2c",
      "confidence": 0.61,
      "revision": 2
    }
  ],
  "discarded": [...],
  "refinement_rounds": 2
}
```

---

## Why Gemma 4 specifically

**128K context = the entire corpus in one call.**
Synthesis requires seeing all evidence simultaneously. A model with an 8K window would need RAG, which loses cross-paper relationships. Gemma 4's 128K window is not a nice-to-have — it is the architectural requirement that makes this approach possible.

**Thinking mode changes synthesis quality.**
Standard generation produces fluent but shallow hypotheses. Thinking mode produces hypotheses that trace intermediate reasoning steps explicitly. The adversarial reviewer benefits equally: dimension-by-dimension structured critiques rather than vague feedback.

**31B dense is the right model for this task.**
The E2B/E4B models are excellent for edge deployment. But holding 200+ claims in context and reasoning about relationships across them requires the full 31B. The task isn't latency-sensitive (a research session takes minutes), so reasoning quality justifies the compute. It runs locally on an M2 Pro with 32 GB RAM — no API, no data leaves the machine.

---

## Project structure

```
litsynth/
├── pipeline.py        # Seven-stage synthesis pipeline
├── schema.py          # Pydantic v2 models with validation
├── ui.py              # Gradio interface
├── requirements.txt
├── results/
│   └── example_synthesis.json   # Real output from a 10-paper run
└── papers/            # Put your PDFs here (gitignored)
```

---

## Robustness features

- **Checkpoint system** — each stage is saved to `checkpoints/`. A crash mid-run resumes from the last completed stage. Checkpoints are automatically invalidated when input PDFs change (MD5 hash of filename + size + mtime).
- **Thread-local LLM instances** — `ChatOllama` is not thread-safe. Each worker thread constructs its own instance to prevent race conditions.
- **Exponential back-off** — LLM calls retry with 1s, 2s, 4s delays before giving up.
- **Token budget guard** — synthesis prompts are truncated to `MAX_CLAIM_CHARS` before sending to prevent context overflow.
- **Schema-level guardrails** — the `Hypothesis` validator rejects absolute language (`"necessary and sufficient"`, `"proves"`, `"objective metric"`) at the Pydantic level, not just the prompt level.
- **Confidence recalibration** — LLM-assigned confidence scores are adjusted after review: `conf − (0.06 × weaknesses) − reviewer_penalty`. Scores are earned, not hallucinated.

---

## License

Apache 2.0 — same as Gemma 4.
