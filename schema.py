# schema.py
import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _clamp(v, default=0.5):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


# ── Primitive evidence unit ────────────────────────────────────────────
class Claim(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id:         str           = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    paper_id:   str
    claim:      str
    method:     Optional[str] = None
    dataset:    Optional[str] = None
    metric:     Optional[str] = None
    value:      Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v): return None if v is None else _clamp(v)


# ── Synthesis layer ────────────────────────────────────────────────────
class Agreement(BaseModel):
    statement:        str
    supporting_papers: List[str] = Field(default_factory=list)
    evidence:         List[str]  = Field(default_factory=list,
        description="Specific claim IDs or quotes that support this agreement")
    confidence:       float      = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v): return _clamp(v)


class Contradiction(BaseModel):
    issue:      str
    paper_a:    str
    paper_b:    str
    claim_a:    str  = Field(description="Exact claim from paper_a")
    claim_b:    str  = Field(description="Exact claim from paper_b")
    reason:     str  = "Not specified"
    resolution: Optional[str] = Field(None,
        description="Possible explanation for why both could be true (methodology diff, population diff, etc.)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v): return _clamp(v)


class Gap(BaseModel):
    id:          str   = Field(default_factory=lambda: f"gap_{str(uuid.uuid4())[:6]}")
    description: str
    caused_by:   List[str] = Field(default_factory=list,
        description="Contradiction IDs or claim IDs that reveal this gap")
    importance:  Literal["critical", "high", "medium", "low"] = "medium"
    confidence:  float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v): return _clamp(v)


# ── Hypothesis (rich, traceable, falsifiable) ──────────────────────────
class Hypothesis(BaseModel):
    id:               str  = Field(default_factory=lambda: f"H_{str(uuid.uuid4())[:6]}")
    hypothesis:       str  = Field(description="Scoped, falsifiable statement. No absolute language.")
    based_on:         List[str] = Field(default_factory=list,
        description="Claim IDs from the evidence base (min 2)")
    gap_addressed:    Optional[str] = Field(None,
        description="Gap ID this hypothesis resolves")
    mechanism:        str  = Field(description=(
        "Specific signal/process: name the signal, its origin layer/module, "
        "and what downstream effect it produces"))
    mechanism_type:   Literal["architectural", "training", "data", "emergent", "hybrid"] = "emergent"
    null_hypothesis:  str  = Field(description="The falsifiable null that experiments should reject")
    experiment:       str  = Field(description=(
        "Design with: (1) independent variable, (2) control condition, "
        "(3) measurements, (4) statistical test"))
    expected_outcome: str
    confidence:       float = Field(default=0.5, ge=0.0, le=1.0)
    revision:         int   = Field(default=0, description="How many refinement rounds this went through")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v): return _clamp(v)

    @model_validator(mode="after")
    def check_absolute_language(self):
        forbidden = [
            "necessary and sufficient", "proves", "definitive", "objective metric",
            "always", "never", "impossible", "certain", "guaranteed",
        ]
        text = (self.hypothesis + self.mechanism).lower()
        hits = [w for w in forbidden if w in text]
        if hits:
            raise ValueError(f"Hypothesis contains overreaching language: {hits}")
        return self


# ── Review (structured critique) ──────────────────────────────────────
class Review(BaseModel):
    hypothesis_id:       str
    round:               int  = 1
    critique:            str
    weaknesses:          List[str] = Field(default_factory=list)
    fatal_flaw:          bool = Field(default=False,
        description="True if the hypothesis should be discarded entirely")
    improved_hypothesis: Optional[str] = None
    confidence_penalty:  float = Field(default=0.0, ge=0.0, le=1.0,
        description="Suggested confidence reduction based on weakness severity")

    @field_validator("confidence_penalty", mode="before")
    @classmethod
    def clamp(cls, v): return _clamp(v)


# ── Top-level result ───────────────────────────────────────────────────
class LiteratureSynthesis(BaseModel):
    papers_analyzed:  List[str]
    total_claims:     int
    agreements:       List[Agreement]
    contradictions:   List[Contradiction]
    gaps:             List[Gap]
    hypotheses:       List[Hypothesis]
    discarded:        List[Hypothesis] = Field(default_factory=list,
        description="Hypotheses rejected after refinement")
    refinement_rounds: int = 0