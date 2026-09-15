"""Strict boundary between a coding model and the research supervisor.

The model may suggest source and a permitted launch configuration.  This module
constructs the immutable candidate identity itself, so a model never supplies
task digests, source digests, or a pass/fail result.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .contracts import CandidateSpec
from .integrity import canonical


def system_prompt() -> str:
    return """You are a CUDA and Triton kernel research agent. Propose one implementation for the
frozen task. Treat all task equations, shape cases, exactness requirements, and launch limits as
non-negotiable. Learn only from the supplied measurement history. Do not claim a benchmark result,
change the task, add host code, use dynamic allocation, or include prose/code fences. Return one JSON
object with exactly: backend, family, parameters, hypothesis, source, parent_ids. The source must be
complete CUDA or Triton source compatible with the requested backend."""


def user_prompt(request: dict[str, Any]) -> str:
    """Serialize only frozen public research context for a provider request."""
    required = {"task", "observations", "candidate_schema"}
    if set(request) != required:
        raise ValueError("invalid agent request")
    return json.dumps({
        "task": request["task"],
        "observations": request["observations"],
        "candidate_contract": request["candidate_schema"],
        "response_contract": {
            "backend": "cuda or triton",
            "family": "lowercase identifier",
            "parameters": "must match CandidateSpec exactly",
            "hypothesis": "specific, falsifiable and at least 20 characters",
            "source": "complete source only",
            "parent_ids": "array of prior candidate ids, or []",
        },
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_from_response(request: dict[str, Any], response: str) -> tuple[CandidateSpec, bytes]:
    """Validate a model response and derive all supervisor-owned identities."""
    try:
        proposal = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("model response is not JSON") from error
    required = {"backend", "family", "parameters", "hypothesis", "source", "parent_ids"}
    if not isinstance(proposal, dict) or set(proposal) != required:
        raise ValueError("model response must contain exactly the proposal fields")
    if not isinstance(proposal["source"], str) or not proposal["source"].strip():
        raise ValueError("model response has no source")
    source = proposal["source"].encode()
    if len(source) > 1_000_000:
        raise ValueError("model source exceeded its resource limit")
    task = request["task"]
    candidate = CandidateSpec(
        task_digest=hashlib.sha256(canonical(task)).hexdigest(),
        backend=proposal["backend"], family=proposal["family"],
        source_digest=hashlib.sha256(source).hexdigest(),
        parameters=proposal["parameters"], hypothesis=proposal["hypothesis"],
        parent_ids=proposal["parent_ids"],
    )
    return candidate, source
