from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .integrity import digest

SHA = r"^[a-f0-9]{64}$"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @property
    def identity(self):
        return digest(self.model_dump(mode="json"))


class NumericalPolicy(Contract):
    exact: bool = True
    max_absolute: FiniteFloat = Field(default=0.0, ge=0)
    max_relative_l2: FiniteFloat = Field(default=0.0, ge=0)
    matching_nonfinite: bool = True


class Case(Contract):
    rows: int = Field(ge=1, le=32768)
    columns: int = Field(ge=1, le=10752)
    column_stride: int = Field(default=1, ge=1, le=4)
    row_padding: int = Field(default=0, ge=0, le=128)
    stress: Literal["normal", "zero", "tiny", "outlier", "nonfinite", "rounding"] = "normal"
    epsilon: FiniteFloat = Field(default=1e-5, ge=0)

    @property
    def row_stride(self):
        return self.columns * self.column_stride + self.row_padding


class TaskSpec(Contract):
    schema_version: Literal[1] = 1
    operation: Literal["vector_add_f32", "fp8_rows_bf16"]
    equations: str = Field(min_length=20)
    cases: list[Case] = Field(min_length=1)
    numerical: NumericalPolicy = Field(default_factory=NumericalPolicy)
    reference_digest: str = Field(pattern=SHA)
    calls_per_update: int = Field(default=1, ge=1)
    measured_region_ms: FiniteFloat = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def supported(self):
        if self.operation == "fp8_rows_bf16":
            if any(case.columns % 128 for case in self.cases):
                raise ValueError("FP8 width must be 128-aligned")
            if not self.numerical.exact:
                raise ValueError("FP8 codes and scales require exact comparison")
        return self


class Budget(Contract):
    max_candidates: int = Field(default=16, ge=1, le=100000)
    max_wall_seconds: FiniteFloat = Field(default=600.0, gt=0)
    worker_timeout_seconds: FiniteFloat = Field(default=60.0, gt=0)
    build_timeout_seconds: FiniteFloat = Field(default=300.0, gt=0)
    max_storage_bytes: int = Field(default=1_000_000_000, ge=1024)
    max_generator_calls: int = Field(default=0, ge=0)
    reserve_confirmation_seconds: FiniteFloat = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def reserve_fits(self):
        if self.reserve_confirmation_seconds >= self.max_wall_seconds:
            raise ValueError("confirmation reserve must leave positive search budget")
        return self


class CampaignSpec(Contract):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9_-]{1,80}$")
    mode: Literal["demo", "hopper"] = "demo"
    gpu_execution: bool = False
    source_snapshot_digest: str = Field(pattern=SHA)
    task: TaskSpec
    budget: Budget = Field(default_factory=Budget)
    gpu_uuid: str | None = Field(default=None, pattern=r"^GPU-[A-Za-z0-9-]+$")
    power_limit_watts: FiniteFloat = Field(default=400.0, gt=0)
    build_image: str | None = None
    target_tokens_per_second: int = Field(default=67000, ge=1)
    tokens_per_update: int = Field(default=32768, ge=1)
    immutable_model: dict = Field(default_factory=dict)
    beam_families: int = Field(default=4, ge=1, le=32)
    timing_repetitions: int = Field(default=20, ge=3, le=1000)

    @model_validator(mode="after")
    def execution_contract(self):
        if self.mode == "demo":
            if self.gpu_execution or self.task.operation != "vector_add_f32":
                raise ValueError("demo cannot execute or certify GPU kernels")
        if self.mode == "hopper":
            if self.task.operation != "fp8_rows_bf16":
                raise ValueError("initial Hopper adapter supports FP8 row quantization")
            if self.gpu_execution and (not self.gpu_uuid or not self.build_image):
                raise ValueError("GPU execution requires a pinned GPU UUID and build image")
            if self.build_image and "@sha256:" not in self.build_image:
                raise ValueError("build image must be pinned by digest")
            required = {"n_layers": 30, "dim": 2048, "ffn_dim": 5376,
                        "vocab_size": 96000, "max_seq_len": 4096}
            if any(self.immutable_model.get(k) != v for k, v in required.items()):
                raise ValueError("Hopper campaign must preserve the frozen Terminus architecture")
        return self


class CandidateSpec(Contract):
    schema_version: Literal[1] = 1
    task_digest: str = Field(pattern=SHA)
    backend: Literal["numpy_demo", "cuda", "triton"]
    family: str = Field(pattern=r"^[a-z0-9_-]{1,80}$")
    source_digest: str = Field(pattern=SHA)
    parameters: dict[str, int] = Field(default_factory=dict)
    parent_ids: list[str] = Field(default_factory=list)
    hypothesis: str = Field(min_length=20, max_length=8000)
    expected_saved_ms: FiniteFloat = Field(default=0.0, ge=0)
    workspace_bytes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def launch_parameters(self):
        allowed = {"numpy_demo": {"chunk"}, "cuda": {"block"},
                   "triton": {"num_warps", "rows_per_cta"}}[self.backend]
        if set(self.parameters) != allowed:
            raise ValueError(f"required parameter keys: {sorted(allowed)}")
        if self.backend == "numpy_demo" and not 1 <= self.parameters["chunk"] <= 1048576:
            raise ValueError("invalid demo chunk")
        if self.backend == "cuda" and self.parameters["block"] not in (128, 256, 512):
            raise ValueError("unsupported CUDA block")
        if self.backend == "triton":
            if self.parameters["num_warps"] not in (4, 8) or self.parameters["rows_per_cta"] not in (1, 4, 8, 16):
                raise ValueError("unsupported Triton launch parameters")
        if self.workspace_bytes:
            raise ValueError("initial trusted adapters do not permit candidate workspace")
        return self


GateStatus = Literal["passed", "failed", "pending", "unavailable", "inconclusive"]


class Gate(Contract):
    status: GateStatus
    details: dict = Field(default_factory=dict)


class EvaluationReceipt(Contract):
    schema_version: Literal[1] = 1
    campaign_digest: str = Field(pattern=SHA)
    candidate_digest: str = Field(pattern=SHA)
    task_digest: str = Field(pattern=SHA)
    judge_digest: str = Field(pattern=SHA)
    artifact_digest: str | None = None
    artifact_manifest_digest: str | None = None
    attempt_id: str
    scope: Literal["cpu_demo", "gpu_microbenchmark", "training_update", "historical_import"]
    gates: dict[str, Gate]
    measurements: dict = Field(default_factory=dict)
    runtime: dict = Field(default_factory=dict)
    error: str | None = None


def schemas():
    return {model.__name__: model.model_json_schema() for model in
            (TaskSpec, CandidateSpec, CampaignSpec, EvaluationReceipt)}
