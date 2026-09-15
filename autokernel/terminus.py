"""Read-only import of a Terminus source/config snapshot; no training module imports on the controller."""
from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path

from . import oracles
from .contracts import Budget, CampaignSpec, Case, TaskSpec
from .evaluator import judge_digest
from .integrity import file_digest, read_json, snapshot, write_json


def dataclass_defaults(path, class_name):
    """Resolve literal dataclass defaults without executing the external repository."""
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    defaults = {}
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and statement.value is not None:
            try:
                defaults[statement.target.id] = ast.literal_eval(statement.value)
            except (ValueError, TypeError) as error:
                raise ValueError(f"nonliteral config default requires a reviewed importer: {statement.target.id}") from error
    return defaults


def import_terminus(repository, config_path, output):
    repository = Path(repository).resolve()
    config = read_json(config_path)
    names = [("model", "model.py", "BitNetLlamaConfig"), ("training", "train.py", "TrainConfig"),
             ("optimizer", "optim.py", "OptimizerConfig")]
    resolved = {}
    for section, filename, class_name in names:
        defaults = dataclass_defaults(repository / "terminus" / filename, class_name)
        unknown = set(config.get(section, {})) - set(defaults)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        resolved[section] = {**defaults, **config.get(section, {})}
    source = snapshot(repository / "terminus")
    result = {"repository": str(repository), "resolved_config": resolved,
              "source": source, "original_config_digest": file_digest(config_path),
              "scope": "Frozen source/config import; no GPU execution or certification"}
    write_json(output, result)
    return result


def hopper_campaign(imported, *, name="terminus-h100", gpu_uuid=None, build_image=None, enable=False):
    cases = [Case(rows=rows, columns=256, epsilon=0.0 if stress in {"zero", "tiny"} else 1e-5,
                  stress=stress, column_stride=2 if rows == 7 else 1, row_padding=8 if rows == 7 else 0)
             for rows, stress in [(1, "normal"), (7, "rounding"), (127, "zero"), (128, "tiny"),
                                  (129, "outlier"), (257, "nonfinite")]]
    cases += [Case(rows=rows, columns=width) for rows in (16384, 32768)
              for width in (2048, 3072, 5376, 10752)]
    task = TaskSpec(operation="fp8_rows_bf16", cases=cases, reference_digest=file_digest(oracles.__file__),
                    equations="Independent 1x128 E4M3FN quantization of BF16 values; FP32 scales; exact RNE codes.",
                    calls_per_update=120, measured_region_ms=33.64)
    return CampaignSpec(name=name, mode="hopper", gpu_execution=enable, task=task,
                        source_snapshot_digest=judge_digest(), gpu_uuid=gpu_uuid, build_image=build_image,
                        immutable_model=imported["resolved_config"]["model"],
                        budget=Budget(max_candidates=16, max_wall_seconds=14400.0,
                                      worker_timeout_seconds=120.0, max_storage_bytes=20_000_000_000,
                                      reserve_confirmation_seconds=3600.0))


def historical_receipt(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
    else:
        data = read_json(path)
    steps = data.get("step_seconds", [])
    if not steps or any(not isinstance(v, (int, float)) or not 0 < v < float("inf") for v in steps):
        raise ValueError("missing or invalid historical timing samples")
    tokens = data["tokens_per_update"]
    return {"scope": "historical_import", "source_digest": file_digest(path),
            "tokens_per_update": tokens, "pooled_tokens_per_second": tokens * len(steps) / sum(steps),
            "step_seconds": steps, "runtime": data.get("runtime"), "certification": "none"}
