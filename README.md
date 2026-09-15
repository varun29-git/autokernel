# Terminus AutoKernel

A separate research framework for optimizing training kernels while preserving the model's numerical and update contracts. It runs bounded searches, judges candidates independently, and keeps reproducible evidence. Production training imports no search controller or model-provider client.

**Status:** the CPU control plane and adversarial search demo are tested. CUDA/Triton compilation, native H100 execution, sanitizer/profiler integration and complete-training adapters are implemented but **not yet validated on an H100**. This repository does not claim a throughput improvement, 67k tokens/s, or multilingual quality parity. The first native search task is **BF16 → rowwise FP8 quantization**, including exact codes/scales and strided inputs. Other kernel families require additional trusted task/ABI/oracle adapters; they are not silently treated as supported.

## Quick start

Python 3.10 or newer, on macOS or Linux:

```bash
uv sync --extra dev
uv run pytest -q
uv run autokernel demo --store artifacts/demo --include-wrong
uv run autokernel verify-store --store artifacts/demo
uv run autokernel report --store artifacts/demo
```

The demo executes actual NumPy kernels with several chunk schedules. Deliberately wrong addition, unwritten-output and input-mutating implementations must be rejected by the supervisor's independent oracle. Timings are labeled `cpu_demo`; no demo result can become a GPU or production qualification. Rerunning the same frozen campaign reuses completed attempts rather than executing them again. Code/config changes require a new campaign directory.

## Implemented components

| Component | Behavior |
| --- | --- |
| Contracts | Strict Pydantic models and exported JSON schemas; unknown fields, nonfinite settings, unsupported launches and architecture changes fail validation |
| Controller | Family-diverse search ordering, seed grids, generated proposals, deduplication, immutable campaigns and resumable state |
| Evidence | Content-addressed artifacts, signed supervisor receipts, chained audit events, recursive source identities and launch-manifest binding |
| Resources | Exclusive process-held device/controller locks, descendant cancellation, worker/build/log limits, conservative crash charging and reserved confirmation budget |
| Independent oracle | NumPy BF16 rounding, E4M3FN encoding, independent scale equations, exact code/scale checks, tiny/zero/outlier/nonfinite/stride cases |
| Native build | Unprivileged, network-disabled Docker compilation to cubin; pinned image digest; no candidate Python imported into the judge |
| Native execution | Trusted CUDA Driver launcher with constrained pointer/scalar ABI; bounds on shape, launch resources and workspace |
| Diagnostics | Separate Compute Sanitizer and Nsight Compute commands; supervisor-side power/clock/memory telemetry; instrumented timings excluded from performance claims |
| Terminus integration | Read-only source/default-config import; FP8 row replacement with incumbent transposed operands; rotating-input full updates; parameter/gradient/Adam/buffer/RNG comparisons |
| Timing analysis | Fixed paired samples, randomized balanced collection, A/A checks, session-level Fieller intervals, multiplicity/error-spending controls and absolute target bounds |
| Quality analysis | Equal-token paired-seed noninferiority statistics for all eleven languages; per-language failure cannot be hidden by an aggregate |
| Export | Numerically passing, versioned **research-only** bundles; automatic production certification is intentionally unavailable |

The system optimizes a complete subgraph or training update before making a training-performance claim. It does not add isolated kernel speedups together. A custom kernel is not preferred over a faster incumbent library.

## AI research agent: generate, evaluate, learn

AutoKernel is the supervisor around a coding model, not a replacement for one. The coding model proposes source; the supervisor builds it in isolation, measures it with hidden inputs, rejects incorrect output, and returns only signed campaign evidence to the next proposal. The model cannot change the task, oracle, workload, pass/fail status or training checkout.

The included `autokernel-llm-provider` is an OpenAI-compatible Chat Completions adapter. It reads the command-provider request on stdin and emits exactly one JSON object containing `candidate` and `source`. It derives task/source digests itself instead of trusting the model to supply them. Set credentials only in the provider process environment; build and evaluation workers receive a scrubbed environment. No provider is contacted by installation or import.

```bash
export AUTOKERNEL_MODEL_API_KEY='...'
export AUTOKERNEL_MODEL_NAME='your-coding-model'
# Optional for a compatible non-default endpoint:
export AUTOKERNEL_MODEL_ENDPOINT='https://provider.example/v1/chat/completions'

uv run autokernel research --campaign artifacts/hopper.json \
  --store artifacts/h100-search -- autokernel-llm-provider
```

Each call receives the frozen task, candidate schema, and summarized signed results from earlier attempts. Its required response is one JSON object with `backend`, `family`, permitted `parameters`, a falsifiable `hypothesis`, complete `source`, and `parent_ids`. Invalid responses consume the bounded generation activity but cannot become candidates. Every accepted proposal records the observation digest that informed it in the signed event chain.

The H100 plan starts with four diverse seed kernels and reserves twelve of the sixteen candidate slots for the coding model. Change this before freezing a campaign with `hopper-plan --seed-candidates N --generator-calls M`; `N + M` cannot exceed the candidate budget.

For a local protocol smoke test, the included generator is deterministic, not an LLM:

```bash
uv run autokernel demo-plan --output artifacts/research-campaign.json --generator-calls 2
uv run autokernel research --campaign artifacts/research-campaign.json \
  --store artifacts/research -- python "$(pwd)/examples/demo_generator.py"
```

`research` alternates bounded proposals and independent evaluation. `propose` submits one generated proposal; `submit --candidate candidate.json --source kernel.cu` accepts a manually produced one. Candidate files cannot alter the judge, references, workload or thresholds.

An isolated-kernel timing is exploratory. The report's `best_exploratory_candidate` is not a production dispatch decision. Confirmation, quality and supported-shape requirements remain separate.

## Prepare the H100 campaign without starting it

The original training repository stays independent. Its local working-tree content, including uncommitted files, is hashed recursively. The importer reads literal dataclass defaults through Python's AST; it does not execute that repository.

```bash
uv run autokernel import-terminus \
  --repo /path/to/terminus-flagship-150m-30btokens \
  --config /path/to/terminus-flagship-150m-30btokens/configs/1.5b_h100_kernel_optimized.json \
  --output artifacts/terminus-import.json
uv run autokernel hopper-plan --imported artifacts/terminus-import.json \
  --output artifacts/hopper-disabled.json
uv run autokernel validate artifacts/hopper-disabled.json
```

The generated campaign preserves 30 layers, width 2,048, FFN width 5,376, vocabulary 96,000 and context 4,096. Its primary workload uses 32,768 tokens/update. `gpu_execution` defaults to false. Native fixtures include all actual projection widths at 16,384 and 32,768 rows, plus small, strided and stress cases. Large fixtures require substantial host RAM and disk.

For a future native run, use the selected H100 host with its pinned Torch/FA3/DeepGEMM runtime. Install this framework into that environment. Prepare a devel image using [docker/README.md](docker/README.md), then create a **new** campaign with the exact UUID and image digest:

```bash
autokernel hopper-plan --imported artifacts/terminus-import.json \
  --gpu-uuid GPU-REPLACE-WITH-ACTUAL-UUID \
  --build-image registry/image@sha256:REPLACE-WITH-64-HEX-DIGEST \
  --seed-candidates 4 --generator-calls 12 \
  --enable-execution --output artifacts/hopper-enabled.json
```

Review/freeze budgets before `autokernel run --campaign ... --store ...`. No command in this framework provisions, starts, stops or changes the power policy of an Azure VM. Locks coordinate this framework's workers; other users' jobs are detected through inventory checks, not forcibly terminated. Docker is mandatory for native compilation; missing isolation does not fall back to unsafe host execution.

## Qualification workflow

Run these on the H100 only after the numerical candidate campaign passes:

```bash
autokernel sanitize --store artifacts/h100 --candidate CANDIDATE_ID --output artifacts/sanitizers
autokernel profile --store artifacts/h100 --candidate CANDIDATE_ID --output artifacts/profile
autokernel training-pair --store artifacts/h100 --candidate CANDIDATE_ID \
  --imported artifacts/terminus-import.json --output artifacts/training-check
```

`training-pair` runs both arms from seeded initialization, performs native preflight, includes backward/clipping/AdamW/cache refresh, and compares final parameters, gradients, moments, buffers and Torch RNG states exactly. The trusted adapter replaces BF16 row quantization and retains the incumbent for unsupported inputs/transposed operands. The additional synthetic input copy is inside measured update time. Full-state snapshots can require roughly **50 GB** of disk for two arms; use sufficient host RAM/storage. `--no-state` produces a pending integration result, never a pass.

Randomized timing collection requires signed state-comparison and sanitizer reports:

```bash
autokernel collect-pairs --store artifacts/h100 --candidate CANDIDATE_ID \
  --imported artifacts/terminus-import.json --policy examples/confirmation_policy.json \
  --integration-report artifacts/training-check/comparison.json \
  --sanitizer-report artifacts/sanitizers/report.json \
  --aa --output artifacts/aa
autokernel calibrate --blocks artifacts/aa/blocks.json \
  --policy examples/confirmation_policy.json --output artifacts/calibration.json
```

Collect a fresh sample without `--aa`, then use `autokernel confirm --blocks ... --policy ... --calibration ... --output ...`. Budget reservation uses maximum worker time, so allocate enough remaining wall time before collection. Plans are written before measurement. Interrupted/partial collections are not accepted as complete samples, and the fixed sample count cannot be extended until an interval happens to pass.

The analysis assumes balanced, approximately independent session means. Session grouping and A/A checks do not prove those assumptions or confidence-interval coverage. Inspect telemetry/drift and use separately initialized sessions when qualifying results. An input file marked `measured` is not authenticated simply by that string: retain and verify the signed collection receipt. Analysis output is statistical evidence, not a release certificate.

`quality-analysis --pairs ... --margins ... --output ...` accepts seed-level held-out NLL measurements at matched token counts. Its result is deliberately labeled statistical: a real-data pilot, provenance, adequate sensitivity, checkpoint/recovery tests and production IO/distributed qualification still need to be performed. It does not invent missing quality data or certify the existing FP8 method.

## Reproducibility and limits

- `verify-store` checks receipt signatures and the audit chain. Protect and back up `supervisor.key`; signatures are local trust, not protection from a compromised supervisor account. Candidate processes never receive that key.
- Source, binary **and launch metadata** are bound to evaluated receipts. A change requires new evidence. H100 SKU/runtime cohorts must not be mixed with an earlier saved throughput number.
- A live process-held lock is never stolen merely because a wall-clock TTL elapsed. Worker descendants inherit leases; recovery requires acquiring the lock after they exit. A GPU driver fault still requires host-level diagnosis.
- Search never resets or commits the training repository's working tree. Bundles are exported separately and do not bypass Terminus checkpoint source/runtime checks.
- The current native ABI supports one row-quantizer family. Extending to GEMM, backward fusion, attention or exact loss requires a new trusted task contract, ABI adapter, independent oracle and adversarial tests. See [extending the framework](docs/extending.md).
- Cloud lifecycle control, a real-data pilot runner, automatic checkpoint migration and automatic production rollout are not implemented. There is no hidden pass for these missing gates.
- “Cannot be improved” is not a meaningful finite-search guarantee. Report the best verified result within a declared workload and budget, alongside the remaining physical-resource and quality evidence.

The [design specification](docs/design.md) records the broader architecture and prioritized optimization backlog. [Historical baseline evidence](docs/baseline.md) is separate from this framework's tests. Export contract schemas with `autokernel schemas --output schemas`.
