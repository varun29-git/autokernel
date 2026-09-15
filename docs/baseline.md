# Historical training baseline

Imported from the Terminus September 15, 2026 optimization pass. These measurements predate this framework and are **not results of running it**.

The selected 30-layer, 1,502,210,108-parameter model measured 43,922 tokens/s against a matched 41,506 control on one 400 W H100 NVL. Updates contained 32,768 tokens. Selected peak allocated/reserved memory was 80.05/80.90 GB. The target of 67,000 tokens/s requires reducing approximately 746.04 ms/update to 489.07 ms/update.

These were synthetic forward/backward/clipping/AdamW/cache-refresh updates. Real data IO, communication, evaluation/checkpoint cadence and multilingual quality were not qualified. The advertisement-based compute-only reference was 237.87 ms/update, which is an optimistic analytical quantity, not attained training performance.

The [source summary](../evidence/baseline_summary.json) retains the original measurements, scope, source hashes and receipt paths. The full original receipts remain in the separate training repository; paths in that summary refer to that repository. The selected configuration is copied in [examples/terminus_config.json](../examples/terminus_config.json).
