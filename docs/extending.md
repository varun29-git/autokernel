# Add a trusted kernel family

Candidate authors can change device code and declared launch parameters. Adding an operation is a change to the evaluator and requires a new source/contract identity.

1. Define the computation, tensor/stride/alias constraints, forward and backward outputs, state mutation and supported shapes in `contracts.py`. Make unsupported inputs take an explicit incumbent path.
2. Implement independent equations in `oracles.py` or a new oracle module. Bind every oracle dependency into the task identity. Do not call candidate helpers. For STE backward, test the specified surrogate equations rather than finite-differencing the quantized forward.
3. Define a fixed, bounded compiled-artifact ABI and trusted adapter in `worker.py`/`cuda.py`. The candidate cannot provide allocation, timers, streams, reference code or arbitrary Python wrappers. Additional scratch arguments must be zero or managed by an explicitly bounded workspace contract.
4. Add reviewed seed implementations and a parameter grid to `templates/` and `search.py`. Compare against a competitive incumbent. Support a new compiler only through the isolated build interface.
5. Test wrong outputs, wrong dtypes, missing gradients, stale outputs, input mutation, tail/stride handling, nonfinite propagation, large offsets and asynchronous completion. Test watchdog cleanup and sanitizer error exits.
6. Add a full-training integration adapter. Bind artifact and adapter identities; compare complete parameter/gradient/optimizer/RNG state. Preserve clipping, tied-weight gradients, update tokens and cache epochs.
7. Run native confirmation, then quality qualification. A CPU test or a correct isolated forward does not certify the new training path.

The next useful families from the design are producer-to-FP8 fusion, packed backward producers, projection weight-gradient schedules and unfiltered full-vocabulary loss. Start with a complete replacement boundary, including the additional passes a fusion still needs. Rowwise and transposed FP8 groups are different numerical operations.
