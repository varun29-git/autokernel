# Native build environment

Build on Linux with a CUDA devel image that includes Python, CUDA headers and `nvcc`; the recorded training runtime uses Torch 2.8 / CUDA 12.6 / Triton 3.4. The build container does not need GPU access. This Dockerfile has not been built in the macOS development environment.

```bash
docker build --build-arg BASE_IMAGE='your-devel-image@sha256:ACTUAL_DIGEST' \
  -t your-registry/terminus-autokernel-builder:local -f docker/Dockerfile .
```

Publish it to your chosen container registry and supply the resulting immutable `repository@sha256:...` digest in the campaign. The framework refuses floating tags. Account for registry/image build activity separately; the kernel search itself does not create cloud resources.

At runtime the container receives only candidate source and the trusted build entrypoint, an output scratch directory, bounded CPU/RAM/PIDs, a read-only root, no network, no capabilities, no credentials and no GPU. Candidate Python may execute **here**, because Triton compilation evaluates Python. The supervisor accepts only a CUDA ELF cubin and a constrained launch manifest. It never imports the candidate Python module into its judge or training adapter.

Compiler changes require a fresh campaign/control. Tool availability, Triton AOT ABI compatibility and sanitizer coverage must be checked on the pinned H100 runtime before claiming native qualification.
