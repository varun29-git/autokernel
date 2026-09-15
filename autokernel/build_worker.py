"""Trusted build entrypoint, executed ONLY inside the restricted build container."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cuda", "triton"], required=True)
    parser.add_argument("--parameters", required=True)
    args = parser.parse_args()
    parameters = json.loads(args.parameters)
    if args.backend == "cuda":
        subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", "--fmad=false", "-lineinfo",
                        f"-DBLOCK_SIZE={parameters['block']}", "/input/kernel.cu",
                        "-o", "/output/kernel.cubin"], check=True)
        metadata = {"symbol": "fp8_rows", "block": parameters["block"], "shared": 0,
                    "rows_per_cta": 1, "extra_pointer_args": 0}
    else:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        spec = importlib.util.spec_from_file_location("candidate", "/input/kernel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        signature = {"X": "*bf16", "Codes": "*u8", "Scales": "*fp32", "rows": "i64", "width": "i64",
                     "row_stride": "i64", "column_stride": "i64", "scale_stride": "i64", "epsilon": "fp32"}
        source = ASTSource(module.fp8_rows, signature=signature, constexprs={"ROWS": parameters["rows_per_cta"]})
        compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                                  options={"num_warps": parameters["num_warps"], "enable_fp_fusion": False})
        Path("/output/kernel.cubin").write_bytes(compiled.asm["cubin"])
        entry = re.search(r"\.entry\s+" + re.escape(compiled.metadata.name) + r"\s*\((.*?)\)",
                          compiled.asm["ptx"], re.S)
        if not entry:
            raise RuntimeError("cannot identify compiled Triton ABI")
        extra = entry.group(1).count(".param") - 9
        if extra not in (0, 1, 2) or any(getattr(compiled.metadata, name, 0) for name in
                                       ("global_scratch_size", "profile_scratch_size")):
            raise RuntimeError("unsupported Triton scratch ABI; needs a reviewed trusted adapter")
        metadata = {"symbol": compiled.metadata.name, "block": parameters["num_warps"] * 32,
                    "shared": compiled.metadata.shared, "rows_per_cta": parameters["rows_per_cta"],
                    "extra_pointer_args": extra}
    Path("/output/metadata.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
