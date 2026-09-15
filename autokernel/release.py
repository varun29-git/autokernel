from __future__ import annotations

from pathlib import Path

from .integrity import atomic_write, digest, file_digest, read_json, write_json


def export_research_bundle(store, candidate_id, destination):
    row = next((r for r in store.rows() if r["id"] == candidate_id), None)
    if not row or row["state"] != "numerical_pass":
        raise ValueError("only numerically passing candidates can be exported")
    receipt = store.receipt(row["receipt"])
    artifact = read_json(store.root / "attempts" / row["attempt"] / "artifact.json")
    if receipt["artifact_digest"] != artifact["digest"] or file_digest(artifact["path"]) != artifact["digest"]:
        raise ValueError("artifact is not the verified artifact")
    if receipt["artifact_manifest_digest"] != digest({"digest": artifact["digest"], "metadata": artifact["metadata"]}):
        raise ValueError("launch manifest changed after evaluation")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    binary_name = "kernel.json" if receipt["scope"] == "cpu_demo" else "kernel.cubin"
    atomic_write(destination / binary_name, Path(artifact["path"]).read_bytes())
    campaign = store.verify(read_json(store.root / "campaign.json"))
    manifest = {"schema_version": 1, "candidate_digest": candidate_id,
                "artifact_digest": artifact["digest"], "artifact_file": binary_name,
                "launch": artifact["metadata"], "scope": receipt["scope"],
                "status": "research_only", "production_qualified": False,
                "task": campaign["task"], "receipt": receipt,
                "fallback": "frozen incumbent; never retry a partially updated CUDA state"}
    write_json(destination / "manifest.json", store.envelope(manifest))
    return manifest
