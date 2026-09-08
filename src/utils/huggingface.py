from __future__ import annotations

from pathlib import Path


def resolve_cached_model(identifier: str) -> str:
    """Prefer an existing complete HF snapshot to avoid accidental network calls."""
    direct = Path(identifier)
    if direct.exists():
        return str(direct)
    cache_name = "models--" + identifier.replace("/", "--")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if snapshots.is_dir():
        weight_names = ("model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack")
        candidates = sorted(
            path for path in snapshots.iterdir()
            if (path / "config.json").is_file()
            and any((path / name).exists() for name in weight_names)
        )
        if candidates:
            return str(candidates[-1])
    return identifier
