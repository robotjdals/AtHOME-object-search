"""meta.json next to every saved LoRA adapter (hand-off to the inference server).

The inference server (vLLM, one base model with the room / search_location /
workspace adapters) checks an adapter against its settings before serving:
base model revision, rank, alpha, target modules, training method, data
version, prompt version, date and the validation results measured on the
training server. ``write_meta`` records these; ``add_validation`` appends a
result of scripts/evaluate_student.py (a list, earlier entries are kept).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Optional

from athome.inference.prompts import PROMPT_VERSION, TEACHER_PROMPT_VERSION

REPO = Path(__file__).resolve().parents[3]


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def data_version(export_manifest: Path) -> dict:
    """Provenance of an SFT export (scripts/export_sft_dataset.py manifest)."""
    manifest = json.loads(export_manifest.read_text(encoding="utf-8"))
    return {"export": str(export_manifest.parent), "manifest_sha256":
            hashlib.sha256(export_manifest.read_bytes()).hexdigest(),
            "export_prompt_version": manifest.get("prompt_version"),
            "categories": manifest.get("categories"), "categories_sha256": manifest.get("categories_sha256"),
            "candidate_order": manifest.get("candidate_order")}


def write_meta(adapter_dir: Path, *, adapter: str, method: str, base_model: dict, lora: dict,
               data: dict, training: dict, results: Optional[dict] = None) -> Path:
    meta = {
        "adapter": adapter, "method": method,
        "base_model": {"name": base_model["name"], "revision": base_model.get("revision"),
                       "dtype": base_model.get("dtype"), "enable_thinking": base_model.get("enable_thinking")},
        "lora": {"r": lora["r"], "alpha": lora["alpha"], "dropout": lora.get("dropout"),
                 "target_modules": lora["target_modules"], "merged": False},
        "prompt_version": PROMPT_VERSION, "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "answer_format": '{"selected_id": "<alias>"}',
        "data": data, "training": training, "training_results": results or {},
        "validation": [],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(), "library_versions": _versions(),
    }
    path = Path(adapter_dir) / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def add_validation(adapter_dir: Path, result: dict) -> None:
    path = Path(adapter_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"meta.json 없음: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.setdefault("validation", []).append(
        {**result, "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
