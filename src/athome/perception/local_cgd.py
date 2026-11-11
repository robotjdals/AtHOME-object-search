"""Perception's CG-D on this computer's GPU instead of the GPU server
(scripts/serve_cgd_http.py, unchanged code of perception).

Measured on the room_baram frames with the RTX 4060 (2026-11-06): 1.48 s per
image against 1.41 s on the lab GPU when free (3.11 s when another model keeps it
at 98 %), the same found checks on 132 frames, up to 7.4 of 8 GB, so the CLIP text
server goes on the CPU. It is for when the lab GPU is busy; the laptop runs hot.

Perception's engine looks for its code in ~/conceptgraph_ws. Here the clones
are directly in the home directory, so the server runs with a private HOME
that links them there and keeps using the normal model caches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

PORT = 18200                        # the local end of the GPU server's tunnel: same url for the node
CLONES = ("Grounded-Segment-Anything", "concept-graphs")


def local_home(home: Path) -> Path:
    """The private HOME, created when missing. Raises if a clone is missing."""
    workspace = home / ".cache/athome/cgd_home/conceptgraph_ws"
    workspace.mkdir(parents=True, exist_ok=True)
    for name in CLONES:
        clone = home / name
        if not clone.is_dir():
            raise FileNotFoundError(f"CG-D 코드 없음: {clone}")
        link = workspace / name
        if link.is_symlink() and link.resolve() != clone.resolve():
            link.unlink()
        if not link.exists():
            link.symlink_to(clone)
    return workspace.parent


def server_command(repo: Path, python: str, home: Path) -> Tuple[List[str], Dict[str, str]]:
    """Command and extra environment of the local detection server."""
    robot_live = home / "athome/semantic_geometry_ws/tools/robot_live"
    if not robot_live.is_dir():
        raise FileNotFoundError(f"인지 코드 없음: {robot_live}")
    env = {
        "HOME": str(local_home(home)),
        "HF_HOME": str(home / ".cache/huggingface"),
        "TORCH_HOME": str(home / ".cache/torch"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    }
    cmd = [python, "-u", str(repo / "scripts/serve_cgd_http.py"), "--robot-live", str(robot_live),
           "--host", "127.0.0.1", "--port", str(PORT)]
    return cmd, env
