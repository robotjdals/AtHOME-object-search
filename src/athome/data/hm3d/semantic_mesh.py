"""Per-instance triangles of an HM3DSem semantic GLB (athome_z_up).

Instance IDs are encoded as per-vertex COLOR_0 (linear RGB) matching the
sRGB hex colors in ``<scene>.semantic.txt``. Supports the inspected static,
uncompressed GLB with identity mesh nodes (transforms only on mesh-less leaf
nodes such as an exported light or camera); anything else fails explicitly.
The annotation table is read as habitat-sim's HM3D loader does
(src/esp/scene/HM3DSemanticScene.cpp): the color is parsed as hex with
``std::stoul(.., 16)`` and inserted into the color -> ID map with ``insert``,
so when colors repeat the first row keeps the color and later rows get no
mesh (they are listed as shadowed).
Parser behavior matches scripts/review_corridor_surfaces.py, which verified
that the mesh is already in athome_z_up (no rotation is applied).
GLB/accessor reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
from typing import Dict, Tuple

import numpy as np


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_glb(path):
    raw = Path(path).read_bytes()
    _require(len(raw) >= 20, "Truncated GLB")
    magic, version, size = struct.unpack_from("<4sII", raw)
    _require(magic == b"glTF" and version == 2 and size == len(raw), "Invalid GLB v2 header")
    doc, binary, offset = None, None, 12
    while offset < size:
        _require(offset + 8 <= size, "Truncated chunk header")
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        _require(offset + length <= size, "Truncated chunk")
        chunk = raw[offset:offset + length]
        if kind == 0x4E4F534A:
            _require(doc is None, "Duplicate JSON chunk")
            doc = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942:
            _require(binary is None, "Duplicate BIN chunk")
            binary = chunk
        offset += length
    _require(doc is not None and binary is not None, "Missing GLB chunks")
    _require(not doc.get("extensionsUsed"), "Extensions need explicit support")
    buffers = doc.get("buffers", [])
    _require(len(buffers) == 1 and "uri" not in buffers[0], "Expected one embedded buffer")
    _require(buffers[0]["byteLength"] <= len(binary), "Truncated buffer")
    return doc, binary


def _accessor(doc, binary, index):
    a = doc["accessors"][index]
    _require("sparse" not in a and "bufferView" in a, "Sparse/implicit accessor is unsupported")
    types = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    _require(a["componentType"] in types and a["type"] in widths, "Unsupported accessor type")
    dtype = np.dtype(types[a["componentType"]])
    width, count = widths[a["type"]], a["count"]
    v = doc["bufferViews"][a["bufferView"]]
    _require(v["buffer"] == 0 and count > 0, "Invalid buffer/count")
    stride = v.get("byteStride", width * dtype.itemsize)
    relative = a.get("byteOffset", 0)
    start = v.get("byteOffset", 0) + relative
    end = relative + (count - 1) * stride + width * dtype.itemsize
    _require(stride >= width * dtype.itemsize and relative >= 0 and end <= v["byteLength"]
             and v.get("byteOffset", 0) + v["byteLength"] <= len(binary),
             "Accessor exceeds bufferView")
    return np.ndarray((count, width), dtype=dtype, buffer=binary,
                      offset=start, strides=(stride, dtype.itemsize)).copy()


def read_color_table(semantic_txt, shadowed=None) -> Dict[int, Tuple[int, str, str]]:
    """packed sRGB -> (semantic_id, category, region) from semantic.txt.

    IDs whose color repeats an earlier row are appended to ``shadowed``."""
    table = {}
    ids = set()
    with Path(semantic_txt).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4 or not row[0].strip().isdigit():
                continue
            sid = int(row[0])
            _require(sid not in ids, f"Duplicate semantic ID: {sid}")
            ids.add(sid)
            value = row[1].strip().lstrip("#")
            _require(0 < len(value) <= 6 and all(c in "0123456789abcdefABCDEF" for c in value),
                     f"Invalid RGB: {sid}")
            code = int(value, 16)
            if code in table:
                if shadowed is not None:
                    shadowed.append(sid)
                continue
            table[code] = (sid, row[2].strip(), row[3].strip())
    return table


def instance_triangles(glb_path, semantic_txt):
    """Return ({semantic_id: (N, 3, 3) triangles}, stats).

    Triangles whose three vertex colors differ, or whose color is not in the
    annotation table, are counted in ``stats`` and not assigned."""
    doc, binary = load_glb(glb_path)
    shadowed = []
    table = read_color_table(semantic_txt, shadowed)
    _require(not doc.get("animations") and not doc.get("skins"),
             "Animated/skinned scenes are unsupported")
    scene = doc["scenes"][doc.get("scene", 0)]
    nodes = doc.get("nodes", [])
    chunks: Dict[int, list] = {}
    stats = {"triangles": 0, "mixed_color": 0, "unannotated_color": 0, "shadowed_ids": shadowed}
    visited = set()

    def visit(index):
        _require(index not in visited, "Repeated/cyclic node")
        visited.add(index)
        node = nodes[index]
        # A transform on a node without mesh or children places nothing.
        _require(not any(k in node for k in ("matrix", "translation", "rotation", "scale", "skin"))
                 or ("mesh" not in node and not node.get("children")),
                 "Node transform differs from inspected file; stop for review")
        if "mesh" in node:
            for p in doc["meshes"][node["mesh"]]["primitives"]:
                _require(p.get("mode", 4) == 4 and not p.get("targets")
                         and not p.get("extensions"), "Unsupported primitive")
                attrs = p["attributes"]
                _require("POSITION" in attrs and "COLOR_0" in attrs,
                         "Missing positions/semantic colors")
                xyz = _accessor(doc, binary, attrs["POSITION"]).astype(float)
                color = _accessor(doc, binary, attrs["COLOR_0"])
                _require(xyz.shape[1] == 3 and len(color) == len(xyz)
                         and color.shape[1] in (3, 4) and np.isfinite(xyz).all(),
                         "Invalid geometry/colors")
                rgb = color[:, :3].astype(float)
                if color.dtype.kind == "u":
                    _require(doc["accessors"][attrs["COLOR_0"]].get("normalized", False),
                             "Unnormalized integer RGB")
                    rgb /= np.iinfo(color.dtype).max
                _require(np.isfinite(rgb).all() and (rgb >= 0).all() and (rgb <= 1).all(),
                         "Invalid RGB range")
                # glTF COLOR_0 is linear; annotation hex colors are sRGB.
                rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                               1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
                rgb = np.rint(rgb * 255).astype(np.uint32)
                packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
                if "indices" in p:
                    idx = _accessor(doc, binary, p["indices"])
                    _require(idx.shape[1] == 1 and idx.dtype.kind == "u", "Invalid triangle indices")
                    idx = idx.ravel().astype(np.int64)
                else:
                    idx = np.arange(len(xyz))
                _require(len(idx) % 3 == 0 and len(idx) > 0
                         and idx.min() >= 0 and idx.max() < len(xyz), "Invalid faces")
                faces = idx.reshape(-1, 3)
                face_rgb = packed[faces]
                uniform = (face_rgb[:, 0] == face_rgb[:, 1]) & (face_rgb[:, 0] == face_rgb[:, 2])
                stats["triangles"] += len(faces)
                stats["mixed_color"] += int((~uniform).sum())
                codes = face_rgb[uniform, 0]
                kept = faces[uniform]
                for code in np.unique(codes):
                    entry = table.get(int(code))
                    selected = kept[codes == code]
                    if entry is None:
                        stats["unannotated_color"] += len(selected)
                        continue
                    chunks.setdefault(entry[0], []).append(xyz[selected])
        for child in node.get("children", []):
            visit(child)

    for index in scene["nodes"]:
        visit(index)
    return {sid: np.concatenate(parts) for sid, parts in chunks.items()}, stats


def instance_aabbs(glb_path, semantic_txt):
    """({semantic_id: (min_xyz, max_xyz)}, stats) from the semantic mesh."""
    triangles, stats = instance_triangles(glb_path, semantic_txt)
    boxes = {}
    for sid, tri in triangles.items():
        points = tri.reshape(-1, 3)
        boxes[sid] = (points.min(axis=0).tolist(), points.max(axis=0).tolist())
    return boxes, stats
