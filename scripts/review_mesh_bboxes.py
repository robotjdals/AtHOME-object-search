"""3D audit of the semantic-mesh bbox rule's non-trivial cases (read-only).

Covers objects kept as multi-part and objects whose detached pieces were
dropped. Per object: components colored by role (largest / other kept /
dropped detached / dropped speck), the old OBB-derived AABB (red) and the
rule AABB (green), with nearby same-room instances as faint context.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from athome.data.hm3d.coordinates import convert_bbox
from athome.data.hm3d.mesh_bbox import MeshBboxRule, rule_dict, split_components
from athome.data.hm3d.semantic_mesh import instance_triangles

COLORS = {"main": "#0072B2", "kept": "#E69F00", "detached": "#CC79A7", "speck": "#999999"}
CONTEXT_MARGIN_M = 0.5


def mesh(triangles, name, color, opacity=1.0, legend=True):
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name,
                     color=color, opacity=opacity, flatshading=True, showlegend=legend,
                     hovertemplate=name + "<extra></extra>")


def box_lines(lo, hi, name, color):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        for idx in (a, b):
            xs.append(corners[idx][0]); ys.append(corners[idx][1]); zs.append(corners[idx][2])
        xs.append(None); ys.append(None); zs.append(None)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                        line=dict(color=color, width=5), hoverinfo="name")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="원본 habitat_native annotation (이전 OBB bbox 표시용)")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rule = MeshBboxRule()
    if audit["rule"] != json.loads(json.dumps(rule_dict(rule))):
        raise RuntimeError("audit와 현재 bbox 규칙이 다릅니다.")
    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    by_id = {o["object_id"]: o for o in source["objects"]}
    scene = Path(source["scene_path"])
    glb = scene.with_name(scene.name.replace(".basis.glb", ".semantic.glb"))
    triangles, _ = instance_triangles(glb, glb.with_suffix(".txt"))
    sid_of = {o["object_id"]: int(o["object_id"].rsplit("_", 1)[1]) for o in source["objects"]}
    rows = {r["object_id"]: r for r in audit["objects"]}

    fig = go.Figure()
    groups, buttons = [], []
    for oid in audit["multi_part_ids"] + audit["detached_dropped_ids"]:
        tri = triangles[sid_of[oid]]
        labels, share, keep = split_components(tri, by_id[oid]["category"] or "", rule)
        main_label = int(np.argmax(share))
        start = len(fig.data)
        for label in np.argsort(-share):
            role = ("main" if label == main_label else "kept" if keep[label]
                    else "speck" if share[label] < rule.min_area_share else "detached")
            fig.add_trace(mesh(tri[labels == label],
                               f"{oid} comp{label} {role} {share[label]*100:.1f}%",
                               COLORS[role]))
        kept = tri[keep[labels]].reshape(-1, 3)
        lo, hi = kept.min(axis=0), kept.max(axis=0)
        fig.add_trace(box_lines(lo, hi, "rule AABB", "#009E73"))
        old = by_id[oid]["bbox"]
        if old is not None:
            z = convert_bbox(old)
            fig.add_trace(box_lines(z["min"], z["max"], "old OBB AABB", "#D55E00"))
        region = by_id[oid]["region_id"]
        for other, obj in by_id.items():
            s = sid_of[other]
            if other == oid or obj["region_id"] != region or s not in triangles:
                continue
            pts = triangles[s].reshape(-1, 3)
            if np.all(pts.max(axis=0) >= lo - CONTEXT_MARGIN_M) and np.all(pts.min(axis=0) <= hi + CONTEXT_MARGIN_M):
                fig.add_trace(mesh(triangles[s], f"context {other}", "#CCCCCC", 0.25, legend=False))
        groups.append((oid, start, len(fig.data)))
    for oid, start, end in groups:
        visible = [start <= i < end for i in range(len(fig.data))]
        r = rows[oid]
        buttons.append(dict(label=oid, method="update", args=[
            {"visible": visible},
            {"title": f"{oid} (room {by_id[oid]['region_id']}): components {r['components']}, "
                      f"dropped detached {r['dropped_detached_area_share']*100:.1f}% / "
                      f"speck {r['dropped_speck_area_share']*100:.1f}%, "
                      f"extent beyond largest {r['extent_beyond_main_component_m']:.2f} m"}]))
    for i, trace in enumerate(fig.data):
        trace.visible = groups[0][1] <= i < groups[0][2]
    fig.update_layout(
        title=buttons[0]["args"][1]["title"],
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.08, showactive=True)],
        scene=dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
        height=850, margin=dict(l=10, r=10, t=110, b=10))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".html.tmp")
    fig.write_html(tmp, include_plotlyjs=True, full_html=True)
    tmp.replace(args.output)
    print("저장:", args.output, f"({len(groups)}개 객체)")


if __name__ == "__main__":
    main()
