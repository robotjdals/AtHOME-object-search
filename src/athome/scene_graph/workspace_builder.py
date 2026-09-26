from copy import deepcopy
import math


def build_workspace_graph(semantic_inputs, semantic_labels):
    input_rooms = semantic_inputs["rooms"]
    label_rooms = semantic_labels["rooms"]

    room_map = {r["room_id"]: r for r in input_rooms}
    label_map = {r["room_id"]: r for r in label_rooms}

    if len(room_map) != len(input_rooms):
        raise ValueError("입력 Room ID 중복")
    if len(label_map) != len(label_rooms):
        raise ValueError("라벨 Room ID 중복")
    if set(room_map) != set(label_map):
        raise ValueError("입력과 라벨의 Room 목록 불일치")

    graph = {
        "schema_version": "0.1",
        "stage": "workspace_graph",
        "coordinate_frame": "athome_z_up",
        "up_axis": "z",
        "length_unit": "meter",
        "semantic_labels_are_ground_truth": False,
        "rooms": [],
        "workspaces": [],
        "objects": [],
    }
    global_ids = set()

    for rid, room in room_map.items():
        if (
            room["coordinate_frame"] != "athome_z_up"
            or room["up_axis"] != "z"
            or room["length_unit"] != "meter"
        ):
            raise ValueError(f"{rid}: 좌표계 또는 단위 불일치")

        objects = {o["object_id"]: o for o in room["objects"]}
        if len(objects) != len(room["objects"]):
            raise ValueError(f"{rid}: 객체 ID 중복")
        if global_ids.intersection(objects):
            raise ValueError(f"{rid}: 다른 Room과 객체 ID 중복")
        global_ids.update(objects)

        label = label_map[rid]
        sources = {
            s["source_object_id"]: s
            for s in label["workspace_sources"]
        }
        if len(sources) != len(label["workspace_sources"]):
            raise ValueError(f"{rid}: Source ID 중복")
        if not set(sources).issubset(objects):
            raise ValueError(f"{rid}: Room에 없는 Source")

        workspaces = {}
        # Child ID -> possible assignments.
        assignments = {}

        for sid in sorted(sources):
            source = objects[sid]
            bbox = source["bbox"]
            # Compute from the original, unrounded bounding box.
            position = [
                (bbox["min"][0] + bbox["max"][0]) / 2,
                (bbox["min"][1] + bbox["max"][1]) / 2,
                bbox["max"][2],
            ]
            if not all(math.isfinite(v) for v in position):
                raise ValueError(f"{sid}: 잘못된 Workspace 위치")

            wid = f"workspace:{rid}:{sid}"
            workspaces[sid] = {
                "workspace_id": wid,
                "room_id": rid,
                "source_object_id": sid,
                "function_label": sources[sid]["function_label"],
                "position": position,
                "child_object_ids": [],
            }

            neighbors = {
                n["object_id"]: n for n in source["neighbors"]
            }
            if len(neighbors) != len(source["neighbors"]):
                raise ValueError(f"{sid}: 이웃 ID 중복")

            children = source["child_candidate_ids"]
            if len(children) != len(set(children)):
                raise ValueError(f"{sid}: Child 후보 ID 중복")

            for child_id in children:
                if child_id not in objects:
                    raise ValueError(f"{sid}: 다른 Room의 Child 후보")
                if child_id == sid:
                    raise ValueError(f"{sid}: 자기 자신을 Child로 참조")
                if child_id in sources:
                    continue

                # Architectural surfaces remain directly under the room.
                child_category = " ".join(
                    objects[child_id]["semantic_tag"].casefold().split()
                )
                if child_category in {
                    "wall", "floor", "ceiling", "staircase wall"
                }:
                    continue

                if child_id not in neighbors:
                    raise ValueError(f"{sid}: Child의 기하 관계 누락")

                relation = neighbors[child_id]
                overlap = float(relation["overlap_xy"])
                delta_z = float(relation["delta_z"])

                if (
                    not math.isfinite(overlap)
                    or not math.isfinite(delta_z)
                    or not 0 <= overlap <= 1 + 1e-6
                    or delta_z < 0
                ):
                    raise ValueError(f"{sid}/{child_id}: 잘못된 기하 수치")

                assignments.setdefault(child_id, []).append(
                    (-overlap, delta_z, sid)
                )

        child_to_source = {}
        for child_id, candidates in assignments.items():
            _, _, sid = min(candidates)
            child_to_source[child_id] = sid
            workspaces[sid]["child_object_ids"].append(child_id)

        for workspace in workspaces.values():
            workspace["child_object_ids"].sort()

        source_ids = set(sources)
        child_ids = set(child_to_source)
        standalone_ids = set(objects) - source_ids - child_ids

        if source_ids & child_ids:
            raise ValueError(f"{rid}: Source와 Child 중복")
        if len(source_ids) + len(child_ids) + len(standalone_ids) != len(objects):
            raise ValueError(f"{rid}: 객체 분류 누락 또는 중복")

        for oid in sorted(objects):
            obj = objects[oid]
            node = {
                "object_id": oid,
                "room_id": rid,
                "semantic_tag": obj["semantic_tag"],
                "bbox": deepcopy(obj["bbox"]),
                "workspace_id": None,
            }

            if oid in sources:
                node["role"] = "source"
                node["parent_id"] = rid
                node["parent_type"] = "room"
                node["workspace_id"] = workspaces[oid]["workspace_id"]
            elif oid in child_to_source:
                sid = child_to_source[oid]
                wid = workspaces[sid]["workspace_id"]
                node["role"] = "child"
                node["parent_id"] = wid
                node["parent_type"] = "workspace"
                node["workspace_id"] = wid
            else:
                node["role"] = "standalone"
                node["parent_id"] = rid
                node["parent_type"] = "room"

            graph["objects"].append(node)

        graph["rooms"].append({
            "room_id": rid,
            "room_label": label["room_label"],
            "workspace_ids": [
                w["workspace_id"] for w in workspaces.values()
            ],
            "source_object_ids": sorted(source_ids),
            "standalone_object_ids": sorted(standalone_ids),
            "direct_object_ids": sorted(source_ids | standalone_ids),
        })
        graph["workspaces"].extend(workspaces.values())

    return graph
