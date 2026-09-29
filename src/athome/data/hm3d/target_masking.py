from copy import deepcopy
import re


def normalize_tag(tag):
    return " ".join(tag.casefold().split())


def target_terms(config, target):
    """Normalized raw tags that count as ``target``: its name and plural, raw
    names aliased to it, and its members (subtypes such as "bedside lamp" for
    "lamp", scripts/build_target_hyponyms.py) from the targets file."""
    target = normalize_tag(target)
    terms = {target, target + "s"}
    terms.update(normalize_tag(k) for k, v in config.get("category_aliases", {}).items()
                 if normalize_tag(v) == target)
    terms.update(normalize_tag(m) for m in config.get("target_members", {}).get(target, []))
    return terms


def mask_target_inputs(semantic_inputs, target, aliases, members=()):
    """``members``: further raw tags that count as the target (subtypes)."""
    target = normalize_tag(target)
    aliases = {
        normalize_tag(k): normalize_tag(v)
        for k, v in aliases.items()
    }
    member_tags = {normalize_tag(m) for m in members}

    # Explicit target words and configured aliases, not substring matching.
    terms = {target, target + "s"}
    terms.update(k for k, v in aliases.items() if v == target)
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(t) for t in sorted(terms))
        + r")(?!\w)"
    )

    def reason_for(tag):
        normalized = normalize_tag(tag)
        canonical = aliases.get(normalized, normalized)
        if canonical == target:
            return "target_category"
        if normalized in member_tags:
            return "target_subtype"
        # Explicit contents labels reveal target presence.
        # Related categories alone are useful context, not target evidence.
        content_leak_tags = {
            "book": {"box with books", "basket with books"},
        }
        if normalized in content_leak_tags.get(target, set()):
            return "explicit_target_content"
        return None

    masked = {
        "schema_version": "0.1",
        "purpose": "masked_scene_construction",
        "rooms": [],
    }
    audit = {
        "target_category": target,
        "mask_policy": "exact_category_and_explicit_contents_v2",
        "removed_objects": [],
        "changed_room_ids": [],
    }

    global_ids = set()
    for room in semantic_inputs["rooms"]:
        rid = room["room_id"]
        objects = room["objects"]
        ids = {obj["object_id"] for obj in objects}
        if len(ids) != len(objects) or ids & global_ids:
            raise ValueError(f"{rid}: duplicate object ID")
        global_ids.update(ids)

        removed_ids = set()
        for obj in objects:
            reason = reason_for(obj["semantic_tag"])
            if reason:
                removed_ids.add(obj["object_id"])
                audit["removed_objects"].append({
                    "room_id": rid,
                    "object_id": obj["object_id"],
                    "semantic_tag": obj["semantic_tag"],
                    "reason": reason,
                })

        if removed_ids:
            audit["changed_room_ids"].append(rid)

        kept_ids = ids - removed_ids
        kept_objects = []

        for obj in objects:
            if obj["object_id"] in removed_ids:
                continue

            neighbors = obj["neighbors"]
            children = obj["child_candidate_ids"]

            if not {n["object_id"] for n in neighbors} <= ids:
                raise ValueError(f"{rid}: invalid neighbor reference")
            if not set(children) <= ids:
                raise ValueError(f"{rid}: invalid child reference")

            clean = deepcopy(obj)
            clean["neighbors"] = [
                n for n in clean["neighbors"]
                if n["object_id"] in kept_ids
            ]
            clean["child_candidate_ids"] = [
                oid for oid in children if oid in kept_ids
            ]

            if reason_for(clean["semantic_tag"]):
                raise ValueError("Target category remains after masking")
            kept_objects.append(clean)

        # Carry only raw room input, not labels inferred before masking.
        clean_room = {
            "room_id": rid,
            "coordinate_frame": room["coordinate_frame"],
            "up_axis": room["up_axis"],
            "length_unit": room["length_unit"],
            "geometry_thresholds": deepcopy(room["geometry_thresholds"]),
            "objects": kept_objects,
        }
        masked["rooms"].append(clean_room)

    return masked, audit
