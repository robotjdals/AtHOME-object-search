"""Target categories for object search from HM3DSem train scenes (open vocabulary).

As HM3D-OVON (379 HM3DSem categories, held-out categories for evaluation)
and HomeRobot OVMM (129 object categories, seen/unseen split), targets are
the household object categories that occur in the data, not a fixed short
list:
- raw names -> canonical category with the official HM3DSem mapping (it also
  merges spellings, e.g. "cair" -> "chair");
- every category with a label (mpcat40 other than "unlabeled"); no class
  subset or size threshold is chosen by hand;
- present in at least ``--min-scenes`` train scenes (enough examples);
- plural names are merged into the singular category when it exists
  ("bags" -> "bag", "boxes" -> "box");
- whether a category is an object a person would ask a home robot to find
  (a movable household item, not a fixture such as a tap, switch or smoke
  detector) is a semantic judgement: GPT-4.1 answers it with the scene-graph
  labeling protocol (9 samples, majority), once per category name
  (``--llm-filter``; external API call);
- ``--unseen-fraction`` of the categories, sampled uniformly (seeded), is held
  out of training and used only for Val/Test. As HM3D-OVON, the held-out
  categories are split with SentenceBERT cosine similarity to the training
  categories: above ``--synonym-threshold`` (0.68, OVON Table II) they are
  "synonyms" of a seen category (couch -> sofa), otherwise "unseen". OVON
  does not name its SentenceBERT model; we use all-mpnet-base-v2, the
  general-purpose model recommended by sentence-transformers.
Inputs are the scoped room-object maps of the train scenes (objects inside
storage furniture already removed).
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-4.1-2025-04-14"
SBERT = "sentence-transformers/all-mpnet-base-v2"


def normalize(tag: str) -> str:
    return " ".join(tag.casefold().split())


def load_mapping(path: Path):
    """normalized raw name -> (canonical category, mpcat40)."""
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out.setdefault(normalize(row["raw_category"]), (normalize(row["category"]), row["mpcat40"]))
    return out


QUESTION = ("For each object category name from 3D scans of homes, answer whether it is an "
            "object a person would ask a home robot to find: a movable household item that "
            "can be misplaced (e.g. mug, bag, remote, towel, lamp). Answer false for fixtures "
            "and installations (tap, light switch, smoke detector, vent, radiator), building "
            "parts, and furniture. Also answer false when the name is not a specific object a "
            "person would ask for by name: generic groups or vague descriptions (e.g. appliance, "
            "container, device, decoration, clutter, item, stuff, something) and names whose "
            "meaning is ambiguous between a movable item and furniture. Answer only with JSON.")


def searchable(names, votes, chunk=60):
    """name -> (majority yes, yes votes) with n samples at temperature 1."""
    from athome.inference.chat import ChatJSON
    client = ChatJSON("https://api.openai.com", MODEL, api_key_env="OPENAI_API_KEY",
                      timeout=120.0, schema_name="searchable_objects", retries=5, backoff_s=2.0)
    out = {}
    for i in range(0, len(names), chunk):
        part = names[i:i + chunk]
        schema = {"type": "object", "additionalProperties": False, "required": ["items"],
                  "properties": {"items": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False, "required": ["name", "searchable"],
                      "properties": {"name": {"type": "string", "enum": part},
                                     "searchable": {"type": "boolean"}}}}}}
        messages = [{"role": "system", "content": QUESTION},
                    {"role": "user", "content": json.dumps({"categories": part})}]
        samples = client.sample(messages, schema, n=votes, temperature=1.0)
        for name in part:
            yes = sum(any(it["name"] == name and it["searchable"] for it in s["items"]) for s in samples)
            out[name] = (yes * 2 > votes, yes)
    print("API 사용량:", client.usage)
    return out


def split_held_out(held_out, seen, threshold):
    """OVON split of held-out categories: {name: (split, nearest seen, similarity)}."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT)
    a = model.encode(list(held_out), normalize_embeddings=True)
    b = model.encode(list(seen), normalize_embeddings=True)
    sim = a @ b.T
    out = {}
    for i, name in enumerate(held_out):
        j = int(sim[i].argmax())
        out[name] = ("synonyms" if sim[i, j] > threshold else "unseen", seen[j], round(float(sim[i, j]), 3))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=Path, nargs="+", required=True, help="train 장면 scoped room-object map")
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/data/hm3dsem_category_mappings.tsv")
    parser.add_argument("--min-scenes", type=int, default=3)
    parser.add_argument("--unseen-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm-filter", action="store_true", help="GPT-4.1 다수결로 찾을 물건인지 판정(외부 API)")
    parser.add_argument("--llm-votes", type=int, default=9)
    parser.add_argument("--reuse-votes", type=Path, nargs="+",
                        help="이전 출력들의 투표 재사용. 여러 개면 모든 질문에서 다수결을 통과해야 남음")
    parser.add_argument("--synonym-threshold", type=float, default=0.68)
    parser.add_argument("--held-out-from", type=Path,
                        help="이전 목록의 평가 전용 범주를 유지(남은 범주로 한정). 이미 라벨링한 장면과 split 일치")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    mapping = load_mapping(args.mapping)
    scenes, sizes, raw_names, classes = defaultdict(set), defaultdict(list), defaultdict(set), {}
    unmapped = defaultdict(int)
    for path in args.maps:
        data = json.loads(path.read_text(encoding="utf-8"))
        objects = data["objects"] if "objects" in data else [o for r in data["rooms"] for o in r["objects"]]
        for obj in objects:
            raw = normalize(obj["semantic_tag"])
            if raw not in mapping:
                unmapped[raw] += 1
                continue
            category, mpcat = mapping[raw]
            if mpcat == "unlabeled" or not obj.get("bbox"):
                continue
            scenes[category].add(path.name.split(".")[0])
            sizes[category].append(max(obj["bbox"]["size"]))
            raw_names[category].add(raw)
            classes[category] = mpcat
    # Plurals: merge into the singular category when it occurs too.
    merged = {}
    for category in sorted(scenes):
        for suffix in ("es", "s"):
            if category.endswith(suffix) and category[:-len(suffix)] in scenes:
                merged[category] = category[:-len(suffix)]
                break
    for plural, singular in merged.items():
        scenes[singular] |= scenes.pop(plural)
        sizes[singular] += sizes.pop(plural)
        raw_names[singular] |= raw_names.pop(plural)
    kept, dropped = {}, {"too_rare": [], "not_searchable": []}
    for category in sorted(scenes):
        median = statistics.median(sizes[category])   # recorded for reference only
        if len(scenes[category]) < args.min_scenes:
            dropped["too_rare"].append(category)
        else:
            kept[category] = {"mpcat40": classes[category], "scenes": len(scenes[category]),
                              "instances": len(sizes[category]), "median_max_size_m": round(median, 3),
                              "raw_names": sorted(raw_names[category])}
    if not args.llm_filter and not args.reuse_votes:
        print("주의: --llm-filter 없이 만들면 설비·가구도 목표에 남습니다(시험용).")
    llm_filter = None
    if args.reuse_votes:
        # Each file is one question answered by the same model and vote count.
        # With several files (e.g. a question and its stricter refinement) a
        # category must pass the majority of every question.
        filters, per_file = [], []
        for path in args.reuse_votes:
            previous = json.loads(path.read_text(encoding="utf-8"))
            f = previous["llm_filter"]
            if f is None or f["model"] != MODEL or f["votes"] != args.llm_votes:
                raise SystemExit(f"{path}: 모델·표 수가 다르거나 투표가 없습니다.")
            # Dropped categories carry no count in the file; only the majority "no" matters.
            votes = {n: v["searchable_votes"] for n, v in previous["categories"].items()}
            votes.update({n: 0 for n in previous["dropped"]["not_searchable"]})
            missing = sorted(set(kept) - set(votes))
            if missing:
                raise SystemExit(f"{path}: 투표가 없는 범주 {missing[:10]}")
            filters.append({**f, "source": str(path)})
            per_file.append(votes)
        verdicts = {n: (all(v[n] * 2 > args.llm_votes for v in per_file), min(v[n] for v in per_file))
                    for n in sorted(kept)}
        llm_filter = filters[0] if len(filters) == 1 else {"all_of": filters}
        args.llm_filter = True
    elif args.llm_filter:
        verdicts = searchable(sorted(kept), args.llm_votes)
        llm_filter = {"model": MODEL, "votes": args.llm_votes, "question": QUESTION}
    if args.llm_filter:
        for name, (ok, votes) in verdicts.items():
            kept[name]["searchable_votes"] = votes
            if not ok:
                dropped["not_searchable"].append(name)
                del kept[name]
    names = sorted(kept)
    if args.held_out_from:
        # Removing categories by a criterion independent of the uniform draw
        # leaves the remaining held-out set a uniform sample of the new list.
        previous = json.loads(args.held_out_from.read_text(encoding="utf-8"))
        held_out = sorted((set(previous.get("synonyms", [])) | set(previous["unseen"])) & set(names))
    else:
        held_out = sorted(random.Random(args.seed).sample(names, round(len(names) * args.unseen_fraction)))
    seen = [n for n in names if n not in held_out]
    ovon = split_held_out(held_out, seen, args.synonym_threshold)
    synonyms = [n for n in held_out if ovon[n][0] == "synonyms"]
    unseen = [n for n in held_out if ovon[n][0] == "unseen"]
    for name, (_, nearest, sim) in ovon.items():
        kept[name]["nearest_seen"] = {"category": nearest, "sbert_cosine": sim}
    result = {
        "version": "1.1", "purpose": "open-vocabulary object search targets",
        "rule": {"candidates": "all labeled categories", "searchable": "GPT-4.1 majority vote",
                 "min_train_scenes": args.min_scenes, "unseen_fraction": args.unseen_fraction,
                 "seed": args.seed,
                 "held_out_from": str(args.held_out_from) if args.held_out_from else None,
                 "held_out_split": {"model": SBERT, "synonym_threshold": args.synonym_threshold,
                                    "reference": "HM3D-OVON (Yokoyama et al. 2024) Sec. III-B, Table II"}},
        "sources": {"mapping": str(args.mapping), "mapping_sha256": hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
                    "train_maps": len(args.maps)},
        "seen": seen, "synonyms": synonyms, "unseen": unseen,
        "categories": kept, "dropped": dropped, "merged_plurals": merged,
        "llm_filter": llm_filter,
        "unmapped_raw_names": dict(sorted(unmapped.items(), key=lambda kv: -kv[1])[:50]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"범주 {len(names)}개 (학습 {len(seen)}, 동의어 평가 {len(synonyms)}, 미공개 {len(unseen)}), "
          f"제외: 드묾 {len(dropped['too_rare'])}, 찾을 물건 아님 {len(dropped['not_searchable'])} → {args.output}")


if __name__ == "__main__":
    main()
