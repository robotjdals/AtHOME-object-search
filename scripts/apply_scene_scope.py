"""Remove out-of-scope objects (stored inside storage furniture) from a
z-up room-object map. See src/athome/data/hm3d/scope.py."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from athome.data.hm3d.scope import apply_storage_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    scope = json.loads(args.config.read_text(encoding="utf-8"))
    result, contained = apply_storage_scope(source, scope)
    result["source_unscoped_map"] = str(args.input.resolve())
    result["source_unscoped_map_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    by_id = {o["object_id"]: o for o in source["objects"]}
    print(f"범위 밖 객체 {len(contained)}개 제외 → 유지 {len(result['objects'])}개")
    print("제외 범주:", Counter(by_id[o]["semantic_tag"] for o in contained).most_common(12))
    print("감싼 가구:", Counter(by_id[c]["semantic_tag"] for c in contained.values()).most_common(12))
    print("저장:", args.output)


if __name__ == "__main__":
    main()
