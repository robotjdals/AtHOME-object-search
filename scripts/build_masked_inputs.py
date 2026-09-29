"""Target-masked semantic inputs and audits for every pilot target."""
import argparse
import json
from pathlib import Path

from athome.data.hm3d.target_masking import mask_target_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="<target>.semantic_inputs.json 출력")
    parser.add_argument("--audit-dir", type=Path, required=True, help="<target>.audit.json 출력")
    args = parser.parse_args()
    inputs = json.loads(args.semantic_inputs.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = [(t, *mask_target_inputs(inputs, t, config["category_aliases"],
                                       config.get("target_members", {}).get(t, ())))
               for t in config["target_categories"]]
    for d in (args.input_dir, args.audit_dir):
        d.mkdir(parents=True, exist_ok=True)
    for target, masked, audit in results:
        for path, value in ((args.input_dir / f"{target}.semantic_inputs.json", masked),
                            (args.audit_dir / f"{target}.audit.json", audit)):
            with path.open("x", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        print(f"{target}: 제거 {len(audit['removed_objects'])}, 변경 Room {audit['changed_room_ids']}")


if __name__ == "__main__":
    main()
