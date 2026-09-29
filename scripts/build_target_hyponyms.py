"""Subtypes that count as a target (e.g. "bedside lamp" is a "lamp").

A person asking a robot to find a lamp is satisfied by a bedside lamp, so the
instances of a subtype category are ground truth for its parent target and
are masked with it. This extends the agreed alias rule of the fixed targets
(2026-09-26: portable subtypes such as table lamp -> lamp) to the open
vocabulary. Candidate pairs are the listed categories whose name is another
listed category with a modifier in front ("bedside lamp" -> "lamp"); whether
the pair really is a kind-of relation ("paper towel" is not a towel) is a
semantic judgement made by GPT-4.1 with the labeling protocol (9 samples,
majority), as the searchable filter of scripts/build_target_categories.py.
Writes a new category file: the input plus ``hyponyms`` {parent: [subtypes]}.
"""
import argparse
import json
from pathlib import Path

from build_target_categories import MODEL

QUESTION = ("For each pair of household object category names, answer whether the first "
            "names a kind of the second, so that a person who asks a home robot to find the "
            "second would be satisfied if it finds the first (e.g. 'bedside lamp' is a kind of "
            "'lamp': true; 'paper towel' is a kind of 'towel': false, a paper towel is not what "
            "people mean by a towel). Answer only with JSON.")


def candidate_pairs(names):
    """(child, parent): parent = the longest listed word suffix of child."""
    listed = set(names)
    pairs = []
    for child in listed:
        words = child.split(" ")
        for k in range(1, len(words)):
            head = " ".join(words[k:])
            if head in listed:
                pairs.append((child, head))
                break
    return sorted(pairs)


def is_kind_of(pairs, votes, chunk=60):
    """(child, parent) -> (majority yes, yes votes) with n samples at temperature 1."""
    from athome.inference.chat import ChatJSON
    client = ChatJSON("https://api.openai.com", MODEL, api_key_env="OPENAI_API_KEY",
                      timeout=120.0, schema_name="kind_of_pairs", retries=5, backoff_s=2.0)
    out = {}
    for i in range(0, len(pairs), chunk):
        part = pairs[i:i + chunk]
        keys = [f"{c} | {p}" for c, p in part]
        schema = {"type": "object", "additionalProperties": False, "required": ["items"],
                  "properties": {"items": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False, "required": ["pair", "kind_of"],
                      "properties": {"pair": {"type": "string", "enum": keys},
                                     "kind_of": {"type": "boolean"}}}}}}
        messages = [{"role": "system", "content": QUESTION},
                    {"role": "user", "content": json.dumps(
                        {"pairs (first | second)": keys})}]
        samples = client.sample(messages, schema, n=votes, temperature=1.0)
        for key, pair in zip(keys, part):
            yes = sum(any(it["pair"] == key and it["kind_of"] for it in s["items"]) for s in samples)
            out[pair] = (yes * 2 > votes, yes)
    print("API 사용량:", client.usage)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", type=Path, required=True, help="build_target_categories.py 출력")
    parser.add_argument("--votes", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"이미 존재하는 출력: {args.output}")
    data = json.loads(args.categories.read_text(encoding="utf-8"))
    pairs = candidate_pairs(data["categories"])
    verdicts = is_kind_of(pairs, args.votes)
    hyponyms, rejected = {}, []
    for (child, parent), (ok, yes) in sorted(verdicts.items()):
        (hyponyms.setdefault(parent, []).append(child) if ok else rejected.append(f"{child} | {parent}"))
    result = {**data, "version": "1.2", "hyponyms": hyponyms,
              "hyponym_rule": {"candidates": "listed category = modifier + listed category",
                               "model": MODEL, "votes": args.votes, "question": QUESTION,
                               "votes_by_pair": {f"{c} | {p}": yes for (c, p), (_, yes) in sorted(verdicts.items())},
                               "rejected": rejected,
                               "source": str(args.categories)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"후보 쌍 {len(pairs)}개 → 하위 종류 인정 {sum(map(len, hyponyms.values()))}개, "
          f"거부 {len(rejected)}개 → {args.output}")
    print(json.dumps(hyponyms, ensure_ascii=False))


if __name__ == "__main__":
    main()
