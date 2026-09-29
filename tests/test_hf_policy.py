"""Candidate scoring vs robot-style greedy decoding (athome.training.hf_policy)."""
import math

import pytest

torch = pytest.importorskip("torch")

from athome.training.hf_policy import candidate_logprobs, greedy_constrained  # noqa: E402

VOCAB = [chr(c) for c in range(32, 127)]


class CharTokenizer:
    eos_token = "~"
    pad_token_id = None
    eos_token_id = VOCAB.index("~")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [VOCAB.index(ch) for ch in text]}


class Bigram(torch.nn.Module):
    """Next-token probabilities depend on the last character only."""

    def __init__(self, table):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.table = table

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
        logits = torch.full((*input_ids.shape, len(VOCAB)), -30.0)
        for b in range(input_ids.shape[0]):
            for t in range(input_ids.shape[1]):
                probs = self.table.get(VOCAB[input_ids[b, t]], {})
                if probs:
                    for ch, p in probs.items():
                        logits[b, t, VOCAB.index(ch)] = math.log(p)
                else:                                     # deterministic JSON scaffolding
                    logits[b, t, :] = 0.0
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return type("Out", (), {"logits": logits})()


def test_greedy_can_differ_from_the_best_whole_answer():
    # After "C": digit 1 (C1, C10, C11) 0.6, digit 2 (C2) 0.4.
    # After "1": 0 or 1 0.45 each, end of alias 0.1.  After "0"/"2": end.
    table = {"C": {"1": 0.6, "2": 0.4}, "1": {"0": 0.45, "1": 0.45, '"': 0.10},
             "0": {'"': 1.0}, "2": {'"': 1.0}}
    model, tok = Bigram(table), CharTokenizer()
    aliases = ["C1", "C2", "C10", "C11"]
    scores = candidate_logprobs(model, tok, "prompt:", aliases)
    assert aliases[int(torch.argmax(scores))] == "C2"            # whole answer: C2 0.40 > C10 0.27 > C1 0.06 > C11 0.03
    assert greedy_constrained(model, tok, "prompt:", aliases) in ("C10", "C11")   # first digit "1" (0.6)
