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

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None, past_key_values=None,
                use_cache=False):
        # Bigram: the next-token distribution needs only the last token, so a
        # key/value cache has nothing to store (a placeholder is returned).
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
        return type("Out", (), {"logits": logits, "past_key_values": object()})()


def test_greedy_can_differ_from_the_best_whole_answer():
    # After "C": digit 1 (C1, C10, C11) 0.6, digit 2 (C2) 0.4.
    # After "1": 0 or 1 0.45 each, end of alias 0.1.  After "0"/"2": end.
    table = {"C": {"1": 0.6, "2": 0.4}, "1": {"0": 0.45, "1": 0.45, '"': 0.10},
             "0": {'"': 1.0}, "2": {'"': 1.0}}
    model, tok = Bigram(table), CharTokenizer()
    aliases = ["C1", "C2", "C10", "C11"]
    scores = candidate_logprobs(model, tok, "prompt:", aliases, shared_prefix=False)   # the mock has no KV cache
    assert aliases[int(torch.argmax(scores))] == "C2"            # whole answer: C2 0.40 > C10 0.27 > C1 0.06 > C11 0.03
    assert greedy_constrained(model, tok, "prompt:", aliases) in ("C10", "C11")   # first digit "1" (0.6)


def test_prefix_cached_scoring_equals_full_scoring():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen3Config(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=512)
    model = transformers.Qwen3ForCausalLM(config).eval()
    tok = CharTokenizer()
    aliases = ["C1", "C2", "C10", "C12"]
    with torch.no_grad():
        full = candidate_logprobs(model, tok, "prompt: pick one", aliases, shared_prefix=False)
        cached = candidate_logprobs(model, tok, "prompt: pick one", aliases, shared_prefix=True)
    assert torch.allclose(full, cached, atol=1e-5)
    # Gradients through the cached path match the full path.
    g = []
    for shared in (False, True):
        model.zero_grad()
        candidate_logprobs(model, tok, "prompt: pick one", aliases, shared_prefix=shared).sum().backward()
        g.append(model.model.layers[0].mlp.up_proj.weight.grad.clone())
    assert torch.allclose(g[0], g[1], atol=1e-5)


def test_cached_greedy_equals_uncached_greedy_on_a_real_architecture():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(1)
    config = transformers.Qwen3Config(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=512)
    model = transformers.Qwen3ForCausalLM(config).eval()
    tok = CharTokenizer()
    aliases = [f"C{i}" for i in range(1, 14)]

    def reference(prompt):
        """Token by token without a cache (re-encodes the whole prefix)."""
        from athome.training.hf_policy import answer_text
        answers = {a: tok(answer_text(a) + tok.eos_token)["input_ids"] for a in aliases}
        ids, live, pos = tok(prompt)["input_ids"], dict(answers), 0
        while True:
            done = [a for a, s in live.items() if len(s) == pos]
            if done:
                return done[0]
            allowed = sorted({s[pos] for s in live.values()})
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
            nxt = allowed[int(torch.argmax(logits[allowed]))]
            live = {a: s for a, s in live.items() if s[pos] == nxt}
            ids.append(nxt)
            pos += 1

    for prompt in ("pick:", "which one?", "room candidates"):
        assert greedy_constrained(model, tok, prompt, aliases) == reference(prompt)


def test_batched_scoring_equals_single_scoring():
    transformers = pytest.importorskip("transformers")
    from athome.training.hf_policy import batch_candidate_logprobs
    torch.manual_seed(2)
    config = transformers.Qwen3Config(vocab_size=len(VOCAB), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=8, max_position_embeddings=512)
    model = transformers.Qwen3ForCausalLM(config).eval()
    tok = CharTokenizer()
    prompts = ["short", "a much longer prompt with more words", "mid length prompt"]
    alias_lists = [["C1", "C2"], [f"C{i}" for i in range(1, 12)], ["C1", "C2", "C3"]]
    with torch.no_grad():
        single = [candidate_logprobs(model, tok, p, a, shared_prefix=False) for p, a in zip(prompts, alias_lists)]
        for budget in ((8192, 64), (40, 5)):          # one chunk / several chunks
            batched = batch_candidate_logprobs(model, tok, prompts, alias_lists, *budget)
            for s, b in zip(single, batched):
                assert torch.allclose(s, b, atol=1e-5)
    # Gradients agree too.
    grads = []
    for batched_mode in (False, True):
        model.zero_grad()
        vals = (batch_candidate_logprobs(model, tok, prompts, alias_lists) if batched_mode else
                [candidate_logprobs(model, tok, p, a, shared_prefix=False) for p, a in zip(prompts, alias_lists)])
        sum(v.sum() for v in vals).backward()
        grads.append(model.model.layers[1].self_attn.q_proj.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5)
