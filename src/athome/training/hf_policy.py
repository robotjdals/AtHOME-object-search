"""Planner policy on a local Hugging Face model with LoRA adapters (GRPO).

The planner answers ``{"selected_id": "<alias>"}`` (athome.inference.prompts).
Instead of free generation, every candidate's answer is scored: its log
probability given the prompt (sum over the answer tokens, rendered exactly as
SFT and the vLLM server render it: chat template with the generation prompt,
thinking disabled, answer + end-of-turn token). Normalized over the candidates
this is the policy over the valid actions, the action-scoring form of an LLM
policy in online RL (GLAM, Carta et al., ICML 2023), which never produces an
answer outside the candidates and gives exact action probabilities for the
GRPO ratio and KL. The robot's vLLM server restricts decoding to the
candidate aliases (JSON schema enum), the same action set.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from athome.inference.prompts import build_messages
from athome.search.policy import Candidate, PlanningContext, Stage


def prompt_text(tokenizer, messages, enable_thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=enable_thinking)


def answer_text(alias: str) -> str:
    return json.dumps({"selected_id": alias})


def candidate_logprobs(model, tokenizer, prompt: str, aliases: Sequence[str], shared_prefix: bool = True):
    """Tensor [len(aliases)]: log p(answer_a | prompt) under the active adapter.

    ``shared_prefix`` (default): the prompt is encoded once and its key/value
    cache is reused for every candidate's answer (prefix caching, as vLLM
    does for shared prompts), so the cost grows with the answer length only;
    the same quantity as encoding prompt + answer per candidate
    (``shared_prefix=False``), up to floating-point rounding. Gradients flow
    through both passes."""
    if shared_prefix:
        return _candidate_logprobs_cached(model, tokenizer, prompt, aliases)
    return _candidate_logprobs_full(model, tokenizer, prompt, aliases)


def _answer_ids(tokenizer, aliases):
    return [tokenizer(answer_text(a) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            for a in aliases]


def _candidate_logprobs_cached(model, tokenizer, prompt, aliases):
    import torch
    device = next(model.parameters()).device
    prompt_ids = torch.tensor([tokenizer(prompt, add_special_tokens=False)["input_ids"]], device=device)
    answers = _answer_ids(tokenizer, aliases)
    n, width, length = len(answers), max(len(a) for a in answers), prompt_ids.shape[1]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    first = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    cache = first.past_key_values
    cache.batch_repeat_interleave(n)
    ids = torch.full((n, width), pad, dtype=torch.long, device=device)
    mask = torch.zeros((n, length + width), dtype=torch.long, device=device)
    mask[:, :length] = 1
    for i, a in enumerate(answers):
        ids[i, :len(a)] = torch.tensor(a, device=device)
        mask[i, length:length + len(a)] = 1
    rest = model(input_ids=ids, attention_mask=mask, past_key_values=cache).logits.float()
    head = torch.log_softmax(first.logits[:, -1].float(), dim=-1)[0]        # predicts answer token 0
    logp = torch.log_softmax(rest, dim=-1)                                  # row j predicts token j + 1
    out = []
    for i, a in enumerate(answers):
        targets = torch.tensor(a, device=device)
        total = head[targets[0]]
        if len(a) > 1:
            total = total + logp[i, torch.arange(len(a) - 1, device=device), targets[1:]].sum()
        out.append(total)
    return torch.stack(out)


def batch_candidate_logprobs(model, tokenizer, prompts: Sequence[str], alias_lists: Sequence[Sequence[str]],
                             max_prompt_tokens: int = 8192, max_answer_rows: int = 64):
    """``candidate_logprobs`` for many decisions at once: list of tensors, one
    per decision. Prompts are left-padded with explicit position ids (as
    batched generation does) and encoded together; each prompt's key/value
    cache is repeated for its candidates, whose answers are then encoded
    together. Decisions are processed in chunks bounded by prompt tokens
    (padded) and answer rows, to fit GPU memory. Same values as scoring each
    decision alone, up to floating-point rounding; gradients flow."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    encoded = {i: tokenizer(prompts[i], add_special_tokens=False)["input_ids"] for i in order}
    answers = {i: _answer_ids(tokenizer, alias_lists[i]) for i in order}
    results, chunk = [None] * len(prompts), []

    def flush():
        if chunk:
            for i, value in zip(chunk, _score_chunk(model, tokenizer, [encoded[i] for i in chunk],
                                                    [answers[i] for i in chunk])):
                results[i] = value
            chunk.clear()

    for i in order:
        width = max([len(encoded[j]) for j in chunk] + [len(encoded[i])])
        rows = sum(len(answers[j]) for j in chunk) + len(answers[i])
        if chunk and (width * (len(chunk) + 1) > max_prompt_tokens or rows > max_answer_rows):
            flush()
        chunk.append(i)
    flush()
    return results


def _score_chunk(model, tokenizer, prompt_ids, answer_ids):
    import torch
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    k, width = len(prompt_ids), max(len(p) for p in prompt_ids)
    ids = torch.full((k, width), pad, dtype=torch.long, device=device)
    mask = torch.zeros((k, width), dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids):
        ids[i, width - len(p):] = torch.tensor(p, device=device)          # left padding
        mask[i, width - len(p):] = 1
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    first = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, logits_to_keep=1)
    heads = torch.log_softmax(first.logits[:, -1].float(), dim=-1)       # predicts answer token 0
    owner = [i for i, a in enumerate(answer_ids) for _ in a]
    flat = [ans for a in answer_ids for ans in a]
    cache = first.past_key_values
    cache.reorder_cache(torch.tensor(owner, device=device))
    rows, awidth = len(flat), max(len(a) for a in flat)
    aids = torch.full((rows, awidth), pad, dtype=torch.long, device=device)
    amask = torch.zeros((rows, width + awidth), dtype=torch.long, device=device)
    apos = torch.zeros((rows, awidth), dtype=torch.long, device=device)
    for r, a in enumerate(flat):
        aids[r, :len(a)] = torch.tensor(a, device=device)
        amask[r, :width] = mask[owner[r]]
        amask[r, width:width + len(a)] = 1
        apos[r] = len(prompt_ids[owner[r]]) + torch.arange(awidth, device=device)
    rest = torch.log_softmax(model(input_ids=aids, attention_mask=amask, position_ids=apos,
                                   past_key_values=cache).logits.float(), dim=-1)   # row j predicts token j+1
    totals = []
    for r, a in enumerate(flat):
        targets = torch.tensor(a, device=device)
        total = heads[owner[r], targets[0]]
        if len(a) > 1:
            total = total + rest[r, torch.arange(len(a) - 1, device=device), targets[1:]].sum()
        totals.append(total)
    out, start = [], 0
    for a in answer_ids:
        out.append(torch.stack(totals[start:start + len(a)]))
        start += len(a)
    return out


def _candidate_logprobs_full(model, tokenizer, prompt, aliases):
    import torch
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answers = _answer_ids(tokenizer, aliases)
    width = len(prompt_ids) + max(len(a) for a in answers)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    ids = torch.full((len(answers), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, a in enumerate(answers):
        seq = prompt_ids + a
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        mask[i, :len(seq)] = 1
    start = len(prompt_ids)
    # Only the positions that predict answer tokens (start-1 ... width-2) are
    # needed; logits_to_keep avoids the full vocabulary x prompt logits.
    keep = width - start + 1
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits.float()
    logp = torch.log_softmax(logits, dim=-1)          # row j predicts token start + j
    out = []
    for i, a in enumerate(answers):
        positions = torch.arange(len(a), device=device)
        targets = torch.tensor(a, device=device)
        out.append(logp[i, positions, targets].sum())
    return torch.stack(out)


# Wall-clock seconds spent in model scoring by all HFCandidatePolicy objects
# (profiling of GRPO steps: simulation vs scoring vs update).
SCORING_SECONDS = [0.0]


class HFCandidatePolicy:
    """Selects among candidates with one adapter of a PeftModel.

    ``sample``: draw from the candidate distribution (GRPO rollouts of the
    trained stage); otherwise take its mode (fixed stages). After ``select``,
    ``last_messages``, ``last_aliases``, ``last_output`` (the answer text) and
    ``last_logprob`` (log pi(chosen) normalized over the candidates) describe
    the decision."""

    def __init__(self, model, tokenizer, adapter: str, sample: bool = False,
                 generator=None, enable_thinking: bool = False, decoding: str = "score",
                 scorer: "LockstepScorer" = None):
        if decoding not in ("score", "greedy"):
            raise ValueError(f"알 수 없는 decoding: {decoding}")
        self.decoding = decoding
        self.scorer = scorer              # batched scoring across concurrent rollouts
        self.model, self.tokenizer, self.adapter = model, tokenizer, adapter
        self.sample, self.generator, self.enable_thinking = sample, generator, enable_thinking
        self.last_messages: Optional[List[dict]] = None
        self.last_aliases: Optional[Dict[str, str]] = None
        self.last_output: Optional[str] = None
        self.last_logprob: Optional[float] = None

    def select(self, stage: Stage, target: str, candidates: Sequence[Candidate],
               context: PlanningContext) -> str:
        import torch
        import time
        if self.scorer is not None:            # timed by the scorer (requests overlap)
            return self._select(stage, target, candidates, context)
        t0 = time.time()
        try:
            return self._select(stage, target, candidates, context)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            SCORING_SECONDS[0] += time.time() - t0

    def _select(self, stage, target, candidates, context):
        import torch
        messages, aliases = build_messages(stage, target, candidates, context)
        if self.scorer is None:                # with a scorer only its thread switches adapters
            self.model.set_adapter(self.adapter)
        if self.decoding == "greedy" and not self.sample:
            if self.scorer is not None:
                raise ValueError("greedy 디코딩은 LockstepScorer와 함께 쓸 수 없습니다.")
            alias = greedy_constrained(self.model, self.tokenizer,
                                       prompt_text(self.tokenizer, messages, self.enable_thinking), list(aliases))
            self.last_messages, self.last_aliases = messages, aliases
            self.last_output, self.last_logprob = answer_text(alias), None
            return aliases[alias]
        if self.scorer is not None:
            logpi = self.scorer.request(self.adapter, messages, aliases)
        else:
            with torch.no_grad():
                scores = candidate_logprobs(self.model, self.tokenizer,
                                            prompt_text(self.tokenizer, messages, self.enable_thinking),
                                            list(aliases))
            logpi = torch.log_softmax(scores, dim=0)
        if self.sample:
            # Sampled on the CPU with the seeded CPU generator (the scores may be on the GPU).
            k = int(torch.multinomial(logpi.exp().float().cpu(), 1, generator=self.generator).item())
        else:
            k = int(torch.argmax(logpi).item())
        alias = list(aliases)[k]
        self.last_messages, self.last_aliases = messages, aliases
        self.last_output, self.last_logprob = answer_text(alias), float(logpi[k])
        return aliases[alias]


def greedy_constrained(model, tokenizer, prompt: str, aliases: Sequence[str]) -> str:
    """The answer the robot's vLLM server would give: temperature 0, one
    token at a time, only tokens that continue some candidate answer (the
    JSON-schema enum of athome.inference.llm_policy). Digits are single tokens
    in the Qwen tokenizer, so with ten or more candidates this can differ from
    the best whole answer (``candidate_logprobs`` argmax)."""
    import torch
    answers = {a: tokenizer(answer_text(a) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
               for a in aliases}
    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    live = dict(answers)
    pos = 0
    # Key/value cache of the prompt and the answer tokens fed so far; tokens
    # with a single allowed choice are appended without a forward and fed
    # together before the next branching point.
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=device), use_cache=True, logits_to_keep=1)
    cache, logits, pending = out.past_key_values, out.logits[0, -1], []
    while True:
        done = [a for a, seq in live.items() if len(seq) == pos]
        if done:
            return done[0]
        allowed = sorted({seq[pos] for seq in live.values()})
        if len(allowed) == 1:
            nxt = allowed[0]
        else:
            if pending:
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([pending], device=device), past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
                cache, logits, pending = out.past_key_values, out.logits[0, -1], []
            nxt = allowed[int(torch.argmax(logits[allowed]).item())]
        live = {a: seq for a, seq in live.items() if seq[pos] == nxt}
        pending.append(nxt)
        pos += 1


class LockstepScorer:
    """Batches the scoring requests of concurrent rollouts (actor-learner
    split, as SEED RL: environments step on CPU threads, one learner scores
    on the GPU). Each round waits until every live rollout has either asked
    for a decision or finished, then scores all waiting requests together,
    ordered by rollout index, so a run is reproducible. Returns log pi over
    the candidates (normalized) on the CPU."""

    def __init__(self, model, tokenizer, enable_thinking: bool = False):
        import threading
        self.model, self.tokenizer, self.enable_thinking = model, tokenizer, enable_thinking
        self._cond = threading.Condition()
        self._local = threading.local()
        self._pending, self._results, self._done, self._n = {}, {}, set(), 0

    def request(self, adapter, messages, aliases):
        wid = self._local.worker
        with self._cond:
            self._pending[wid] = (adapter, messages, list(aliases))
            self._cond.notify_all()
            self._cond.wait_for(lambda: wid in self._results)
            return self._results.pop(wid)

    def run(self, jobs):
        """Run ``jobs`` (callables) concurrently; returns their results in order."""
        import threading
        import torch
        self._n, self._done, outputs, errors = len(jobs), set(), [None] * len(jobs), []

        def worker(k, job):
            self._local.worker = k
            try:
                outputs[k] = job()
            except BaseException as e:  # noqa: BLE001 - re-raised in the caller
                errors.append(e)
            finally:
                with self._cond:
                    self._done.add(k)
                    self._cond.notify_all()

        threads = [threading.Thread(target=worker, args=(k, j), daemon=True) for k, j in enumerate(jobs)]
        for t in threads:
            t.start()
        while True:
            with self._cond:
                self._cond.wait_for(lambda: len(self._pending) + len(self._done) >= self._n)
                if not self._pending:
                    break
                batch = sorted(self._pending.items())
                self._pending = {}
            import time
            t0 = time.time()
            by_adapter = {}
            for wid, (adapter, messages, aliases) in batch:
                by_adapter.setdefault(adapter, []).append((wid, messages, aliases))
            results = {}
            for adapter, items in sorted(by_adapter.items()):
                self.model.set_adapter(adapter)
                with torch.no_grad():
                    scores = batch_candidate_logprobs(
                        self.model, self.tokenizer,
                        [prompt_text(self.tokenizer, m, self.enable_thinking) for _, m, _ in items],
                        [a for _, _, a in items])
                for (wid, _, _), s in zip(items, scores):
                    results[wid] = torch.log_softmax(s.float(), dim=0).cpu()
            SCORING_SECONDS[0] += time.time() - t0
            with self._cond:
                self._results.update(results)
                self._cond.notify_all()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return outputs
