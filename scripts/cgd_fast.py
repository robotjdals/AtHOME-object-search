"""Two changes to perception's CG-D for the search's server, applied from
outside (scripts/serve_cgd_http.py, scripts/time_cgd_stages.py); their code
stays unchanged and the pre-map generation does not use this.

Whole names. GroundingDINO names a box by the caption tokens above its text
threshold and turns those into text, so a name of several tokens can come back
in pieces ("tumbler" = tumble ##r, "ketchup" = ke ##tch ##up): "##up" matches
no class and the box becomes "other item". A token's position says which name
it belongs to, as GLIP and MMDetection's Grounding DINO read token scores per
category span, so a phrase holding no whole name becomes the name most of its
tokens are in.

CLIP in batches. The original encodes each box's crop and its class name one
at a time (about 0.13 s per box), and the engine uses only the crop features.
The crops are made the same way and encoded together.
"""
import sys
from typing import List, Optional, Sequence

import numpy as np

BATCH = 32
PADDING = 20                    # conceptgraph.utils.model_utils.compute_clip_features


def whole_name_tokens(selected: Sequence[int], input_ids: Sequence[int], separators) -> Optional[List[int]]:
    """Token ids of the name the ``selected`` positions are a piece of, None
    if the phrase already holds a whole name (or no name's token). The earlier
    name when two hold as many."""
    spans, span = [], []
    for i, token in enumerate(input_ids):
        if token in separators:
            if span:
                spans.append(span)
            span = []
        else:
            span.append(i)
    if span:
        spans.append(span)
    chosen = set(selected)
    best, most = None, 0
    for span in spans:
        hits = len(chosen.intersection(span))
        if hits == len(span):
            return None
        if hits > most:
            best, most = span, hits
    return None if best is None else [input_ids[i] for i in best]


def crop_box(box, width, height, padding=PADDING):
    """The box with ``padding`` around it, inside the image."""
    x0, y0, x1, y1 = box
    return max(x0 - padding, 0), max(y0 - padding, 0), min(x1 + padding, width), min(y1 + padding, height)


def install(gsa, gsa_fast):
    """Replace GroundingDINO's phrase and CG-D's CLIP step in the engine's ``gsa``."""
    import torch

    inference = sys.modules[gsa.Model.__module__]
    original = inference.get_phrases_from_posmap

    def phrase(posmap, tokenized, tokenizer):
        separators = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.convert_tokens_to_ids(".")}
        tokens = whole_name_tokens(posmap.nonzero(as_tuple=True)[0].tolist(), tokenized["input_ids"], separators)
        return original(posmap, tokenized, tokenizer) if tokens is None else tokenizer.decode(tokens)
    inference.get_phrases_from_posmap = phrase

    def clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
        pil = gsa.Image.fromarray(image)
        crops = [clip_preprocess(pil.crop(crop_box(box, pil.width, pil.height))) for box in detections.xyxy]
        half = next(clip_model.parameters()).dtype == torch.float16
        feats = []

        def encode():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
                for k in range(0, len(crops), BATCH):
                    f = clip_model.encode_image(torch.stack(crops[k:k + BATCH]).to(device))
                    feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        gsa_fast.stage("clip", encode)
        return None, np.concatenate(feats, axis=0), None
    gsa.compute_clip_features = clip_features
