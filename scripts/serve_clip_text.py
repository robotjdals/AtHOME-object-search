"""CLIP text embedding server for target matching (runs on the GPU server).

    python scripts/serve_clip_text.py --model ViT-H-14 --pretrained laion2b_s32b_b79k

POST /embed_text {"model": "<name>", "texts": [...]}
  -> {"model": "<name>", "embeddings": [[...], ...]}   (L2-normalized)

Use the same model as perception's image features. The model name is
"<arch>/<pretrained>" and requests for another model are rejected.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import open_clip
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    name = f"{args.model}/{args.pretrained}"
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    @torch.no_grad()
    def embed(texts):
        features = model.encode_text(tokenizer(texts).to(args.device)).float()
        return torch.nn.functional.normalize(features, dim=-1).cpu().tolist()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/embed_text":
                return self._send(404, {"error": "unknown path"})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body.get("model") != name:
                    return self._send(400, {"error": f"model is {name}"})
                texts = body["texts"]
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    return self._send(400, {"error": "texts must be a list of strings"})
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(200, {"model": name, "embeddings": embed(texts)})

        def _send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"serving {name} on {args.host}:{args.port} ({args.device})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
