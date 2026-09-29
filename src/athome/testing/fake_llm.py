"""OpenAI-compatible stand-in for the planner LLM (integration tests).

    python3 -m athome.testing.fake_llm --port 8000 --pick last

Planner requests: picks the first or last listed candidate, so its choices
are distinguishable from the minimum-cost baseline.
Command parsing requests: maps a few Korean nouns to target names.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEYWORDS = {"컵": "cup", "리모컨": "remote", "주전자": "kettle", "책": "book",
            "바나나": "banana", "접시": "plate"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pick", choices=["first", "last"], default="last")
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            props = body["response_format"]["json_schema"]["schema"]["properties"]
            if "targets" in props:
                text = body["messages"][-1]["content"]
                found = sorted((text.find(k), v) for k, v in KEYWORDS.items() if k in text)
                answer = {"targets": [v for _, v in found]}
            else:
                enum = props["selected_id"]["enum"]
                answer = {"selected_id": enum[0] if args.pick == "first" else enum[-1]}
            print(f"[{body['model']}] -> {answer}", flush=True)
            data = json.dumps({"choices": [{"message": {
                "content": json.dumps(answer, ensure_ascii=False)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/v1/models":
                self.send_error(404)
                return
            data = json.dumps({"object": "list", "data": [
                {"id": name, "object": "model"}
                for name in ("room", "search_location", "workspace")]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
