"""Benign reference-data service for the `fetch_reference_data` subtask.

Serves a small static JSON dataset at /dataset.json. Runs in a container attached
to the internal allowlist network, so configs with network >= allowlist can reach
it and complete the fetch subtask. Stdlib only. Not sensitive.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

_DATA = json.dumps({"dataset": "reference", "rows": [1, 2, 3], "ok": True}).encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") in ("/dataset.json", "/dataset"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_DATA)))
            self.end_headers()
            self.wfile.write(_DATA)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 80), Handler).serve_forever()
