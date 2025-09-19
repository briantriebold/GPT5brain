from __future__ import annotations

import json
import os
import urllib.request
from urllib.parse import urlparse


def _origin_to_repo(origin_url: str) -> tuple[str, str]:
    # Supports HTTPS and SSH shorthand
    if origin_url.startswith("git@github.com:"):
        path = origin_url.split(":", 1)[1]
    else:
        path = urlparse(origin_url).path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    owner, repo = path.split("/", 1)
    return owner, repo


def create_pull_request(origin_url: str, head: str, base: str, title: str, body: str = "") -> dict:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("Missing GITHUB_TOKEN/GH_TOKEN in environment")
    owner, repo = _origin_to_repo(origin_url)
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    payload = {"title": title, "head": head, "base": base, "body": body}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:  # nosec - controlled endpoint
        return json.loads(resp.read().decode("utf-8", errors="ignore"))

