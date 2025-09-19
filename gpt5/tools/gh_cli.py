from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def available() -> bool:
    return shutil.which("gh") is not None


def _run(cmd: list[str]) -> tuple[int, str, str]:  # noqa: ANN001
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True)
        return cp.returncode, cp.stdout, cp.stderr
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


def pr_create(base: str, head: str, title: str, body: str = "") -> dict:
    code, out, err = _run(["gh", "pr", "create", "-B", base, "-H", head, "-t", title, "-b", body])
    if code != 0:
        raise RuntimeError(f"gh pr create failed: {err or out}")
    return {"url": out.strip()}


def issue_create(title: str, body: str = "", assignees: Optional[str] = None, labels: Optional[str] = None) -> dict:
    cmd = ["gh", "issue", "create", "-t", title, "-b", body]
    if assignees:
        cmd += ["-a", assignees]
    if labels:
        cmd += ["-l", labels]
    code, out, err = _run(cmd)
    if code != 0:
        raise RuntimeError(f"gh issue create failed: {err or out}")
    return {"url": out.strip()}

