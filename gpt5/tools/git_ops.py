from __future__ import annotations

import subprocess
from typing import List


def _run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as e:  # noqa: PERF203
        return e.output.decode("utf-8", errors="ignore")


def git_init(path: str = ".") -> str:
    return _run(["git", "init", path])


def git_status(path: str = ".") -> str:
    return _run(["git", "-C", path, "status", "--porcelain=2", "-b"])


def git_add_all(path: str = ".") -> str:
    return _run(["git", "-C", path, "add", "-A"])  # noqa: S603


def git_commit(path: str = ".", message: str = "update") -> str:
    return _run(["git", "-C", path, "commit", "-m", message])


def git_checkout_new(path: str, branch: str) -> str:
    return _run(["git", "-C", path, "checkout", "-b", branch])


def git_checkout(path: str, branch: str) -> str:
    return _run(["git", "-C", path, "checkout", branch])


def git_current_branch(path: str = ".") -> str:
    return _run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"]).strip()


def git_push(path: str, remote: str, branch: str, set_upstream: bool = True) -> str:
    cmd = ["git", "-C", path, "push"]
    if set_upstream:
        cmd.extend(["-u", remote, branch])
    else:
        cmd.extend([remote, branch])
    return _run(cmd)
