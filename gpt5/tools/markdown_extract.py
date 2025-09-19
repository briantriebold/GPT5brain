from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MDExtracted:
    title: str = ""
    headings: List[str] = field(default_factory=list)
    bullets: List[str] = field(default_factory=list)


def extract_from_markdown(md: str) -> MDExtracted:
    out = MDExtracted()
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("# ") and not out.title:
            out.title = line[2:].strip()
            out.headings.append(out.title)
            continue
        if line.startswith("## "):
            out.headings.append(line[3:].strip())
            continue
        if line.startswith("### "):
            out.headings.append(line[4:].strip())
            continue
        ls = line.lstrip()
        if ls.startswith("- ") or ls.startswith("* "):
            item = ls[2:].strip()
            if item:
                out.bullets.append(item)

    # Dedup
    def _uniq(seq: List[str]) -> List[str]:
        seen = set()
        res = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                res.append(s)
        return res

    out.headings = _uniq(out.headings)
    out.bullets = _uniq(out.bullets)
    return out

