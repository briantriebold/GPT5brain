from __future__ import annotations

from typing import Iterable, List

from gpt5.core.memory import MemoryStore


def build_feature_map(memory: MemoryStore, source_specs: Iterable[str], title: str) -> str:
    lines: List[str] = [f"# {title}", "", "This feature map consolidates extracted features from crawled sources.", ""]
    sources = list(dict.fromkeys(source_specs))
    lines.append("## Sources")
    for s in sources:
        rec = memory.load_spec(s) if hasattr(memory, "load_spec") else None
        url = ""
        if rec:
            _, _, _, meta = rec
            # cheap parse of url
            key = '"url"'
            idx = meta.find(key)
            if idx != -1:
                after = meta[idx + len(key):]
                q1 = after.find('"')
                q2 = after.find('"', q1 + 1)
                if q1 != -1 and q2 != -1:
                    url = after[q1 + 1 : q2]
        lines.append(f"- {s}{f' ({url})' if url else ''}")
    lines.append("")

    for s in sources:
        rec = memory.load_spec(s)
        if not rec:
            continue
        _, name, content, _ = rec
        lines.append(f"## Source: {name.replace('crawl:', '')}")
        lines.append(content)
        lines.append("")

    lines.extend([
        "## GPT5 Capability Mapping",
        "- Planning: gpt5/modules/planning.py",
        "- PRD generation: gpt5/modules/prd.py",
        "- Process mapping: gpt5/modules/process_map.py",
        "- Execution checklists: gpt5/modules/checklist.py",
        "- Evergreen memory (SQLite): gpt5/core/memory.py",
        "- Wisdom capture: gpt5/modules/wisdom.py",
        "- Swarm coordination: gpt5/modules/swarm.py + gpt5/core/orchestrator.py",
        "- Web fetching + cache: gpt5/tools/web_fetch.py + gpt5/modules/web.py",
        "- Crawling + extraction: gpt5/modules/crawl.py + gpt5/tools/*extract.py",
        "- Advanced Math Engine: gpt5/math/engine.py",
    ])
    return "\n".join(lines).rstrip()

