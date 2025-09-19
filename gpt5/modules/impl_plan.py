from __future__ import annotations

from typing import List

from gpt5.core.memory import MemoryStore


def implementation_plan_from_feature_map(memory: MemoryStore, feature_map_name: str, plan_name: str, author: str) -> str:
    rec = memory.load_spec(feature_map_name)
    sources: List[str] = []
    if rec:
        _, _, _, meta = rec
        if '"sources"' in meta:
            start = meta.find('[', meta.find('"sources"'))
            end = meta.find(']', start)
            if start != -1 and end != -1:
                for chunk in meta[start + 1 : end].split(','):
                    s = chunk.strip().strip('"')
                    if s:
                        sources.append(s)

    lines = [
        f"# {plan_name}",
        f"_Author: {author}_",
        "",
        "## Overview",
        f"Operationalize capabilities identified in '{feature_map_name}', exceeding reference modules.",
        "",
        "## Objectives",
        "- Robust crawl + extraction of README and docs",
        "- Consolidated feature map persisted to memory",
        "- Spec-driven planning + PRD + execution",
        "- Swarm coordination + wisdom capture",
        "- Advanced math engine (linear algebra, calculus, ODEs, optimization, symbolic)",
        "",
        "## Workstreams",
        "1) Crawl & Normalize — GitHub README preferred, HTML fallback",
        "2) Feature Map — merge sources, map to GPT5 modules",
        "3) Planning & PRD — spec-first, guardrails, stored in memory",
        "4) Execute — swarm simulation, persist tasks, record lessons",
        "5) Math Engine — high-accuracy numerical + symbolic stack",
        "",
        "## Risks & Mitigations",
        "- Network disruptions → cache + refreshable crawl",
        "- HTML noise → prefer README and parse markdown",
        "",
        "## Metrics",
        "- Feature map exists and up to date",
        "- Execution completed with persisted tasks",
        "- Math engine verified on sample problems",
    ]
    if sources:
        lines.extend(["", "## Sources", *[f"- {s}" for s in sources]])
    return "\n".join(lines).rstrip()

