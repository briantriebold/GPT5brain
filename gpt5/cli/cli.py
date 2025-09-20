from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpt5.core.memory import MemoryStore
from gpt5.core.tasks import Task
from gpt5.modules import checklist as checklist_module
from gpt5.modules import planning as planning_module
from gpt5.modules import prd as prd_module
from gpt5.modules import process_map as process_map_module
from gpt5.modules import crawl as crawl_module
from gpt5.modules import feature_map as fmap_module
from gpt5.modules import impl_plan as impl_module
from gpt5.modules import swarm as swarm_module
from gpt5.modules import web as web_module
from gpt5.modules import wisdom as wisdom_module
from gpt5.tools import reporting as reporting_tool
from gpt5.math import engine as math_engine
from gpt5.plugins.loader import discover_plugins, load_plugin, before_command_all, after_command_all
from gpt5.tools import git_ops
from gpt5.tools import github_api
from gpt5.modules.deficiency import DeficiencyDetector
from gpt5.tools import validators as validators
from gpt5.tools import report_html as report_html
from gpt5.tools.pathutil import sanitize_filename, ensure_dir
from gpt5 import config as gpt5_config
from gpt5.tools import gh_cli
from gpt5.tools import snapshot as snapshot_tool
from gpt5.tools import changelog as changelog_tool

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"


def _memory() -> MemoryStore:
    return MemoryStore(path=DATA_PATH)


def _write(path: Path, content: str, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.write_text(content, encoding="utf-8")


def _print(data: Any, as_json: bool) -> None:  # noqa: ANN401
    if as_json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        if isinstance(data, str):
            print(data)
        else:
            print(json.dumps(data, ensure_ascii=False, indent=2))


def cmd_init(args: argparse.Namespace) -> None:
    _ = _memory()  # Ensures DB and directories exist
    _print({"memory": str(DATA_PATH), "status": "initialized"}, args.json)


def cmd_plan(args: argparse.Namespace) -> None:
    plan_items = planning_module.generate_plan(args.objective, constraints=args.constraints)
    if args.json:
        data = [
            {
                "title": item.title,
                "description": item.description,
                "tasks": [
                    {"title": t.title, "description": t.description, "capabilities": t.capabilities}
                    for t in item.tasks
                ],
            }
            for item in plan_items
        ]
        _print(data, True)
        return
    for item in plan_items:
        print(f"## {item.title}")
        print(item.description)
        for task in item.tasks:
            print(f"- {task.title}: {task.description}")
        print()


def cmd_prd(args: argparse.Namespace) -> None:
    memory = _memory()
    prd = prd_module.create_prd(name=args.name, author=args.author or "unknown")
    markdown = prd.to_markdown()
    output_path = Path(args.output).resolve() if args.output else Path.cwd() / f"{args.name.replace(' ', '_')}.prd.md"
    output_path.write_text(markdown, encoding="utf-8")
    memory.save_spec(args.name, markdown, metadata={"author": prd.author})
    _print({"path": str(output_path), "name": args.name, "author": prd.author}, args.json)
    # Auto-export HTML if configured
    try:
        settings = gpt5_config.load_settings(Path.cwd())
        ae = settings.get("autoExport", {})
        if ae.get("enabled", False) and any(args.name.startswith(p) for p in ae.get("prefixes", [])):
            outdir = Path(ae.get("dir", "reports"))
            ensure_dir(outdir)
            html_text = report_html.render_html(markdown, title=args.name)
            fname = sanitize_filename(f"{args.name}.html")
            (outdir / fname).write_text(html_text, encoding="utf-8")
    except Exception:
        pass


def cmd_checklist(args: argparse.Namespace) -> None:
    checklist = checklist_module.create_execution_checklist(args.phase)
    if args.json:
        data = {
            "name": checklist.name,
            "items": [
                {"name": i.name, "description": i.description, "completed": i.completed}
                for i in checklist.items
            ],
        }
        _print(data, True)
        return
    print(f"Checklist: {checklist.name}")
    for item in checklist.items:
        status = "[ ]"
        print(f"{status} {item.name} - {item.description}")


def cmd_process(args: argparse.Namespace) -> None:
    items = planning_module.generate_plan(args.objective)
    flat: list[Task] = []
    for item in items:
        flat.extend(item.tasks)
    text = process_map_module.default_process_for_plan(flat)
    _print(text, args.json)


def cmd_swarm(args: argparse.Namespace) -> None:
    profiles = [
        swarm_module.AgentProfile(name="planner", description="Planning agent", capabilities=["analysis", "planning"]),
        swarm_module.AgentProfile(name="designer", description="Design agent", capabilities=["design", "process"]),
        swarm_module.AgentProfile(name="builder", description="Implementation agent", capabilities=["build"]),
        swarm_module.AgentProfile(name="qa", description="Quality agent", capabilities=["qa"]),
    ]

    def handler_factory(profile: swarm_module.AgentProfile):
        def handler(task: Task, agent, context):  # noqa: ANN001
            return f"{profile.name} handled task '{task.title}'"

        return handler

    orchestrator = swarm_module.spawn_swarm(
        swarm_module.SwarmConfig(name=args.name or "default", agent_profiles=profiles), handler_factory
    )
    tasks = [Task(title="example", description="Demonstrate swarm execution", capabilities=["analysis"])]
    completed = swarm_module.allocate_tasks(orchestrator, tasks)

    if args.json:
        data = [{"title": t.title, "status": t.status, "result": t.result} for t in completed]
        _print(data, True)
        return

    print(reporting_tool.format_task_report(completed))


def cmd_web(args: argparse.Namespace) -> None:
    memory = _memory() if args.cache else None
    fetcher = web_module.create_web_tool(memory=memory)
    validators.validate_url(args.url)
    content = fetcher.fetch(args.url, retries=2)
    if args.json:
        _print({"url": args.url, "content": content[: args.limit]}, True)
        return
    print(content[: args.limit])


def cmd_wisdom(args: argparse.Namespace) -> None:
    memory = _memory()
    lesson = wisdom_module.lesson_template(args.event, args.impact, args.mitigation)
    wisdom_module.record_lesson(memory, lesson)
    _print({"status": "recorded", "topic": args.event}, args.json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT5 orchestration CLI")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="Initialize local memory and folders")
    init.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    init.set_defaults(func=cmd_init)

    crawl = sub.add_parser("crawl", help="Crawl URLs and capture features into memory")
    crawl.add_argument("urls", nargs="+")
    crawl.add_argument("--max-bullets", type=int, default=200)
    crawl.add_argument("--prefer-readme", action="store_true", help="Prefer README.md extraction for GitHub repos")
    crawl.add_argument("--refresh", action="store_true", help="Bypass cache and refetch content")
    crawl.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    def cmd_crawl(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        validators.require_non_empty("urls", args.urls)
        results = crawl_module.crawl_and_capture(
            memory, args.urls, max_bullets=args.max_bullets, prefer_readme=args.prefer_readme, refresh=args.refresh
        )
        payload = [
            {"url": r.url, "spec": r.spec_name, "headings": r.headings_count, "bullets": r.bullets_count}
            for r in results
        ]
        _print(payload, args.json)
    crawl.set_defaults(func=cmd_crawl)

    featuremap = sub.add_parser("featuremap", help="Build a consolidated feature map from crawled specs")
    featuremap.add_argument("--name", default="Claude-Flow + Specify7 Feature Map")
    featuremap.add_argument("--sources", nargs="*", help="Explicit spec names (defaults to all crawl: specs)")
    featuremap.add_argument("--output", help="Output markdown path (defaults to ./FEATURE_MAP.md)")
    featuremap.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    def cmd_featuremap(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        sources = args.sources or memory.find_specs_by_prefix("crawl:")
        if not sources:
            _print({"status": "no-sources"}, args.json)
            return
        md = fmap_module.build_feature_map(memory, sources, title=args.name)
        out = Path(args.output).resolve() if args.output else (Path.cwd() / "FEATURE_MAP.md")
        out.write_text(md, encoding="utf-8")
        memory.save_spec(args.name, md, metadata={"type": "feature-map", "sources": sources})
        _print({"status": "saved", "path": str(out), "sources": sources}, args.json)
        try:
            settings = gpt5_config.load_settings(Path.cwd())
            ae = settings.get("autoExport", {})
            if ae.get("enabled", False) and any(args.name.startswith(p) for p in ae.get("prefixes", [])):
                outdir = Path(ae.get("dir", "reports"))
                ensure_dir(outdir)
                html_text = report_html.render_html(md, title=args.name)
                fname = sanitize_filename(f"{args.name}.html")
                (outdir / fname).write_text(html_text, encoding="utf-8")
        except Exception:
            pass
    featuremap.set_defaults(func=cmd_featuremap)

    implplan = sub.add_parser("impl-plan", help="Generate an Implementation Plan PRD from a feature map")
    implplan.add_argument("--from-featuremap", default="Claude-Flow + Specify7 Feature Map")
    implplan.add_argument("--name", default="GPT5 Implementation Plan")
    implplan.add_argument("--author", default="unknown")
    implplan.add_argument("--output")
    implplan.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    def cmd_implplan(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        md = impl_module.implementation_plan_from_feature_map(memory, args.from_featuremap, args.name, args.author)
        out = Path(args.output).resolve() if args.output else (Path.cwd() / "IMPLEMENTATION_PLAN.md")
        out.write_text(md, encoding="utf-8")
        memory.save_spec(args.name, md, metadata={"type": "impl-plan", "from": args.from_featuremap})
        _print({"status": "saved", "path": str(out)}, args.json)
        try:
            settings = gpt5_config.load_settings(Path.cwd())
            ae = settings.get("autoExport", {})
            if ae.get("enabled", False) and any(args.name.startswith(p) for p in ae.get("prefixes", [])):
                outdir = Path(ae.get("dir", "reports"))
                ensure_dir(outdir)
                html_text = report_html.render_html(md, title=args.name)
                fname = sanitize_filename(f"{args.name}.html")
                (outdir / fname).write_text(html_text, encoding="utf-8")
        except Exception:
            pass
    implplan.set_defaults(func=cmd_implplan)

    execute = sub.add_parser("execute", help="Execute a plan for an objective (simulated swarm)")
    execute.add_argument("--objective", required=True)
    execute.add_argument("--constraints", nargs="*", default=None)
    execute.add_argument("--strategy", choices=["pipeline", "broadcast", "roundrobin"], default="pipeline")
    execute.add_argument("--report", help="Optional path to save a rich Markdown report")
    execute.add_argument("--html", help="Optional path to save a standalone HTML report")
    execute.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    def cmd_execute(args: argparse.Namespace) -> None:  # noqa: ANN001
        from dataclasses import replace
        memory = _memory()
        items = planning_module.generate_plan(args.objective, constraints=args.constraints)
        flat: list[Task] = []
        for it in items:
            flat.extend(it.tasks)
        profiles = [
            swarm_module.AgentProfile(name="planner", description="Planning agent", capabilities=["analysis", "planning"]),
            swarm_module.AgentProfile(name="designer", description="Design agent", capabilities=["design", "process"]),
            swarm_module.AgentProfile(name="builder", description="Implementation agent", capabilities=["build"]),
            swarm_module.AgentProfile(name="qa", description="Quality agent", capabilities=["qa"]),
        ]
        def handler_factory(profile: swarm_module.AgentProfile):
            def handler(task: Task, agent, context):  # noqa: ANN001
                return f"{profile.name} handled task '{task.title}'"
            return handler
        orch = swarm_module.spawn_swarm(swarm_module.SwarmConfig(name="exec", agent_profiles=profiles), handler_factory)
        agent_counts = {p.name: 0 for p in profiles}
        completed: list[Task] = []
        if args.strategy == "pipeline":
            # Default orchestrated run
            for t in flat:
                # best-effort agent selection mirror
                sel = None
                for a in orch.agents.values():
                    if not t.capabilities or any(c in a.profile.capabilities for c in t.capabilities):
                        sel = a
                        break
                if sel:
                    t.assigned_agent = sel.name
                    agent_counts[sel.name] += 1
                completed.append(orch.run_task(t))
        elif args.strategy == "broadcast":
            # Parallel fan-out to capable agents
            import concurrent.futures as _f
            tasks_to_run = []
            for t in flat:
                for a in orch.agents.values():
                    if not t.capabilities or any(c in a.profile.capabilities for c in t.capabilities):
                        tt = replace(t, assigned_agent=a.name)
                        tasks_to_run.append(tt)
                        agent_counts[a.name] += 1
            with _f.ThreadPoolExecutor(max_workers=max(4, len(orch.agents))) as ex:
                futs = [ex.submit(orch.run_task, tt) for tt in tasks_to_run]
                for fut in futs:
                    completed.append(fut.result())
        else:  # roundrobin
            names = list(orch.agents.keys())
            idx = 0
            for t in flat:
                a = orch.agents[names[idx % len(names)]]
                tt = replace(t, assigned_agent=a.name)
                agent_counts[a.name] += 1
                completed.append(orch.run_task(tt))

        from gpt5.tools.reporting import format_task_report
        report_lines = [
            f"# Execution Report — {args.objective}",
            "",
            f"Strategy: {args.strategy}",
            "",
            "## Agent Allocation",
        ]
        for name, cnt in agent_counts.items():
            report_lines.append(f"- {name}: {cnt} tasks")
        report_lines.append("")
        report_lines.append("## Tasks")
        report_lines.append(format_task_report(completed))
        # Add mermaid diagram
        mer = ["```mermaid", "flowchart TD"]
        last = None
        for i, t in enumerate(completed):
            nid = f"T{i}"
            mer.append(f"    {nid}([" + t.title.replace('"','\"') + "]):::task")
            if last is not None:
                mer.append(f"    {last} --> {nid}")
            last = nid
        mer.append("    classDef task fill:#fff8e1,stroke:#ff6f00,stroke-width:1px;")
        mer.append("```")
        report_lines.append("")
        report_lines.append("## Diagram")
        report_lines.extend(mer)
        # Sequence diagram (orchestrator -> agent)
        seq = ["```mermaid", "sequenceDiagram", "    participant O as orchestrator"]
        for t in completed:
            agent = t.assigned_agent or "agent"
            seq.append(f"    O->>+{agent}: {t.title}")
            seq.append(f"    {agent}-->>-O: done ({(t.duration_ms or 0)} ms)")
        seq.append("```")
        report_lines.append("")
        report_lines.append("## Agent Sequence")
        report_lines.extend(seq)
        # Gantt timeline (sequential synthetic timing)
        gantt = ["```mermaid", "gantt", "    dateFormat  X", "    title Execution Timeline", "    section Tasks"]
        for i, _t in enumerate(completed):
            start = i
            dur = 1
            gantt.append(f"    T{i} : {start}, {dur}")
        gantt.append("```")
        report_lines.append("")
        report_lines.append("## Timeline")
        report_lines.extend(gantt)
        report = "\n".join(report_lines)
        if args.report:
            Path(args.report).resolve().write_text(report, encoding="utf-8")
        memory.save_spec(f"execution:{args.objective}", report, metadata={"objective": args.objective, "strategy": args.strategy})
        _print({"status": "executed", "tasks": len(completed), "strategy": args.strategy}, args.json)
        if args.html:
            html_text = report_html.render_html(report, title=f"execution:{args.objective}")
            Path(args.html).resolve().write_text(html_text, encoding="utf-8")
        try:
            settings = gpt5_config.load_settings(Path.cwd())
            ae = settings.get("autoExport", {})
            name = f"execution:{args.objective}"
            if ae.get("enabled", False) and any(name.startswith(p) for p in ae.get("prefixes", [])):
                outdir = Path(ae.get("dir", "reports"))
                ensure_dir(outdir)
                html_text = report_html.render_html(report, title=name)
                fname = sanitize_filename(f"{name}.html")
                (outdir / fname).write_text(html_text, encoding="utf-8")
        except Exception:
            pass
    execute.set_defaults(func=cmd_execute)

    math = sub.add_parser("math", help="Advanced mathematics processing")
    math_sub = math.add_subparsers(dest="math_cmd")
    me = math_sub.add_parser("expr", help="Evaluate expression with variables")
    me.add_argument("expr")
    me.add_argument("--vars", help="JSON of variables, e.g. {\"x\":2}")
    me.add_argument("--var", action="append", help="key=value (repeatable)")
    me.add_argument("--json", action="store_true")
    def cmd_math_expr(args: argparse.Namespace) -> None:  # noqa: ANN001
        import json as _json
        vars_map = validators.ensure_json_or_hint("--vars", args.vars) if args.vars else {}
        if args.var:
            for kv in args.var:
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                try:
                    vars_map[k] = float(v)
                except ValueError:
                    pass
        val = math_engine.evaluate_expression(args.expr, vars_map)
        _print({"expr": args.expr, "value": val}, args.json)
    me.set_defaults(func=cmd_math_expr)

    mo = math_sub.add_parser("ode", help="Solve dy/dt = f(t,y) with initial value")
    mo.add_argument("rhs", help="Right-hand side f(t,y), e.g., -y")
    mo.add_argument("t0", type=float)
    mo.add_argument("t1", type=float)
    mo.add_argument("y0", type=float)
    mo.add_argument("--samples", type=int, default=100)
    mo.add_argument("--csv", help="Optional path to export CSV of t,y")
    mo.add_argument("--plot", help="Optional path to save a PNG/SVG plot")
    mo.add_argument("--json", action="store_true")
    def cmd_math_ode(args: argparse.Namespace) -> None:  # noqa: ANN001
        import numpy as np
        t, y = math_engine.solve_ode(args.rhs, (args.t0, args.t1), [args.y0])
        if t.size > args.samples:
            idx = np.linspace(0, t.size - 1, args.samples).astype(int)
            t_s = t[idx]
            y_s = y[0, idx]
        else:
            t_s = t
            y_s = y[0]
        if args.csv:
            import csv
            outp = Path(args.csv).resolve()
            with outp.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "y"])
                for ti, yi in zip(t_s, y_s):
                    w.writerow([float(ti), float(yi)])
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(t_s, y_s, label=f"dy/dt={args.rhs}")
            ax.set_xlabel("t")
            ax.set_ylabel("y")
            ax.legend()
            outp = Path(args.plot).resolve()
            fig.tight_layout()
            fig.savefig(outp)
            plt.close(fig)
        _print({"points": int(min(args.samples, t.size)), "t0": float(t_s[0]), "t1": float(t_s[-1]), "y0": float(y_s[0]), "y1": float(y_s[-1])}, args.json)
    mo.set_defaults(func=cmd_math_ode)

    ms = math_sub.add_parser("solve", help="Solve linear system Ax=b")
    ms.add_argument("--A", required=True, help="JSON matrix")
    ms.add_argument("--b", required=True, help="JSON vector")
    ms.add_argument("--json", action="store_true")
    def cmd_math_solve(args: argparse.Namespace) -> None:  # noqa: ANN001
        import json as _json
        A = _json.loads(args.A)
        b = _json.loads(args.b)
        x = math_engine.solve_linear(A, b)
        _print({"x": x.tolist()}, args.json)
    ms.set_defaults(func=cmd_math_solve)

    mi = math_sub.add_parser("integrate", help="Definite integral")
    mi.add_argument("expr")
    mi.add_argument("var")
    mi.add_argument("a", type=float)
    mi.add_argument("b", type=float)
    mi.add_argument("--json", action="store_true")
    def cmd_math_integrate(args: argparse.Namespace) -> None:  # noqa: ANN001
        val = math_engine.integrate_definite(args.expr, args.var, args.a, args.b)
        _print({"value": val}, args.json)
    mi.set_defaults(func=cmd_math_integrate)

    mo = math_sub.add_parser("optimize", help="Minimize expression f(vars)")
    mo.add_argument("expr")
    mo.add_argument("--vars", required=True, help="Comma-separated variable names")
    mo.add_argument("--x0", required=True, help="JSON array initial guess")
    mo.add_argument("--json", action="store_true")
    def cmd_math_opt(args: argparse.Namespace) -> None:  # noqa: ANN001
        import json as _json
        vars_order = [v.strip() for v in args.vars.split(",") if v.strip()]
        x0 = _json.loads(args.x0)
        x, fval = math_engine.minimize_expression(args.expr, vars_order, x0)
        _print({"x": x.tolist(), "f": fval}, args.json)
    mo.set_defaults(func=cmd_math_opt)

    mf = math_sub.add_parser("fft", help="Compute real FFT magnitude spectrum")
    mf.add_argument("--data", required=True, help="JSON array of samples, e.g., [0,1,0,-1,...]")
    mf.add_argument("--rate", type=float, default=1.0, help="Sample rate (Hz)")
    mf.add_argument("--top", type=int, default=5, help="Top-K peaks to report")
    mf.add_argument("--json", action="store_true")
    def cmd_math_fft(args: argparse.Namespace) -> None:  # noqa: ANN001
        import json as _json
        data = _json.loads(args.data)
        xf, mag = math_engine.fft_real(data, sample_rate=args.rate)
        # extract top-K (exclude DC unless it's significant)
        idx = mag.argsort()[::-1][: max(1, args.top)]
        peaks = [{"freq": float(xf[i]), "mag": float(mag[i])} for i in idx]
        _print({"peaks": peaks, "n": len(data)}, args.json)
    mf.set_defaults(func=cmd_math_fft)

    mp = math_sub.add_parser("polyfit", help="Polynomial least squares fit")
    mp.add_argument("--x", required=True, help="JSON array of x values")
    mp.add_argument("--y", required=True, help="JSON array of y values")
    mp.add_argument("--deg", type=int, required=True)
    mp.add_argument("--json", action="store_true")
    def cmd_math_polyfit(args: argparse.Namespace) -> None:  # noqa: ANN001
        import json as _json
        x = _json.loads(args.x)
        y = _json.loads(args.y)
        coeffs = math_engine.polyfit_fit(x, y, args.deg)
        _print({"coeffs": [float(c) for c in coeffs]}, args.json)
    mp.set_defaults(func=cmd_math_polyfit)

    mr = math_sub.add_parser("root", help="Solve f(var)=0 for root")
    mr.add_argument("expr")
    mr.add_argument("var")
    mr.add_argument("--x0", type=float)
    mr.add_argument("--a", type=float)
    mr.add_argument("--b", type=float)
    mr.add_argument("--json", action="store_true")
    def cmd_math_root(args: argparse.Namespace) -> None:  # noqa: ANN001
        root = math_engine.solve_root(args.expr, args.var, x0=args.x0, a=args.a, b=args.b)
        _print({"root": float(root)}, args.json)
    mr.set_defaults(func=cmd_math_root)

    mpl = math_sub.add_parser("plot", help="Plot y=f(x) over a range; saves PNG/SVG")
    mpl.add_argument("expr")
    mpl.add_argument("var")
    mpl.add_argument("a", type=float)
    mpl.add_argument("b", type=float)
    mpl.add_argument("--samples", type=int, default=200)
    mpl.add_argument("--out", required=True, help="Output image path (.png/.svg)")
    mpl.add_argument("--json", action="store_true")
    def cmd_math_plot(args: argparse.Namespace) -> None:  # noqa: ANN001
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr as _parse
        x = sp.Symbol(args.var)
        e = _parse(args.expr, evaluate=True)
        f = sp.lambdify(x, e, modules=["numpy"])
        xs = np.linspace(args.a, args.b, args.samples)
        ys = f(xs)
        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        ax.set_xlabel(args.var)
        ax.set_ylabel("f(x)")
        fig.tight_layout()
        outp = Path(args.out).resolve()
        fig.savefig(outp)
        plt.close(fig)
        _print({"saved": str(outp), "points": int(args.samples)}, args.json)
    mpl.set_defaults(func=cmd_math_plot)


    plan = sub.add_parser("plan", help="Generate execution plan")
    plan.add_argument("objective")
    plan.add_argument("--constraints", nargs="*", default=None)
    plan.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    plan.set_defaults(func=cmd_plan)

    prd = sub.add_parser("prd", help="Create PRD")
    prd.add_argument("name")
    prd.add_argument("--author")
    prd.add_argument("--output")
    prd.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    prd.set_defaults(func=cmd_prd)

    checklist = sub.add_parser("checklist", help="Generate checklist for phase")
    checklist.add_argument("phase")
    checklist.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    checklist.set_defaults(func=cmd_checklist)

    process = sub.add_parser("process", help="Render process map for objective")
    process.add_argument("objective")
    process.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    process.set_defaults(func=cmd_process)

    swarm = sub.add_parser("swarm", help="Spawn demo swarm")
    swarm.add_argument("--name")
    swarm.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    swarm.set_defaults(func=cmd_swarm)

    web = sub.add_parser("web", help="Fetch web content")
    web.add_argument("url")
    web.add_argument("--limit", type=int, default=2000)
    web.add_argument("--cache", action="store_true")
    web.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    web.set_defaults(func=cmd_web)

    wisdom = sub.add_parser("wisdom", help="Store lesson learned")
    wisdom.add_argument("event")
    wisdom.add_argument("impact")
    wisdom.add_argument("mitigation")
    wisdom.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    wisdom.set_defaults(func=cmd_wisdom)

    # Plugins
    plugins = sub.add_parser("plugins", help="Manage plugins")
    plugins.add_argument("action", choices=["list", "load"])
    plugins.add_argument("--dir", default=str((Path(__file__).resolve().parent.parent / "plugins")))
    plugins.add_argument("--name", help="Plugin filename to load")
    plugins.add_argument("--json", action="store_true")
    def cmd_plugins(args: argparse.Namespace) -> None:  # noqa: ANN001
        pdir = Path(args.dir)
        if args.action == "list":
            paths = [str(p.name) for p in discover_plugins(pdir)]
            _print({"plugins": paths}, args.json)
        elif args.action == "load":
            if not args.name:
                _print({"error": "--name required"}, args.json)
                return
            mod = load_plugin(pdir / args.name)
            _print({"loaded": getattr(mod, "__name__", args.name)}, args.json)
    plugins.set_defaults(func=cmd_plugins)

    # Memory search
    mem = sub.add_parser("memory", help="Memory utilities")
    mem_sub = mem.add_subparsers(dest="mem_cmd")
    msrch = mem_sub.add_parser("search", help="Search indexed text by tokens")
    msrch.add_argument("query")
    msrch.add_argument("--top", type=int, default=5)
    msrch.add_argument("--json", action="store_true")
    def cmd_mem_search(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        _print({"results": memory.search_text(args.query, top_k=args.top)}, args.json)
    msrch.set_defaults(func=cmd_mem_search)

    mindx = mem_sub.add_parser("reindex", help="Rebuild token index from all specs")
    mindx.add_argument("--json", action="store_true")
    def cmd_mem_reindex(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        n = memory.index_all_specs()
        _print({"indexed": n}, args.json)
    mindx.set_defaults(func=cmd_mem_reindex)

    mexp = mem_sub.add_parser("export", help="Export all specs to a directory")
    mexp.add_argument("--dir", default="exported-specs")
    mexp.add_argument("--json", action="store_true")
    def cmd_mem_export(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        outdir = Path(args.dir).resolve()
        ensure_dir(outdir)
        rows = memory.list_specs()
        count = 0
        for _id, name, _created in rows:  # noqa: N806
            rec = memory.load_spec(name)
            if not rec:
                continue
            _, spec_name, content, _meta = rec
            fname = sanitize_filename(f"{spec_name}.md")
            (outdir / fname).write_text(content, encoding="utf-8")
            count += 1
        _print({"exported": count, "dir": str(outdir)}, args.json)
    mexp.set_defaults(func=cmd_mem_export)

    mweb = mem_sub.add_parser("search-web", help="Search only web: cached content")
    mweb.add_argument("query")
    mweb.add_argument("--top", type=int, default=5)
    mweb.add_argument("--json", action="store_true")
    def cmd_mem_search_web(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        all_hits = memory.search_text(args.query, top_k=50)
        filt = [(n, s) for (n, s) in all_hits if str(n).startswith("web:")][: args.top]
        _print({"results": filt}, args.json)
    mweb.set_defaults(func=cmd_mem_search_web)

    # Git ops
    git = sub.add_parser("git", help="Basic git operations")
    git.add_argument("action", choices=["init", "status", "add", "commit", "branch", "checkout", "current", "push", "remote-add", "remote-list"])
    git.add_argument("--path", default=str(Path.cwd()))
    git.add_argument("--message", default="update")
    git.add_argument("--name", help="Branch name")
    git.add_argument("--remote", default="origin")
    git.add_argument("--url", help="Remote URL for remote-add")
    git.add_argument("--json", action="store_true")
    def cmd_git(args: argparse.Namespace) -> None:  # noqa: ANN001
        if args.action == "init":
            out = git_ops.git_init(args.path)
        elif args.action == "status":
            out = git_ops.git_status(args.path)
        elif args.action == "add":
            out = git_ops.git_add_all(args.path)
        elif args.action == "commit":
            out = git_ops.git_commit(args.path, message=args.message)
        elif args.action == "branch":
            if not args.name:
                _print({"error": "--name required"}, args.json)
                return
            out = git_ops.git_checkout_new(args.path, args.name)
        elif args.action == "checkout":
            if not args.name:
                _print({"error": "--name required"}, args.json)
                return
            out = git_ops.git_checkout(args.path, args.name)
        elif args.action == "current":
            out = git_ops.git_current_branch(args.path)
        elif args.action == "push":
            if not args.name:
                # If no branch provided, use current
                args.name = git_ops.git_current_branch(args.path)
            out = git_ops.git_push(args.path, args.remote, args.name)
        elif args.action == "remote-add":
            if not args.url or not args.remote:
                _print({"error": "--url and --remote required"}, args.json)
                return
            out = git_ops.git_remote_add(args.path, args.remote, args.url)
        elif args.action == "remote-list":
            out = git_ops.git_remote_list(args.path)
        else:
            out = ""
        _print({"out": out}, args.json)
    git.set_defaults(func=cmd_git)

    # Report export
    report = sub.add_parser("report", help="Export reports")
    rep_sub = report.add_subparsers(dest="rep_cmd")
    rhtml = rep_sub.add_parser("html", help="Export markdown/spec with Mermaid to a standalone HTML file")
    rhtml.add_argument("--spec", help="Spec name stored in memory (e.g., execution:..., mission:..., deficiency:dashboard)")
    rhtml.add_argument("--spec-latest", action="store_true", help="Export latest saved spec (optionally filter by --prefix)")
    rhtml.add_argument("--prefix", help="Prefix to match when using --spec-latest")
    rhtml.add_argument("--input", help="Path to a markdown file to export")
    rhtml.add_argument("--title", help="HTML title")
    rhtml.add_argument("--output", required=True, help="Output .html path")
    rhtml.add_argument("--json", action="store_true")
    def cmd_report_html(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        content = None
        title = args.title or "Report"
        if args.spec:
            rec = memory.load_spec(args.spec)
            if not rec:
                _print({"error": "spec-not-found", "spec": args.spec}, args.json)
                return
            _, name, md, _ = rec
            content = md
            title = args.title or name
        elif args.spec_latest:
            rows = memory.list_specs()
            md_name = None
            if args.prefix:
                # find first matching by prefix
                for (_id, name, _created) in rows:  # noqa: N806
                    if str(name).startswith(args.prefix):
                        md_name = name
                        break
            else:
                md_name = rows[0][1] if rows else None
            if not md_name:
                _print({"error": "no-specs", "hint": "no matching specs or memory empty"}, args.json)
                return
            rec = memory.load_spec(md_name)
            _, name, md, _ = rec
            content = md
            title = args.title or name
        if not content and args.input:
            p = Path(args.input).resolve()
            if not p.exists():
                _print({"error": "input-not-found", "path": str(p)}, args.json)
                return
            content = p.read_text(encoding="utf-8")
        if not content:
            _print({"error": "no-content", "hint": "provide --spec or --input"}, args.json)
            return
        html_text = report_html.render_html(content, title=title)
        outp = Path(args.output).resolve()
        outp.write_text(html_text, encoding="utf-8")
        _print({"status": "saved", "path": str(outp)}, args.json)
    rhtml.set_defaults(func=cmd_report_html)

    rchg = rep_sub.add_parser("changelog", help="Generate CHANGELOG.md from git history")
    rchg.add_argument("--output", default="CHANGELOG.md")
    rchg.add_argument("--json", action="store_true")
    def cmd_report_changelog(args: argparse.Namespace) -> None:  # noqa: ANN001
        outp = Path(args.output).resolve()
        md = changelog_tool.generate_changelog(outp)
        _print({"status": "saved", "path": str(outp), "length": len(md)}, args.json)
    rchg.set_defaults(func=cmd_report_changelog)

    rindex = rep_sub.add_parser("index", help="Build reports/index.html")
    rindex.add_argument("--dir", default="reports")
    rindex.add_argument("--json", action="store_true")
    def cmd_report_index(args: argparse.Namespace) -> None:  # noqa: ANN001
        outdir = Path(args.dir).resolve()
        ensure_dir(outdir)
        html = snapshot_tool.build_index(outdir)
        (outdir / "index.html").write_text(html, encoding="utf-8")
        _print({"status": "saved", "path": str(outdir / "index.html")}, args.json)
    rindex.set_defaults(func=cmd_report_index)

    # Pull request automation
    pr = sub.add_parser("pr", help="Pull request automation")
    pr.add_argument("action", choices=["create"]) 
    pr.add_argument("--base", default="main")
    pr.add_argument("--head", help="Head branch", default=None)
    pr.add_argument("--title", required=True)
    pr.add_argument("--body", default="")
    pr.add_argument("--use-gh", action="store_true", help="Prefer GitHub CLI (gh) if available")
    pr.add_argument("--json", action="store_true")
    def cmd_pr(args: argparse.Namespace) -> None:  # noqa: ANN001
        # Determine origin URL
        origin = git_ops.git_remote_list(str(Path.cwd()))
        origin_url = None
        for line in origin.splitlines():
            if "(push)" in line and "origin\t" in line:
                origin_url = line.split("\t", 1)[1].split(" ")[0]
                break
        if not origin_url:
            _print({"error": "no-origin-remote"}, args.json)
            return
        head = args.head or git_ops.git_current_branch(str(Path.cwd()))
        try:
            if args.use_gh and gh_cli.available():
                r = gh_cli.pr_create(args.base, head, args.title, args.body)
                _print({"status": "created", "url": r.get("url")}, args.json)
            else:
                r = github_api.create_pull_request(origin_url, head=head, base=args.base, title=args.title, body=args.body)
                _print({"status": "created", "url": r.get("html_url"), "number": r.get("number")}, args.json)
        except Exception as exc:  # noqa: BLE001
            _print({"error": str(exc)}, args.json)
    pr.set_defaults(func=cmd_pr)

    # Optimize-now (chain mission -> regressions -> dashboard -> export latest)
    opt = sub.add_parser("optimize", help="Self-optimization utilities")
    opt_sub = opt.add_subparsers(dest="opt_cmd")
    onow = opt_sub.add_parser("now")
    onow.add_argument("--objective", default="Self-Optimization Now")
    onow.add_argument("--strategy", choices=["pipeline", "broadcast", "roundrobin"], default="pipeline")
    onow.add_argument("--json", action="store_true")
    def cmd_optimize_now(args: argparse.Namespace) -> None:  # noqa: ANN001
        # 1) Mission run
        mem = _memory()
        mid = mem.mission_create(args.objective, status="planning")
        items = planning_module.generate_plan(args.objective)
        flat: list[Task] = []
        for it in items:
            flat.extend(it.tasks)
        profiles = [
            swarm_module.AgentProfile(name="planner", description="Planning agent", capabilities=["analysis", "planning"]),
            swarm_module.AgentProfile(name="designer", description="Design agent", capabilities=["design", "process"]),
            swarm_module.AgentProfile(name="builder", description="Implementation agent", capabilities=["build"]),
            swarm_module.AgentProfile(name="qa", description="Quality agent", capabilities=["qa"]),
        ]
        def handler_factory(profile: swarm_module.AgentProfile):
            def handler(task: Task, agent, context):  # noqa: ANN001
                return f"{profile.name} handled task '{task.title}'"
            return handler
        orch = swarm_module.spawn_swarm(swarm_module.SwarmConfig(name="optimize", agent_profiles=profiles), handler_factory)
        from dataclasses import replace
        completed: list[Task] = []
        if args.strategy == "pipeline":
            for t in flat:
                completed.append(orch.run_task(t))
        elif args.strategy == "broadcast":
            for t in flat:
                for _a in orch.agents.values():
                    completed.append(orch.run_task(replace(t, assigned_agent=_a.name)))
        else:
            names = list(orch.agents.keys())
            for i, t in enumerate(flat):
                a = orch.agents[names[i % len(names)]]
                completed.append(orch.run_task(replace(t, assigned_agent=a.name)))
        from gpt5.tools.reporting import format_task_report
        report_lines = [
            f"# Optimize Report — {args.objective}",
            "",
            f"Strategy: {args.strategy}",
            "",
            "## Tasks",
            format_task_report(completed),
        ]
        # simple mermaid
        mer = ["```mermaid", "flowchart TD"]
        last = None
        for i, t in enumerate(completed):
            nid = f"T{i}"
            mer.append(f"    {nid}([" + t.title.replace('"','\"') + "]):::task")
            if last is not None:
                mer.append(f"    {last} --> {nid}")
            last = nid
        mer.append("    classDef task fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px;")
        mer.append("```")
        report_lines.append("")
        report_lines.append("## Diagram")
        report_lines.extend(mer)
        # Sequence diagram
        seq = ["```mermaid", "sequenceDiagram", "    participant O as orchestrator"]
        for t in completed:
            agent = t.assigned_agent or "agent"
            seq.append(f"    O->>+{agent}: {t.title}")
            seq.append(f"    {agent}-->>-O: done ({(t.duration_ms or 0)} ms)")
        seq.append("```")
        report_lines.append("")
        report_lines.append("## Agent Sequence")
        report_lines.extend(seq)
        report = "\n".join(report_lines)
        mem.save_spec(f"mission:{mid}:execution", report, metadata={"objective": args.objective, "strategy": args.strategy})
        mem.mission_update(mid, status="complete", progress=1.0, report=report)

        # 2) Run regressions
        import subprocess, sys as _sys, json as _json
        rows = mem.list_regressions(None)
        reg_results = []
        for r in rows:
            rid, sig, argv_json = r[0], r[1], r[2]
            argv = _json.loads(argv_json)
            cp = subprocess.run([_sys.executable, "-m", "gpt5", *argv], capture_output=True, text=True)
            ok = cp.returncode == 0
            mem.update_regression_result(rid, ok, (cp.stdout or "")[-400:] + (cp.stderr or "")[-400:])
            reg_results.append({"id": rid, "signature": sig, "pass": ok})

        # 3) Dashboard refresh
        defs = mem.list_deficiencies()
        by_status = {"open": 0, "proposed": 0, "mitigated": 0}
        top = sorted(({"signature": d[1], "count": d[3], "status": d[4]} for d in defs), key=lambda x: x["count"], reverse=True)
        for d in defs:
            st = d[4] or "open"
            if st not in by_status:
                by_status[st] = 0
            by_status[st] += 1
        total_reg = len(rows)
        passed = sum(1 for _r in mem.list_regressions(None) if _r[3] == "passed")
        failed = sum(1 for _r in mem.list_regressions(None) if _r[3] == "failed")
        md = ["# Deficiency Dashboard", "", "## Status Summary", f"- Open: {by_status.get('open',0)}", f"- Proposed: {by_status.get('proposed',0)}", f"- Mitigated: {by_status.get('mitigated',0)}", "", "## Top Signatures"]
        for item in top[:10]:
            md.append(f"- {item['signature']} — count: {item['count']} — status: {item['status']}")
        md.extend(["", "## Regression Summary", f"- Total regressions: {total_reg}", f"- Passed: {passed}", f"- Failed: {failed}"])
        dash = "\n".join(md)
        mem.save_spec("deficiency:dashboard", dash, metadata={"open": by_status.get('open',0), "proposed": by_status.get('proposed',0), "mitigated": by_status.get('mitigated',0), "regressions": total_reg, "passed": passed, "failed": failed})

        # 4) Export latest mission and dashboard to HTML
        settings = gpt5_config.load_settings(Path.cwd())
        outdir = Path(settings.get("autoExport", {}).get("dir", "reports"))
        ensure_dir(outdir)
        # mission html
        from gpt5.tools import report_html as _rhtml
        (outdir / sanitize_filename(f"mission:{mid}:execution.html")).write_text(_rhtml.render_html(report, title=f"mission:{mid}:execution"), encoding="utf-8")
        (outdir / sanitize_filename("deficiency:dashboard.html")).write_text(_rhtml.render_html(dash, title="deficiency:dashboard"), encoding="utf-8")
        _print({"mission": mid, "regressions": reg_results, "dashboard": {"open": by_status.get('open',0), "proposed": by_status.get('proposed',0), "mitigated": by_status.get('mitigated',0)}}, args.json)
    onow.set_defaults(func=cmd_optimize_now)

    osched = opt_sub.add_parser("schedule")
    osched.add_argument("--run", action="store_true", help="Attempt to register Windows scheduled task now")
    osched.add_argument("--json", action="store_true")
    def cmd_optimize_schedule(args: argparse.Namespace) -> None:  # noqa: ANN001
        script = Path(__file__).resolve().parents[2] / "scripts" / "schedule_optimize.ps1"
        if not script.exists():
            _print({"error": "script-not-found", "path": str(script)}, args.json)
            return
        if args.run:
            import subprocess
            cp = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)], capture_output=True, text=True)
            _print({"status": "ran", "stdout": cp.stdout, "stderr": cp.stderr}, args.json)
        else:
            _print({"status": "ready", "path": str(script)}, args.json)
    osched.set_defaults(func=cmd_optimize_schedule)

    # Mission autopilot
    mission = sub.add_parser("mission", help="Autopilot planâ†’executeâ†’learn")
    mission_sub = mission.add_subparsers(dest="mission_cmd")
    mstart = mission_sub.add_parser("start")
    mstart.add_argument("objective")
    mstart.add_argument("--constraints", nargs="*", default=None)
    mstart.add_argument("--strategy", choices=["pipeline", "broadcast", "roundrobin"], default="pipeline")
    mstart.add_argument("--report", help="Optional path to save mission report")
    mstart.add_argument("--html", help="Optional path to save a standalone HTML mission report")
    mstart.add_argument("--json", action="store_true")
    def cmd_mission_start(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        mid = memory.mission_create(args.objective, status="planning")
        items = planning_module.generate_plan(args.objective, constraints=args.constraints)
        flat: list[Task] = []
        for it in items:
            flat.extend(it.tasks)
        profiles = [
            swarm_module.AgentProfile(name="planner", description="Planning agent", capabilities=["analysis", "planning"]),
            swarm_module.AgentProfile(name="designer", description="Design agent", capabilities=["design", "process"]),
            swarm_module.AgentProfile(name="builder", description="Implementation agent", capabilities=["build"]),
            swarm_module.AgentProfile(name="qa", description="Quality agent", capabilities=["qa"]),
        ]
        def handler_factory(profile: swarm_module.AgentProfile):
            def handler(task: Task, agent, context):  # noqa: ANN001
                return f"{profile.name} handled task '{task.title}'"
            return handler
        orch = swarm_module.spawn_swarm(swarm_module.SwarmConfig(name="mission", agent_profiles=profiles), handler_factory)
        memory.mission_update(mid, status="executing", progress=0.3)
        # Strategy execution mirroring 'execute'
        from dataclasses import replace
        agent_counts = {p.name: 0 for p in profiles}
        completed: list[Task] = []
        if args.strategy == "pipeline":
            for t in flat:
                sel = None
                for a in orch.agents.values():
                    if not t.capabilities or any(c in a.profile.capabilities for c in t.capabilities):
                        sel = a
                        break
                if sel:
                    t.assigned_agent = sel.name
                    agent_counts[sel.name] += 1
                completed.append(orch.run_task(t))
        elif args.strategy == "broadcast":
            import concurrent.futures as _f
            tasks_to_run = []
            for t in flat:
                for a in orch.agents.values():
                    if not t.capabilities or any(c in a.profile.capabilities for c in t.capabilities):
                        tt = replace(t, assigned_agent=a.name)
                        tasks_to_run.append(tt)
                        agent_counts[a.name] += 1
            with _f.ThreadPoolExecutor(max_workers=max(4, len(orch.agents))) as ex:
                futs = [ex.submit(orch.run_task, tt) for tt in tasks_to_run]
                for fut in futs:
                    completed.append(fut.result())
        else:
            names = list(orch.agents.keys())
            idx = 0
            for t in flat:
                a = orch.agents[names[idx % len(names)]]
                tt = replace(t, assigned_agent=a.name)
                agent_counts[a.name] += 1
                completed.append(orch.run_task(tt))

        from gpt5.tools.reporting import format_task_report
        # Mermaid diagram for tasks (simple flow)
        mer = ["```mermaid", "flowchart TD"]
        last = None
        for i, t in enumerate(completed):
            nid = f"T{i}"
            mer.append(f"    {nid}([" + t.title.replace('"','\"') + "]):::task")
            if last is not None:
                mer.append(f"    {last} --> {nid}")
            last = nid
        mer.append("    classDef task fill:#e0f7fa,stroke:#006064,stroke-width:1px;")
        mer.append("```")
        report_lines = [
            f"# Mission Report — {args.objective}",
            "",
            f"Strategy: {args.strategy}",
            "",
            "## Agent Allocation",
        ]
        for name, cnt in agent_counts.items():
            report_lines.append(f"- {name}: {cnt} tasks")
        report_lines.append("")
        report_lines.append("## Tasks")
        report_lines.append(format_task_report(completed))
        report_lines.append("")
        report_lines.append("## Diagram")
        report_lines.extend(mer)
        # Gantt timeline (sequential synthetic timing)
        gantt = ["```mermaid", "gantt", "    dateFormat  X", "    title Mission Timeline", "    section Tasks"]
        for i, _t in enumerate(completed):
            start = i
            dur = 1
            gantt.append(f"    T{i} : {start}, {dur}")
        gantt.append("```")
        report_lines.append("")
        report_lines.append("## Timeline")
        report_lines.extend(gantt)
        report = "\n".join(report_lines)
        if args.report:
            Path(args.report).resolve().write_text(report, encoding="utf-8")
        memory.save_spec(f"mission:{mid}:execution", report, metadata={"objective": args.objective, "strategy": args.strategy})
        memory.mission_update(mid, status="complete", progress=1.0, report=report)
        _print({"mission": mid, "status": "complete", "tasks": len(completed), "strategy": args.strategy}, args.json)
        if args.html:
            html_text = report_html.render_html(report, title=f"mission:{mid}:execution")
            Path(args.html).resolve().write_text(html_text, encoding="utf-8")
        try:
            settings = gpt5_config.load_settings(Path.cwd())
            ae = settings.get("autoExport", {})
            name = f"mission:{mid}:execution"
            if ae.get("enabled", False) and any(name.startswith(p) for p in ae.get("prefixes", [])):
                outdir = Path(ae.get("dir", "reports"))
                ensure_dir(outdir)
                html_text = report_html.render_html(report, title=name)
                fname = sanitize_filename(f"{name}.html")
                (outdir / fname).write_text(html_text, encoding="utf-8")
        except Exception:
            pass
    mstart.set_defaults(func=cmd_mission_start)

    mstatus = mission_sub.add_parser("status")
    mstatus.add_argument("--json", action="store_true")
    def cmd_mission_status(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        rows = memory.missions()
        _print({"missions": [
            {"id": r[0], "objective": r[1], "status": r[2], "progress": r[3], "created_at": r[4], "updated_at": r[5]}
            for r in rows
        ]}, args.json)
    mstatus.set_defaults(func=cmd_mission_status)

    # Deficiency controls
    defi = sub.add_parser("deficiency", help="Deficiency detector controls")
    defi_sub = defi.add_subparsers(dest="defi_cmd")
    dlist = defi_sub.add_parser("list")
    dlist.add_argument("--json", action="store_true")
    def cmd_defi_list(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        rows = memory.list_deficiencies()
        _print({
            "deficiencies": [
                {"id": r[0], "signature": r[1], "pattern": r[2], "count": r[3], "status": r[4], "updated_at": r[5]}
                for r in rows
            ]
        }, args.json)
    dlist.set_defaults(func=cmd_defi_list)

    ddetail = defi_sub.add_parser("detail")
    ddetail.add_argument("signature")
    ddetail.add_argument("--json", action="store_true")
    def cmd_defi_detail(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        row = memory.get_deficiency(args.signature)
        if not row:
            _print({"error": "not-found"}, args.json)
            return
        payload = {
            "id": row[0],
            "signature": row[1],
            "pattern": row[2],
            "count": row[3],
            "last_error": row[4],
            "last_context": row[5],
            "status": row[6],
            "countermeasure": row[7],
            "created_at": row[8],
            "updated_at": row[9],
        }
        _print(payload, args.json)
    ddetail.set_defaults(func=cmd_defi_detail)

    dapply = defi_sub.add_parser("apply")
    dapply.add_argument("signature", help="Deficiency signature to mitigate")
    dapply.add_argument("--json", action="store_true")
    def cmd_defi_apply(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        sig = args.signature
        # For known JSONDecodeError on math vars, generate an adapter plugin
        created = None
        if sig.startswith("JSONDecodeError"):
            plugin_path = Path(__file__).resolve().parent.parent / "plugins" / "json_vars_adapter.py"
            if not plugin_path.exists():
                code = (
                    "from __future__ import annotations\n\n"
                    "def _kv_to_json(s: str) -> str:\n"
                    "    s = s.strip()\n"
                    "    if s.startswith('{') and s.endswith('}'):\n"
                    "        # fix single quotes\n"
                    "        return s.replace("'", '"')\n"
                    "    # parse key=value pairs split by comma or space\n"
                    "    parts = [p for ch in ', ' for p in s.replace('\n',' ').split(ch) if p]\n"
                    "    kv = {}\n"
                    "    for p in parts:\n"
                    "        if '=' in p:\n"
                    "            k,v = p.split('=',1)\n"
                    "            try:\n"
                    "                kv[k.strip()] = float(v)\n"
                    "            except Exception:\n"
                    "                pass\n"
                    "    if kv:\n"
                    "        import json as _json\n"
                    "        return _json.dumps(kv)\n"
                    "    return s\n\n"
                    "def before_command(args):\n"
                    "    # Adapt math expr --vars when it's not valid JSON\n"
                    "    if getattr(args,'command',None) == 'math' and getattr(args,'math_cmd',None) == 'expr':\n"
                    "        s = getattr(args, 'vars', None)\n"
                    "        if isinstance(s, str) and s:\n"
                    "            try:\n"
                    "                import json as _json\n"
                    "                _json.loads(s)\n"
                    "            except Exception:\n"
                    "                fixed = _kv_to_json(s)\n"
                    "                setattr(args, 'vars', fixed)\n"
                )
                plugin_path.write_text(code, encoding="utf-8")
                created = str(plugin_path)
            memory.set_deficiency_countermeasure(sig, f"plugin:{plugin_path.name}", status="mitigated")
            _print({"status": "applied", "plugin": created or str(plugin_path)}, args.json)
            return
        _print({"status": "no-template", "signature": sig}, args.json)
    # New streamlined apply that uses module helper and seeds regression
    def cmd_defi_apply2(args: argparse.Namespace) -> None:  # noqa: ANN001
        from gpt5.modules.deficiency import DeficiencyDetector
        memory = _memory()
        sig = args.signature
        if sig.startswith("JSONDecodeError"):
            det = DeficiencyDetector(memory)
            plugin_path = det.ensure_json_vars_adapter_plugin()
            memory.set_deficiency_countermeasure(sig, f"plugin:{Path(plugin_path).name}", status="mitigated")
            try:
                import json as _json
                argv = ["math", "expr", "sin(x)", "--vars", "x=1.2", "--json"]
                memory.add_regression(sig, _json.dumps(argv))
            except Exception:
                pass
            _print({"status": "applied", "plugin": plugin_path}, args.json)
            return
        _print({"status": "no-template", "signature": sig}, args.json)
    dapply.set_defaults(func=cmd_defi_apply2)

    ddash = defi_sub.add_parser("dashboard")
    ddash.add_argument("--output")
    ddash.add_argument("--json", action="store_true")
    def cmd_defi_dashboard(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        defs = memory.list_deficiencies()
        reg = memory.list_regressions()
        by_status = {"open": 0, "proposed": 0, "mitigated": 0}
        top = sorted(({"signature": d[1], "count": d[3], "status": d[4]} for d in defs), key=lambda x: x["count"], reverse=True)
        for d in defs:
            st = d[4] or "open"
            if st not in by_status:
                by_status[st] = 0
            by_status[st] += 1
        # Regression summary
        total_reg = len(reg)
        passed = sum(1 for r in reg if r[3] == "passed")
        failed = sum(1 for r in reg if r[3] == "failed")
        md = []
        md.append("# Deficiency Dashboard")
        md.append("")
        md.append("## Status Summary")
        md.append(f"- Open: {by_status.get('open',0)}")
        md.append(f"- Proposed: {by_status.get('proposed',0)}")
        md.append(f"- Mitigated: {by_status.get('mitigated',0)}")
        md.append("")
        md.append("## Top Signatures")
        for item in top[:10]:
            md.append(f"- {item['signature']} — count: {item['count']} — status: {item['status']}")
        md.append("")
        md.append("## Regression Summary")
        md.append(f"- Total regressions: {total_reg}")
        md.append(f"- Passed: {passed}")
        md.append(f"- Failed: {failed}")
        content = "\n".join(md)
        out = Path(args.output).resolve() if args.output else (Path.cwd() / "DEFICIENCY_DASHBOARD.md")
        out.write_text(content, encoding="utf-8")
        memory.save_spec("deficiency:dashboard", content, metadata={"open": by_status.get('open',0), "proposed": by_status.get('proposed',0), "mitigated": by_status.get('mitigated',0), "regressions": total_reg, "passed": passed, "failed": failed})
        _print({"status": "saved", "path": str(out), "open": by_status.get('open',0), "proposed": by_status.get('proposed',0), "mitigated": by_status.get('mitigated',0), "regressions": total_reg}, args.json)
        try:
            settings = gpt5_config.load_settings(Path.cwd())
            ae = settings.get("autoExport", {})
            name = "deficiency:dashboard"
            if ae.get("enabled", False) and any(name.startswith(p) for p in ae.get("prefixes", [])):
                outdir = Path(ae.get("dir", "reports"))
                ensure_dir(outdir)
                html_text = report_html.render_html(content, title=name)
                fname = sanitize_filename(f"{name}.html")
                (outdir / fname).write_text(html_text, encoding="utf-8")
        except Exception:
            pass
    ddash.set_defaults(func=cmd_defi_dashboard)

    # Regression harness
    reg = sub.add_parser("regression", help="Regression tests for deficiencies")
    reg_sub = reg.add_subparsers(dest="reg_cmd")

    radd = reg_sub.add_parser("add")
    radd.add_argument("signature")
    radd.add_argument("argv_json", help="JSON array of argv, e.g. ['math','expr','sin(x)']")
    radd.add_argument("--json", action="store_true")
    def cmd_reg_add(args: argparse.Namespace) -> None:  # noqa: ANN001
        import json as _json
        memory = _memory()
        # Validate JSON list
        argv = _json.loads(args.argv_json)
        rid = memory.add_regression(args.signature, _json.dumps(argv))
        _print({"id": rid, "signature": args.signature}, args.json)
    radd.set_defaults(func=cmd_reg_add)

    rlist = reg_sub.add_parser("list")
    rlist.add_argument("--signature")
    rlist.add_argument("--json", action="store_true")
    def cmd_reg_list(args: argparse.Namespace) -> None:  # noqa: ANN001
        memory = _memory()
        rows = memory.list_regressions(args.signature)
        _print({
            "regressions": [
                {"id": r[0], "signature": r[1], "argv": r[2], "status": r[3], "pass": r[4], "fail": r[5], "updated_at": r[6]}
                for r in rows
            ]
        }, args.json)
    rlist.set_defaults(func=cmd_reg_list)

    rrun = reg_sub.add_parser("run")
    rrun.add_argument("--signature")
    rrun.add_argument("--json", action="store_true")
    def cmd_reg_run(args: argparse.Namespace) -> None:  # noqa: ANN001
        import json as _json
        import subprocess, sys
        memory = _memory()
        rows = memory.list_regressions(args.signature)
        results = []
        for r in rows:
            rid, sig, argv_json = r[0], r[1], r[2]
            argv = _json.loads(argv_json)
            try:
                cp = subprocess.run([sys.executable, "-m", "gpt5", *argv], capture_output=True, text=True)
                passed = cp.returncode == 0
                log = (cp.stdout or "")[-500:] + (cp.stderr or "")[-500:]
                memory.update_regression_result(rid, passed, log)
                results.append({"id": rid, "signature": sig, "passed": passed})
            except Exception as exc:  # noqa: BLE001
                memory.update_regression_result(rid, False, str(exc))
                results.append({"id": rid, "signature": sig, "passed": False, "error": str(exc)})
        _print({"results": results}, args.json)
    rrun.set_defaults(func=cmd_reg_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return
    # Load settings
    settings = gpt5_config.load_settings(Path.cwd())
    # Load plugins eagerly (if enabled)
    pdir = Path(__file__).resolve().parent.parent / "plugins"
    plugin_modules = [load_plugin(p) for p in discover_plugins(pdir)] if settings.get("plugins", {}).get("enabled", True) else []

    # Always-on deficiency detector wrapper
    detector = DeficiencyDetector(memory=_memory())
    try:
        # Before-command plugin hooks
        before_command_all(plugin_modules, args)
        ret = args.func(args)
        # After-command plugin hooks
        after_command_all(plugin_modules, args, ret)
    except Exception as exc:  # noqa: BLE001
        info = detector.on_exception(where=f"cli:{getattr(args, 'command', 'unknown')}", args=vars(args), exc=exc)
        # Emit a concise error for both JSON/non-JSON modes
        is_json = bool(getattr(args, "json", False))
        payload = {"error": str(exc), "deficiency": info}
        if is_json:
            _print(payload, True)
        else:
            print(f"Error: {payload}")
        raise


if __name__ == "__main__":
    main()

