from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
from urllib.parse import urlparse

from gpt5.core.memory import MemoryStore
from gpt5.tools.html_extract import extract_from_html
from gpt5.tools.markdown_extract import extract_from_markdown
from gpt5.tools.web_fetch import WebFetcher


def _slugify_url(url: str) -> str:
    clean = url.replace("https://", "").replace("http://", "").strip("/")
    return clean.replace("/", "_")


def _github_readme_raw_url(github_url: str) -> Optional[str]:
    try:
        p = urlparse(github_url)
        parts = [s for s in p.path.split('/') if s]
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        for name in ("README.md", "README.MD", "Readme.md", "readme.md"):
            return f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{name}"
    except Exception:
        return None
    return None


@dataclass
class CrawlResult:
    url: str
    spec_name: str
    headings_count: int
    bullets_count: int


def crawl_and_capture(
    memory: MemoryStore,
    urls: Iterable[str],
    max_bullets: int = 200,
    prefer_readme: bool = True,
    refresh: bool = False,
) -> List[CrawlResult]:
    fetcher = WebFetcher(memory=memory)
    results: List[CrawlResult] = []
    for url in urls:
        extracted_title: Optional[str] = None
        headings: List[str] = []
        bullets: List[str] = []

        if prefer_readme and "github.com" in url:
            guess = _github_readme_raw_url(url)
            raw = None
            if guess:
                raw = None if refresh else memory.get_cached_web_content(guess)
                raw = raw or fetcher.fetch(guess)
            if raw and ("<!DOCTYPE" not in raw.upper()):
                md = extract_from_markdown(raw)
                extracted_title = md.title or url
                headings = md.headings
                bullets = md.bullets

        if not headings and not bullets:
            html = None if refresh else memory.get_cached_web_content(url)
            html = html or fetcher.fetch(url)
            html_ex = extract_from_html(html)
            extracted_title = html_ex.title or extracted_title or url
            headings = html_ex.headings
            bullets = html_ex.bullets

        title = extracted_title or url
        bullets = bullets[:max_bullets]
        # Compose markdown spec
        lines: List[str] = [f"# Crawl Summary: {title}", "", f"Source: {url}", ""]
        if headings:
            lines.append("## Headings")
            lines += [f"- {h}" for h in headings]
            lines.append("")
        if bullets:
            lines.append("## Bullets")
            lines += [f"- {b}" for b in bullets]
            lines.append("")
        content = "\n".join(lines).rstrip()

        spec_name = f"crawl:{_slugify_url(url)}"
        memory.save_spec(spec_name, content, metadata={
            "url": url,
            "title": title,
            "headings_count": len(headings),
            "bullets_count": len(bullets),
        })
        results.append(CrawlResult(url=url, spec_name=spec_name, headings_count=len(headings), bullets_count=len(bullets)))

    return results

