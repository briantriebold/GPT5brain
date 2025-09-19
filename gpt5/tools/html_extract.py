from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional


@dataclass
class Extracted:
    title: str = ""
    description: str = ""
    headings: List[str] = field(default_factory=list)
    bullets: List[str] = field(default_factory=list)


class _GitHubLikeExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.extracted = Extracted()
        self._in_title = False
        self._in_heading: Optional[str] = None
        self._in_li = False

    def handle_starttag(self, tag: str, attrs):  # noqa: ANN001
        attrs_dict: Dict[str, str] = {k: v for k, v in attrs}
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3"}:
            self._in_heading = tag
        if tag == "li":
            self._in_li = True
        if tag == "meta":
            prop = attrs_dict.get("property") or attrs_dict.get("name")
            if prop in {"og:description", "description"}:
                content = attrs_dict.get("content", "").strip()
                if content and not self.extracted.description:
                    self.extracted.description = content

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"h1", "h2", "h3"} and self._in_heading == tag:
            self._in_heading = None
        if tag == "li":
            self._in_li = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.extracted.title += (" " if self.extracted.title else "") + text
        elif self._in_heading:
            self.extracted.headings.append(text)
        elif self._in_li:
            if len(text) >= 6:
                self.extracted.bullets.append(text)


def extract_from_html(html: str) -> Extracted:
    parser = _GitHubLikeExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    # Dedup preserve order
    def _uniq(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    ex = parser.extracted
    ex.headings = _uniq(ex.headings)
    ex.bullets = _uniq(ex.bullets)
    return ex

