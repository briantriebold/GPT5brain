from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

from .tasks import Task


MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    capabilities TEXT,
    tags TEXT,
    dependencies TEXT,
    assigned_agent TEXT,
    status TEXT,
    created_at TEXT,
    updated_at TEXT,
    result TEXT
);

CREATE TABLE IF NOT EXISTS specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    summary TEXT NOT NULL,
    actions TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS web_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    metadata TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS missions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0.0,
    report TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS text_index (
    spec_name TEXT PRIMARY KEY,
    tokens TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS deficiencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature TEXT NOT NULL UNIQUE,
    pattern TEXT,
    count INTEGER DEFAULT 0,
    last_error TEXT,
    last_context TEXT,
    status TEXT DEFAULT 'open',
    countermeasure TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS regressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature TEXT NOT NULL,
    argv TEXT NOT NULL,
    status TEXT DEFAULT 'unknown',
    pass_count INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    last_log TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class MemoryStore:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(MEMORY_SCHEMA)
            # Migrations: add new columns if missing
            self._ensure_task_columns(conn)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_task_columns(self, conn: sqlite3.Connection) -> None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}  # type: ignore[index]
        adds = []
        if "started_at" not in cols:
            adds.append("ALTER TABLE tasks ADD COLUMN started_at TEXT")
        if "finished_at" not in cols:
            adds.append("ALTER TABLE tasks ADD COLUMN finished_at TEXT")
        if "duration_ms" not in cols:
            adds.append("ALTER TABLE tasks ADD COLUMN duration_ms INTEGER")
        for sql in adds:
            try:
                conn.execute(sql)
            except Exception:
                pass
        if adds:
            conn.commit()

    def persist_task(self, task: Task) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    title, description, capabilities, tags, dependencies,
                    assigned_agent, status, created_at, updated_at, result,
                    started_at, finished_at, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.title,
                    task.description,
                    ",".join(task.capabilities),
                    ",".join(task.tags),
                    ",".join(task.dependencies),
                    task.assigned_agent,
                    task.status,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                    task.result or "",
                    task.started_at.isoformat() if task.started_at else None,
                    task.finished_at.isoformat() if task.finished_at else None,
                    task.duration_ms,
                ),
            )
            conn.commit()

    def save_spec(self, name: str, content: str, metadata: Optional[dict] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO specs (name, content, metadata) VALUES (?, ?, ?)",
                (name, content, json.dumps(metadata or {})),
            )
            conn.commit()
        # Auto-index saved content for semantic-ish search
        try:
            self.index_text(name, content)
        except Exception:
            # Never fail caller on indexing issues
            pass

    def list_specs(self) -> Iterable[tuple]:
        with self._connect() as conn:
            return conn.execute("SELECT id, name, created_at FROM specs ORDER BY id DESC").fetchall()

    def load_spec(self, name: str) -> Optional[Tuple[int, str, str, str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, content, COALESCE(metadata,'{}') FROM specs WHERE name = ? ORDER BY id DESC LIMIT 1",
                (name,),
            ).fetchone()
            return row if row else None

    def find_specs_by_prefix(self, prefix: str) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM specs WHERE name LIKE ? ORDER BY id DESC", (f"{prefix}%",)).fetchall()
            return [r[0] for r in rows]

    def index_all_specs(self) -> int:
        """Index all specs currently stored. Returns count indexed."""
        count = 0
        with self._connect() as conn:
            rows = conn.execute("SELECT name, content FROM specs").fetchall()
        for name, content in rows:
            try:
                self.index_text(name, content)
                count += 1
            except Exception:
                continue
        return count

    # Regressions
    def add_regression(self, signature: str, argv_json: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO regressions (signature, argv, status) VALUES (?, ?, ?)",
                (signature, argv_json, "unknown"),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_regressions(self, signature: Optional[str] = None) -> list[tuple]:
        with self._connect() as conn:
            if signature:
                return conn.execute(
                    "SELECT id, signature, argv, status, pass_count, fail_count, updated_at FROM regressions WHERE signature = ? ORDER BY id DESC",
                    (signature,),
                ).fetchall()
            return conn.execute(
                "SELECT id, signature, argv, status, pass_count, fail_count, updated_at FROM regressions ORDER BY id DESC"
            ).fetchall()

    def update_regression_result(self, rid: int, passed: bool, log: str) -> None:
        with self._connect() as conn:
            if passed:
                conn.execute(
                    "UPDATE regressions SET status='passed', pass_count=pass_count+1, last_log=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (log, rid),
                )
            else:
                conn.execute(
                    "UPDATE regressions SET status='failed', fail_count=fail_count+1, last_log=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (log, rid),
                )
            conn.commit()

    def save_lesson(self, topic: str, summary: str, actions: Optional[Iterable[str]] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO lessons (topic, summary, actions) VALUES (?, ?, ?)",
                (topic, summary, json.dumps(list(actions or []))),
            )
            conn.commit()

    def cache_web_content(self, url: str, content: str, metadata: Optional[dict] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO web_cache (url, content, metadata) VALUES (?, ?, ?)",
                (url, content, json.dumps(metadata or {})),
            )
            conn.commit()
        # Also index cached content for search
        try:
            self.index_text(f"web:{url}", content)
        except Exception:
            pass

    def get_cached_web_content(self, url: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT content FROM web_cache WHERE url = ?", (url,)).fetchone()
            return row[0] if row else None

    # Lightweight token index for semantic-ish search
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in ''.join([c.lower() if c.isalnum() else ' ' for c in text]).split() if t]

    def index_text(self, spec_name: str, content: str) -> None:
        tokens = list(dict.fromkeys(self._tokenize(content)))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO text_index (spec_name, tokens) VALUES (?, ?)",
                (spec_name, json.dumps(tokens)),
            )
            conn.commit()

    def search_text(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q = set(self._tokenize(query))
        with self._connect() as conn:
            rows = conn.execute("SELECT spec_name, tokens FROM text_index").fetchall()
        scored: list[tuple[str, float]] = []
        for name, tokens_json in rows:
            try:
                toks = set(json.loads(tokens_json))
            except Exception:
                toks = set()
            if not toks:
                continue
            inter = len(q & toks)
            union = len(q | toks)
            score = (inter / union) if union else 0.0
            if score > 0:
                scored.append((name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # Missions and events
    def mission_create(self, objective: str, status: str = "created") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO missions (objective, status) VALUES (?, ?)",
                (objective, status),
            )
            conn.commit()
            return int(cur.lastrowid)

    def mission_update(self, mission_id: int, status: Optional[str] = None, progress: Optional[float] = None, report: Optional[str] = None) -> None:
        sets = []
        vals: list[object] = []
        if status is not None:
            sets.append("status = ?")
            vals.append(status)
        if progress is not None:
            sets.append("progress = ?")
            vals.append(progress)
        if report is not None:
            sets.append("report = ?")
            vals.append(report)
        sets.append("updated_at = CURRENT_TIMESTAMP")
        vals.append(mission_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE missions SET {', '.join(sets)} WHERE id = ?", vals)
            conn.commit()

    def missions(self) -> list[tuple]:
        with self._connect() as conn:
            return conn.execute("SELECT id, objective, status, progress, created_at, updated_at FROM missions ORDER BY id DESC").fetchall()

    def emit_event(self, kind: str, payload: Optional[dict] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (kind, payload) VALUES (?, ?)",
                (kind, json.dumps(payload or {})),
            )
            conn.commit()

    # Deficiencies
    def update_deficiency(self, signature: str, pattern: str, last_error: str, last_context: str) -> tuple[int, int]:
        """Upsert deficiency and increment count. Returns (id, new_count)."""
        with self._connect() as conn:
            row = conn.execute("SELECT id, count FROM deficiencies WHERE signature = ?", (signature,)).fetchone()
            if row:
                did, cnt = int(row[0]), int(row[1]) + 1
                conn.execute(
                    "UPDATE deficiencies SET pattern=?, count=?, last_error=?, last_context=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (pattern, cnt, last_error, last_context, did),
                )
                conn.commit()
                return did, cnt
            cur = conn.execute(
                "INSERT INTO deficiencies (signature, pattern, count, last_error, last_context) VALUES (?, ?, ?, ?, ?)",
                (signature, pattern, 1, last_error, last_context),
            )
            conn.commit()
            return int(cur.lastrowid), 1

    def list_deficiencies(self) -> list[tuple]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, signature, pattern, count, status, updated_at FROM deficiencies ORDER BY updated_at DESC"
            ).fetchall()

    def get_deficiency(self, signature: str) -> Optional[tuple]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, signature, pattern, count, last_error, last_context, status, countermeasure, created_at, updated_at FROM deficiencies WHERE signature=?",
                (signature,),
            ).fetchone()

    def set_deficiency_countermeasure(self, signature: str, countermeasure: str, status: str = "mitigated") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE deficiencies SET countermeasure=?, status=?, updated_at=CURRENT_TIMESTAMP WHERE signature=?",
                (countermeasure, status, signature),
            )
            conn.commit()
