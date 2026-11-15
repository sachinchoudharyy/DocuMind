import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from app.chains.chat_history import ChatHistory
from app.config import settings


class SessionManager:
    """SQLite persistence for chat sessions, messages, and session-owned PDFs."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path(settings.DATA_DIR)
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = data_dir / "chat_history.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, message_id)
            """)
            # Backward-compatible migration for databases created before citations
            # were persisted with assistant messages.
            message_columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(messages)"
                ).fetchall()
            }
            if "sources" not in message_columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'"
                )
            # Safe migration: this table is additive and does not alter existing data.
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_session_id
                ON documents(session_id, uploaded_at)
            """)

    def create_session(self, title: str = "New conversation") -> str:
        session_id = str(uuid.uuid4())
        now = ChatHistory.timestamp_now()
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, title.strip() or "New conversation", now, now))
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def add_message(self, session_id: str, role: str, content: str, sources: list[dict] | None = None):
        role = role.strip().lower()
        content = content.strip()
        if role not in {"user", "assistant"}:
            raise ValueError("role must be either 'user' or 'assistant'.")
        if not content:
            raise ValueError("Message content cannot be empty.")
        now = ChatHistory.timestamp_now()
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone() is None:
                raise ValueError(f"Unknown chat session: {session_id}")
            connection.execute("""
                INSERT INTO messages (session_id, role, content, sources, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, json.dumps(sources or []), now))
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            if role == "user":
                connection.execute("""
                    UPDATE sessions SET title = ?
                    WHERE session_id = ? AND title = 'New conversation'
                """, (content[:70], session_id))

    def get_messages(self, session_id: str) -> list[dict]:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone() is None:
                raise ValueError(f"Unknown chat session: {session_id}")
            rows = connection.execute("""
                SELECT role, content, sources, created_at FROM messages
                WHERE session_id = ? ORDER BY message_id ASC
            """, (session_id,)).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            try:
                item["sources"] = json.loads(item.get("sources") or "[]")
            except (TypeError, json.JSONDecodeError):
                item["sources"] = []
            messages.append(item)
        return messages

    def get_history_text(self, session_id: str, max_messages: int = 10) -> str:
        return ChatHistory.format_messages(
            self.get_messages(session_id), max_messages=max_messages
        )

    def list_sessions(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(DISTINCT m.message_id) AS message_count,
                       COUNT(DISTINCT d.document_id) AS document_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.session_id
                LEFT JOIN documents d ON d.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def add_document(
        self, session_id: str, original_filename: str,
        stored_path: str, size_bytes: int
    ) -> str:
        if not self.session_exists(session_id):
            raise ValueError(f"Unknown chat session: {session_id}")
        document_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO documents
                (document_id, session_id, original_filename, stored_path,
                 size_bytes, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                document_id, session_id, original_filename, stored_path,
                int(size_bytes), ChatHistory.timestamp_now()
            ))
        return document_id

    def get_documents(self, session_id: str) -> list[dict]:
        if not self.session_exists(session_id):
            raise ValueError(f"Unknown chat session: {session_id}")
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT document_id, session_id, original_filename, stored_path,
                       size_bytes, uploaded_at
                FROM documents
                WHERE session_id = ?
                ORDER BY uploaded_at ASC
            """, (session_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, session_id: str, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("""
                SELECT document_id, session_id, original_filename, stored_path,
                       size_bytes, uploaded_at
                FROM documents
                WHERE session_id = ? AND document_id = ?
            """, (session_id, document_id)).fetchone()
        return dict(row) if row else None

    def delete_document(self, session_id: str, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("""
                SELECT document_id, session_id, original_filename, stored_path,
                       size_bytes, uploaded_at
                FROM documents
                WHERE session_id = ? AND document_id = ?
            """, (session_id, document_id)).fetchone()
            if row is None:
                return None
            connection.execute(
                "DELETE FROM documents WHERE session_id = ? AND document_id = ?",
                (session_id, document_id),
            )
        return dict(row)

    def delete_session(self, session_id: str):
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
