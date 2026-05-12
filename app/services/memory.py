"""In-memory session stores.

Two separate stores:
  session_store         — for MVP AgentSession (unchanged)
  analysis_session_store — for new AnalysisSession
"""
from __future__ import annotations

from app.domain.analysis_models import AnalysisSession
from app.domain.models import AgentSession


class InMemorySessionStore:
    """Key-value store for AgentSession objects (MVP)."""

    def __init__(self) -> None:
        self._store: dict[str, AgentSession] = {}

    def get_or_create(self, session_id: str) -> AgentSession:
        if session_id not in self._store:
            self._store[session_id] = AgentSession(session_id=session_id)
        return self._store[session_id]

    def save(self, session: AgentSession) -> None:
        self._store[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class AnalysisSessionStore:
    """Key-value store for AnalysisSession objects."""

    def __init__(self) -> None:
        self._store: dict[str, AnalysisSession] = {}

    def get_or_create(self, session_id: str) -> AnalysisSession:
        if session_id not in self._store:
            self._store[session_id] = AnalysisSession(session_id=session_id)
        return self._store[session_id]

    def get(self, session_id: str) -> AnalysisSession | None:
        return self._store.get(session_id)

    def save(self, session: AnalysisSession) -> None:
        self._store[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._store)


session_store = InMemorySessionStore()
analysis_session_store = AnalysisSessionStore()
