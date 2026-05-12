"""Domain exception hierarchy.

All custom exceptions inherit from AgentError so callers can catch
the base type for any agent-specific failure.
"""
from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agent errors."""


class SessionNotFoundError(AgentError):
    """Raised when a session_id is not found in the store."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session '{session_id}' not found")
        self.session_id = session_id


class AgentLoopError(AgentError):
    """Raised when the standard tool-use loop exceeds its safety cap."""


class ReActLoopError(AgentError):
    """Raised when the ReAct loop cannot converge to a Final Answer.

    Attributes:
        iterations: How many iterations were attempted.
        last_thought: Last parsed Thought string, for diagnostics.
    """

    def __init__(
        self,
        reason: str,
        iterations: int = 0,
        last_thought: str = "",
    ) -> None:
        super().__init__(f"ReAct loop failed after {iterations} iterations: {reason}")
        self.iterations = iterations
        self.last_thought = last_thought


class ReActParseError(AgentError):
    """Raised when the ReAct parser cannot extract a valid Action from Claude's text."""

    def __init__(self, raw_text: str, reason: str) -> None:
        super().__init__(f"ReAct parse error — {reason}")
        self.raw_text = raw_text
        self.reason = reason


class CodeExecutionError(AgentError):
    """Raised when Python code execution fails in any backend.

    Attributes:
        backend: 'subprocess', 'jupyter', or 'anthropic'
        stderr: Raw stderr output.
        timeout: True if caused by a timeout.
    """

    def __init__(
        self,
        message: str,
        backend: str = "subprocess",
        stderr: str = "",
        timeout: bool = False,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.stderr = stderr
        self.timeout = timeout


class PhysicalValidationError(AgentError):
    """Raised when a physical quantity fails hard validation (not just a warning)."""

    def __init__(self, quantity: str, reason: str) -> None:
        super().__init__(f"Physical validation failed for '{quantity}': {reason}")
        self.quantity = quantity
        self.reason = reason


class DatasetLoadError(AgentError):
    """Raised when a dataset cannot be loaded or parsed."""

    def __init__(self, file_name: str, reason: str) -> None:
        super().__init__(f"Cannot load dataset '{file_name}': {reason}")
        self.file_name = file_name
        self.reason = reason
