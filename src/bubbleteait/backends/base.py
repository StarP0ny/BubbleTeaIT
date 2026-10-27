"""The implemented index/retrieval contract; answer generation is separate."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supports_index: bool = True
    supports_retrieve: bool = True
    returns_source_coordinates: bool = True
    generates_answers: bool = False


@runtime_checkable
class RetrievalBackend(Protocol):
    capabilities: BackendCapabilities

    def index(self, unit_id: str) -> dict: ...

    def search(self, unit_id: str, query: str, *, limit: int = 10, paths: list[str] | None = None) -> list[dict]: ...
