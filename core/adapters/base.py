"""
Base Log Adapter for GraphMemory Agent deterministic structured log ingestion.
Defines the abstract interface connecting raw event records to the NormalizedEvent model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizedEvent:
    """Canonical event model for structured telemetry records."""
    source_id: str
    record_id: str
    source_type: str
    event_time: str
    ingested_at: str
    event_type: str
    subject: str
    predicate: str
    object_entity: str
    raw_record: str
    source_location: str
    evidence_basis: str
    purdue_level: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    parser_version: str = "1.0"


class BaseLogAdapter(ABC):
    """Abstract base class for all format-specific telemetry adapters."""

    @abstractmethod
    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        """Determines if this adapter can process the given file/record format."""
        pass

    @abstractmethod
    def parse_record(
        self,
        record: Dict[str, Any],
        source_id: str,
        file_rel: str,
        line_idx: int,
        evidence_basis: str,
        event_time: str,
        ingested_at: str,
    ) -> List[NormalizedEvent]:
        """Parses a structured record dictionary into a list of NormalizedEvents."""
        pass
