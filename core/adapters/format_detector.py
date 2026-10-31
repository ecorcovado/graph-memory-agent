"""
Format Detection Module for GraphMemory Agent.
Deterministically inspects file paths, extensions, byte signatures, and sample records
to select and route structured logs to the correct format adapter.
"""

from typing import Any, Dict, List, Optional, Tuple
from core.adapters.base import BaseLogAdapter
from core.adapters.sysmon import SysmonAdapter
from core.adapters.zeek import ZeekAdapter
from core.adapters.windows_security import WindowsSecurityAdapter
from core.adapters.industrial import (
    FirewallAdapter, HistorianAdapter, ControllerAuditAdapter, ProcessTelemetryAdapter
)


class FormatDetector:
    """Deterministically identifies log formats and routes to specialized adapters."""

    def __init__(self):
        self.adapters: List[BaseLogAdapter] = [
            SysmonAdapter(),
            ZeekAdapter(),
            WindowsSecurityAdapter(),
            FirewallAdapter(),
            HistorianAdapter(),
            ControllerAuditAdapter(),
            ProcessTelemetryAdapter(),
        ]

    def detect_and_get_adapter(
        self,
        file_rel: str,
        sample_records: List[Dict[str, Any]],
        manifest_source_type: Optional[str] = None,
    ) -> Tuple[str, Optional[BaseLogAdapter]]:
        """
        Detects log format and returns the corresponding canonical format label and adapter.
        Order of precedence:
        1. Explicit manifest source type (if known and matched by an adapter)
        2. Heuristic inspection of record schemas and file names
        """
        sample_record = sample_records[0] if sample_records else {}

        # 1. Check against registered adapters
        for adapter in self.adapters:
            if adapter.can_handle(manifest_source_type or "", file_rel, sample_record):
                # Derive canonical format name
                if isinstance(adapter, SysmonAdapter):
                    return "SYSMON_EVENT", adapter
                elif isinstance(adapter, ZeekAdapter):
                    return manifest_source_type or "ZEEK_FLOW", adapter
                elif isinstance(adapter, WindowsSecurityAdapter):
                    return "WINDOWS_SECURITY", adapter
                elif isinstance(adapter, FirewallAdapter):
                    return manifest_source_type or "FIREWALL_LOG", adapter
                elif isinstance(adapter, HistorianAdapter):
                    return manifest_source_type or "HISTORIAN_AUDIT", adapter
                elif isinstance(adapter, ControllerAuditAdapter):
                    return "CONTROLLER_AUDIT", adapter
                elif isinstance(adapter, ProcessTelemetryAdapter):
                    return "PROCESS_TELEMETRY", adapter

        # 2. Fallback to manifest type if provided
        if manifest_source_type:
            return manifest_source_type, None

        return "UNCLASSIFIED", None
