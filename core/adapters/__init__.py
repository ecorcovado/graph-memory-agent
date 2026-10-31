"""
Adapters package for GraphMemory Agent deterministic structured log ingestion.
Exports all adapters, base interfaces, and the deterministic format detector.
"""

from core.adapters.base import BaseLogAdapter, NormalizedEvent
from core.adapters.sysmon import SysmonAdapter
from core.adapters.zeek import ZeekAdapter
from core.adapters.windows_security import WindowsSecurityAdapter
from core.adapters.industrial import (
    FirewallAdapter, HistorianAdapter, ControllerAuditAdapter, ProcessTelemetryAdapter
)
from core.adapters.format_detector import FormatDetector

__all__ = [
    "BaseLogAdapter",
    "NormalizedEvent",
    "SysmonAdapter",
    "ZeekAdapter",
    "WindowsSecurityAdapter",
    "FirewallAdapter",
    "HistorianAdapter",
    "ControllerAuditAdapter",
    "ProcessTelemetryAdapter",
    "FormatDetector",
]
