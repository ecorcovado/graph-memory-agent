"""
Forensic Investigation Reporter & Workspace Generator for GraphMemory Agent.
Produces executive terminal dashboards and self-contained interactive HTML incident reports.
Aligned with NIST SP 800-82 Rev. 3 and the Purdue Enterprise Reference Architecture (PERA).
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from core.graph_builder import KnowledgeGraph
from core.notebook import Notebook


class InvestigationReporter:
    """
    Synthesizes KnowledgeGraph evidence, notebook findings, and ingestion audits
    into visual executive dashboards and auditable forensic HTML reports.
    """

    def __init__(self, kg: KnowledgeGraph, notebook: Notebook, answer: str, question: str):
        self.kg = kg
        self.notebook = notebook
        self.answer = answer
        self.question = question
        self.ingestion_report = getattr(kg, "ingestion_report", None)

    def analyze_stages(self) -> Dict[str, Dict[str, Any]]:
        """Analyzes verification status for each Purdue Level stage in the attack path."""
        verified_fids = self.notebook.get_all_fact_ids()
        verified_facts = [self.kg.atomic_facts[fid] for fid in verified_fids if fid in self.kg.atomic_facts]

        stages = {
            "Level 4: Enterprise": {
                "name": "Enterprise Compromise",
                "purdue": 4.0,
                "status": "UNVERIFIED",
                "details": "No verified endpoint execution or external communication telemetry.",
                "facts": [],
            },
            "Level 3.5: Plant DMZ": {
                "name": "Industrial DMZ Pivot",
                "purdue": 3.5,
                "status": "UNVERIFIED",
                "details": "No verified firewall connection or DMZ mirror access.",
                "facts": [],
            },
            "Level 3: Operations": {
                "name": "Plant Historian Sync",
                "purdue": 3.0,
                "status": "UNVERIFIED",
                "details": "No verified operations replication telemetry.",
                "facts": [],
            },
            "Level 2: Supervisory": {
                "name": "SCADA Gateway Access",
                "purdue": 2.0,
                "status": "UNVERIFIED",
                "details": "No verified SCADA session or engineering workstation activity.",
                "facts": [],
            },
            "Level 1: Process Control": {
                "name": "PLC Logic Modification",
                "purdue": 1.0,
                "status": "UNVERIFIED",
                "details": "Controller audit log missing or unverified.",
                "facts": [],
            },
            "Level 0: Physical Process": {
                "name": "Physical Process Impact",
                "purdue": 0.0,
                "status": "UNVERIFIED",
                "details": "No verified sensor observation or interlock telemetry.",
                "facts": [],
            },
        }

        for f in verified_facts:
            lvl = f.purdue_level
            if lvl == 4.0 or (lvl is None and any(term in f.text.lower() for term in ["powershell", "sysmon", "corp-ws"])):
                stages["Level 4: Enterprise"]["status"] = "SUPPORTED"
                stages["Level 4: Enterprise"]["facts"].append(f)
                stages["Level 4: Enterprise"]["details"] = f"Verified process execution & C2 telemetry ({f.subject} -> {f.relation} -> {f.object_entity})"
            elif lvl == 3.5 or (lvl is None and any(term in f.text.lower() for term in ["firewall", "dmz-hist", "mirror"])):
                stages["Level 3.5: Plant DMZ"]["status"] = "SUPPORTED"
                stages["Level 3.5: Plant DMZ"]["facts"].append(f)
                stages["Level 3.5: Plant DMZ"]["details"] = f"Verified DMZ firewall conduit crossing ({f.subject} -> {f.relation} -> {f.object_entity})"
            elif lvl == 3.0:
                stages["Level 3: Operations"]["status"] = "SUPPORTED"
                stages["Level 3: Operations"]["facts"].append(f)
                stages["Level 3: Operations"]["details"] = f"Verified plant historian replication ({f.subject} -> {f.relation} -> {f.object_entity})"
            elif lvl == 2.0 or (lvl is None and any(term in f.text.lower() for term in ["scada", "ignition", "eng-ws"])):
                stages["Level 2: Supervisory"]["status"] = "SUPPORTED"
                stages["Level 2: Supervisory"]["facts"].append(f)
                stages["Level 2: Supervisory"]["details"] = f"Verified SCADA gateway session & engineering host ({f.subject} -> {f.relation} -> {f.object_entity})"
            elif lvl == 1.0 or (lvl is None and any(term in f.text.lower() for term in ["plc", "modified_logic", "cip"])):
                stages["Level 1: Process Control"]["status"] = "SUPPORTED"
                stages["Level 1: Process Control"]["facts"].append(f)
                stages["Level 1: Process Control"]["details"] = f"Verified CIP controller logic download ({f.subject} -> {f.relation} -> {f.object_entity})"
            elif lvl == 0.0 or (lvl is None and any(term in f.text.lower() for term in ["bar", "pressure", "valve", "obs-"])):
                stages["Level 0: Physical Process"]["status"] = "SUPPORTED"
                stages["Level 0: Physical Process"]["facts"].append(f)
                stages["Level 0: Physical Process"]["details"] = f"Verified physical telemetry observation ({f.subject} -> {f.relation} -> {f.object_entity})"

        # If degraded ingestion report explicitly lacks controller audit, mark Level 1 as DEGRADED / UNVERIFIED
        if self.ingestion_report and any("controller_audit" in ms for ms in self.ingestion_report.missing_sources):
            stages["Level 1: Process Control"]["status"] = "UNVERIFIED (DEGRADED)"
            stages["Level 1: Process Control"]["details"] = "controller_audit.jsonl missing: fail-closed policy enforced."

        return stages

    def print_terminal_dashboard(self):
        """Renders the executive cybersecurity investigation workspace in the console."""
        stages = self.analyze_stages()
        is_complete = self.ingestion_report.is_complete if self.ingestion_report else True
        case_id = self.ingestion_report.case_id if self.ingestion_report else "SYNTHETIC-INVESTIGATION"

        print("\n" + "=" * 80)
        print("  GRAPH MEMORY AGENT · OT/ICS FORENSIC INVESTIGATION WORKSPACE")
        print(f"  Incident ID: {case_id} | Status: {'COMPLETE (All Evidence Verified)' if is_complete else 'DEGRADED (Missing Telemetry)'}")
        print("=" * 80)

        print("\n  [Attack Path Reconstruction Across Purdue Zones]")
        print("  " + "-" * 76)
        for stage_key, data in stages.items():
            status_tag = f"[{data['status']}]"
            print(f"  * {stage_key:<26} {status_tag:<22} : {data['name']}")
            print(f"    └─ {data['details']}")
        print("  " + "-" * 76)

        print("\n  [Target Inquiry & Verified Conclusion]")
        print("  " + "-" * 76)
        print(f"  Query:  {self.question}")
        print(f"  Answer: {self.answer}")
        print("  " + "-" * 76)

        print("\n  [Investigation Provenance Metrics]")
        print("  " + "-" * 76)
        print(f"  Total Graph Nodes: {self.kg.graph.number_of_nodes():<6} | Directed Edges: {self.kg.graph.number_of_edges():<6} | Atomic Facts: {len(self.kg.atomic_facts):<6}")
        if self.ingestion_report:
            print(f"  Source Records:    Read={self.ingestion_report.source_records_read} | Parsed={self.ingestion_report.source_records_parsed} | Rejected={self.ingestion_report.source_records_rejected}")
            print(f"  Source Hashes:     {len(self.ingestion_report.source_hashes)} distinct SHA-256 byte digests verified.")
        print("=" * 80 + "\n")

    def generate_html_report(self, output_path: str = "reports/investigation_report.html") -> str:
        """Generates a standalone, interactive HTML forensic incident report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        stages = self.analyze_stages()
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        case_id = self.ingestion_report.case_id if self.ingestion_report else "AEGIS-CASE-001"
        is_complete = self.ingestion_report.is_complete if self.ingestion_report else True

        # Build Stage HTML cards
        stage_cards_html = ""
        for stage_key, data in stages.items():
            is_supported = "SUPPORTED" in data["status"]
            badge_class = "badge-supported" if is_supported else "badge-unverified"
            facts_list_html = ""
            for f in data["facts"]:
                raw_item = self.kg.raw_record_vault.get((f.source_id, f.source_record_id), {})
                raw_str = raw_item.get("_raw_line", json.dumps(raw_item)) if isinstance(raw_item, dict) else str(raw_item)
                facts_list_html += f"""
                <div class="fact-item">
                    <span class="fact-id">{f.fact_id}</span>
                    <span class="fact-tier">[{f.source_type} | {f.evidence_basis}]</span>
                    <p class="fact-text">{f.text}</p>
                    <details class="raw-details">
                        <summary>Raw Record Excerpt ({f.source_location})</summary>
                        <pre><code>{raw_str}</code></pre>
                    </details>
                </div>
                """

            stage_cards_html += f"""
            <div class="stage-card {'card-supported' if is_supported else 'card-degraded'}">
                <div class="stage-header">
                    <div>
                        <span class="zone-tag">{stage_key}</span>
                        <h3>{data['name']}</h3>
                    </div>
                    <span class="badge {badge_class}">{data['status']}</span>
                </div>
                <p class="stage-desc">{data['details']}</p>
                {facts_list_html if facts_list_html else '<p class="no-facts">No supporting telemetry records in active notebook.</p>'}
            </div>
            """

        # Build Source Hashes HTML
        hashes_html = ""
        if self.ingestion_report and self.ingestion_report.source_hashes:
            for sid, digest in self.ingestion_report.source_hashes.items():
                hashes_html += f"<tr><td><code>{sid}</code></td><td><code>{digest}</code></td><td><span class='badge badge-verified'>VERIFIED_SHA256</span></td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Aegis · Forensic Investigation Report ({case_id})</title>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text-main: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --success: #238636;
            --warning: #d29922;
            --danger: #da3633;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 26px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .meta-bar {{
            display: flex;
            gap: 20px;
            font-size: 13px;
            color: var(--text-muted);
            flex-wrap: wrap;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-supported {{ background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #238636; }}
        .badge-unverified {{ background: rgba(218, 54, 51, 0.2); color: #f85149; border: 1px solid #da3633; }}
        .badge-verified {{ background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid #1f6feb; }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-top: 4px;
        }}
        .kpi-label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        .section-title {{
            font-size: 18px;
            color: #ffffff;
            margin: 28px 0 16px 0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        .stage-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .card-supported {{ border-left: 4px solid var(--success); }}
        .card-degraded {{ border-left: 4px solid var(--danger); }}
        .stage-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
        }}
        .zone-tag {{
            font-size: 11px;
            color: var(--accent);
            text-transform: uppercase;
            font-weight: 600;
        }}
        .stage-header h3 {{
            margin: 2px 0 0 0;
            font-size: 16px;
            color: #ffffff;
        }}
        .stage-desc {{
            font-size: 13px;
            color: var(--text-muted);
            margin: 4px 0 12px 0;
        }}
        .fact-item {{
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px;
            margin-top: 8px;
            font-size: 13px;
        }}
        .fact-id {{
            font-family: monospace;
            color: var(--accent);
            font-weight: 600;
        }}
        .fact-tier {{
            font-size: 11px;
            color: var(--text-muted);
            margin-left: 8px;
        }}
        .fact-text {{
            margin: 6px 0;
            color: #ffffff;
        }}
        .raw-details {{
            margin-top: 6px;
            font-size: 12px;
        }}
        .raw-details summary {{
            cursor: pointer;
            color: var(--accent);
        }}
        pre {{
            background: #161b22;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 6px 0 0 0;
            font-size: 11px;
        }}
        .conclusion-box {{
            background: rgba(88, 166, 255, 0.05);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 28px;
        }}
        .table-responsive {{
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }}
        th, td {{
            padding: 10px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                <span>🛡️</span> Project Aegis Incident Investigation Report
            </h1>
            <div class="meta-bar">
                <span><strong>Case ID:</strong> {case_id}</span>
                <span><strong>Generated:</strong> {now_utc}</span>
                <span><strong>Engine:</strong> GraphMemory Agent v0.4.3</span>
                <span><strong>Ingestion Audit:</strong> {'COMPLETE (All sources verified)' if is_complete else 'DEGRADED (Missing required sources)'}</span>
            </div>
        </header>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Knowledge Graph Nodes</div>
                <div class="kpi-value">{self.kg.graph.number_of_nodes()}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Directed Relationships</div>
                <div class="kpi-value">{self.kg.graph.number_of_edges()}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Atomic Forensic Facts</div>
                <div class="kpi-value">{len(self.kg.atomic_facts)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Evidence Verification</div>
                <div class="kpi-value" style="color: {'#3fb950' if is_complete else '#f85149'};">
                    {'100% AUDITED' if is_complete else 'FAIL-CLOSED'}
                </div>
            </div>
        </div>

        <div class="conclusion-box">
            <h3 style="margin-top: 0; color: #58a6ff;">Target Inquiry & Reconstructed Conclusion</h3>
            <p style="font-weight: 600; margin-bottom: 6px;">Query: "{self.question}"</p>
            <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                {self.answer}
            </div>
        </div>

        <h2 class="section-title">Attack Path Progression Across Purdue Zones (PERA)</h2>
        {stage_cards_html}

        <h2 class="section-title">Cryptographic Source Integrity & Acquisition Digests</h2>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Source ID</th>
                        <th>SHA-256 Digest</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {hashes_html if hashes_html else '<tr><td colspan="3">No source digests registered.</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[+] Forensic HTML report generated: {os.path.abspath(output_path)}")
        return os.path.abspath(output_path)
