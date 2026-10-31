"""
CLI Runner for Graph Memory Agent
Demonstrates multi-hop graph exploration using local Ollama (qwen2.5-coder:7b).
Supports both unstructured document ingestion (--file) and deterministic structured case bundles (--case).
"""

import os
import sys
import argparse
from core.graph_builder import LocalOllamaClient, GraphBuilder
from core.agent import GraphReaderAgent
from core.case_loader import CaseLoader


DEFAULT_DEMO_TEXT = """
Project Aegis is a cyber defence telemetry pipeline developed by the Australian Cyber Security Operations team.
The lead engineer who architected the ingestion engine for Project Aegis is Marcus Thorne.
Marcus Thorne previously led the threat research division at Sentinel Core Labs based in Melbourne.
Sentinel Core Labs was founded in 2018 inside the historic Old Treasury Building in Melbourne, Victoria.
The primary protocol used by Project Aegis to transmit zero-trust authentication tokens is called Cerberus-v2.
"""

DEFAULT_QUESTION = "In which historic building was the lab founded where the lead architect of Project Aegis previously worked?"


def main():
    parser = argparse.ArgumentParser(description="Graph Memory Agent - Local Multi-Hop LLM Reasoning")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:7b", help="Ollama model name (default: qwen2.5-coder:7b)")
    parser.add_argument("--file", type=str, default=None, help="Path to custom text file or incident narrative")
    parser.add_argument("--case", type=str, default=None, help="Path to structured case bundle directory (e.g. examples/purdue_ot_incident)")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, help="Target multi-hop question")
    args = parser.parse_args()

    print("=" * 70)
    print("  GRAPH MEMORY AGENT (Inspired by GraphReader - arXiv:2406.14550)")
    print(f"  Target Model: {args.model} (100% Local via Ollama)")
    print("=" * 70)

    client = LocalOllamaClient(model=args.model)

    # 1. Deterministic Case Bundle Ingestion (v0.4 Pipeline)
    if args.case:
        try:
            print(f"[*] Ingesting structured case bundle from: {args.case}")
            loader = CaseLoader(args.case)
            print("\n[Phase 1] Ingesting Structured Case Bundle...")
            kg = loader.load_case()
            if kg.ingestion_report and not kg.ingestion_report.is_complete:
                print("\n" + "!" * 70)
                print("  [!] WARNING: CASE INGESTION IS INCOMPLETE OR DEGRADED")
                print(f"  Reason: {kg.ingestion_report.degraded_reason or 'Missing or rejected sources'}")
                if kg.ingestion_report.missing_sources:
                    print(f"  Missing sources: {', '.join(kg.ingestion_report.missing_sources)}")
                if kg.ingestion_report.empty_sources:
                    print(f"  Empty sources: {', '.join(kg.ingestion_report.empty_sources)}")
                print("  Proceeding under degraded mode: conclusions requiring missing sources will fail closed.")
                print("!" * 70 + "\n")
        except Exception as e:
            print(f"[!] Error loading case '{args.case}': {e}")
            sys.exit(1)

    # 2. Unstructured Document Ingestion
    elif args.file:
        try:
            target_path = args.file
            if os.path.isdir(target_path):
                overview_path = os.path.join(target_path, "incident_overview.txt")
                if os.path.exists(overview_path):
                    target_path = overview_path
                    print(f"[*] Case directory provided. Loaded narrative: {target_path}")
            with open(target_path, "r", encoding="utf-8") as f:
                target_text = f.read()
            print(f"[*] Loaded document from: {target_path} ({len(target_text)} chars)")
            builder = GraphBuilder(client)
            print("\n[Phase 1] Building Knowledge Graph from text...")
            kg = builder.build_graph(target_text)
        except Exception as e:
            print(f"[!] Error reading file '{args.file}': {e}")
            sys.exit(1)

    # 3. Default Demo Benchmark
    else:
        print("[*] No --case or --file specified. Using built-in Project Aegis benchmark.")
        builder = GraphBuilder(client)
        print("\n[Phase 1] Building Knowledge Graph from benchmark text...")
        kg = builder.build_graph(DEFAULT_DEMO_TEXT)

    print("\n[Phase 2] Initializing GraphReader Agent...")
    agent = GraphReaderAgent(client, kg, max_steps=6, verbose=True)

    print(f"\n[Phase 3] Solving Multi-Hop Query:\n>> '{args.question}'")
    answer = agent.run(args.question)

    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    print("=" * 70)


if __name__ == "__main__":
    main()
