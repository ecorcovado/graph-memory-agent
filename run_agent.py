"""
CLI Runner for Graph Memory Agent
Demonstrates multi-hop graph exploration using local Ollama (qwen2.5-coder:7b).
"""

import sys
import argparse
from core.graph_builder import LocalOllamaClient, GraphBuilder
from core.agent import GraphReaderAgent


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
    parser.add_argument("--model", type=str, default="qwen2.5-coder:7b", help="Ollama model name")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, help="Target question")
    args = parser.parse_args()

    print("=" * 70)
    print("  GRAPH MEMORY AGENT (Inspired by GraphReader - arXiv:2406.14550)")
    print(f"  Target Model: {args.model} (100% Local via Ollama)")
    print("=" * 70)

    client = LocalOllamaClient(model=args.model)
    builder = GraphBuilder(client)

    print("\n[Phase 1] Building Knowledge Graph from text...")
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
