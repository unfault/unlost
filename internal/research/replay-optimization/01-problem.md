# Problem Statement

## Context

AI coding agents (Claude Code, OpenCode, Cursor, Aider, etc.) generate rich conversation histories as users work on software projects. These histories are stored locally, typically as JSONL files, and contain valuable signal:

- **Intent**: What the user was trying to accomplish
- **Decisions**: What approaches were chosen and why
- **Artifacts**: Which files, functions, and symbols were touched
- **Failures**: Where things went wrong and how they were resolved

**Unlost** is a local-first memory system that extracts this signal into structured "IntentCapsules" enabling:
1. **Semantic search** across project history
2. **Friction detection** to catch agent failure modes in real-time
3. **Narrative recall** to reconstruct "the story so far"

## The Problem

Users often want to **replay existing conversation history** into unlost:
- History from before unlost was installed
- History from sessions recorded by other tools
- Bulk import when adopting unlost on an existing project

The current replay approach processes each conversation turn through an LLM to extract a structured IntentCapsule. This is:

| Issue | Impact |
|-------|--------|
| **Slow** | ~1-2 seconds per turn (network latency + inference) |
| **Expensive** | ~$0.001-0.01 per turn depending on model |
| **Rate-limited** | Cannot parallelize beyond API limits |

**For a user with 1,000 conversation turns:**
- Current approach: 15-30 minutes, $1-10
- Desired: < 1 minute, < $0.10

## The Deeper Question

The surface problem is performance. But the deeper question is:

> **What is the minimal information we must extract from a conversation turn to provide value for downstream tasks?**

This is fundamentally an **information compression** problem:
- **Input**: Raw conversation turn (~1-10KB of text)
- **Output**: IntentCapsule (~500 bytes of structured JSON)
- **Compression ratio**: 10-20x

The LLM acts as a **lossy semantic compressor**. But different downstream tasks may need different information:

| Task | What it needs |
|------|---------------|
| Semantic search | Embedding of semantic content |
| Friction detection | Symbols touched + failure mode |
| Narrative recall | Intent + decision + rationale |

**Key insight**: We may be over-extracting for some tasks and could achieve acceptable utility with less extraction.

## Research Questions

1. **RQ1**: What information does each downstream task (search, friction, recall) actually require?

2. **RQ2**: Can we extract different information at different costs (tiered extraction)?

3. **RQ3**: What is the utility-cost Pareto frontier across extraction strategies?

4. **RQ4**: Is there a "universal good enough" representation, or must users always choose?

## Success Criteria

A successful research outcome will:

1. **Quantify** the utility-cost tradeoff with empirical data
2. **Identify** Pareto-optimal extraction strategies
3. **Provide** actionable recommendations for unlost (e.g., `--fast` vs `--quality` modes)
4. **Contribute** generalizable insights for the AI tooling community

## Scope

**In scope:**
- Extraction strategies for conversation turns
- Utility measurement for search, friction, recall
- Cost measurement (latency, API cost)
- OpenCode and Claude Code history formats

**Out of scope:**
- Real-time recording optimization (different constraints)
- Fine-tuning custom models (future work)
- Multi-modal content (images, diagrams)
