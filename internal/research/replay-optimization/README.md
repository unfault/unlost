# Research: Extraction Granularity for Agent Memory Systems

## Abstract

When building memory systems for AI coding agents, we face a fundamental question: how much information must we extract from conversation history to provide value? Current approaches use LLM-based extraction for every turn, which is slow and expensive. We hypothesize that different use cases (search, friction detection, narrative recall) have different information requirements, and that a tiered extraction strategy can dramatically reduce cost while preserving utility.

This research aims to:
1. Characterize the information requirements of each downstream task
2. Measure the utility-cost tradeoff across extraction strategies
3. Identify the Pareto-optimal approaches for different user priorities

## Status

- [x] Problem definition (01-problem.md)
- [x] Hypothesis formulation (02-hypothesis.md)
- [x] Methodology design (03-methodology.md)
- [x] Data characterization (04-data.md)
- [x] Ground truth design (05-ground-truth-design.md)
- [x] Initial Results: Search (06-results.md)
- [x] Ground truth creation (ground_truth.json)
- [x] Experiment implementation (robustness_marathon.py)
- [x] Results analysis (06-results.md, PROGRESS.md)
- [x] Academic Alignment (EASE'25 paper)
- [ ] Blog post draft (07-blog-draft.md) - Pending final review

## Current Findings (Sprint Set - 50 turns)

| Strategy | Latency (50 turns) | Search Recall@5 | Friction F1 |
|----------|-------------------|-----------------|-------------|
| **Raw Text Embedding** | ~15s (300ms/turn) | **87%** | 0.0 |
| **Level 0 Heuristic** | ~10s (200ms/turn) | 74% | 0.0 |

**Insight**: Raw turn text provides better retrieval context than naive local extraction for search. Latency is dominated by local embedding generation. Friction detection remains a challenge for zero-LLM strategies.

## Documents

| Document | Description |
|----------|-------------|
| [00-research-framing.md](./00-research-framing.md) | Philosophical and theoretical foundations |
| [01-problem.md](./01-problem.md) | Problem statement and motivation |
| [02-hypothesis.md](./02-hypothesis.md) | Research hypotheses |
| [03-methodology.md](./03-methodology.md) | Experimental methodology |
| [04-data.md](./04-data.md) | Dataset characterization |
| [05-ground-truth-design.md](./05-ground-truth-design.md) | Design of the Golden Set |
| [06-results.md](./06-results.md) | Experimental results |
| [07-blog-draft.md](./07-blog-draft.md) | Blog post draft |

## Quick Links

- [00-research-framing.md](./00-research-framing.md) - The "Deeper Question" foundations
- [notes-initial.md](./notes-initial.md) - Initial brainstorming notes
