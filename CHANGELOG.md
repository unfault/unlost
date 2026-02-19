# Changelog

## [0.7.0] - 2026-02-19

### Added
- **`unlost brief`**: New command that produces a staff-engineer-style codebase debrief. Answers "what do I need to know to work here without getting surprised?" by scanning all recorded history (not just recent turns), scoring capsules by importance (failure modes, explicit rationale, cross-session recurrence), and producing four structured sections: MENTAL MODEL, KEY DESIGN DECISIONS, THINGS THAT BITE, ENTRY POINTS. Ends with a GO DEEPER section of suggested `unlost` commands to drill down further. Scoped variant (`unlost brief src/governor.rs`) narrows the debrief to a specific file or concept.
- **Git commit ingestion**: Git commits are now first-class capsules. Each commit becomes an `IntentCapsule` with `category: "GitCommit"`, subject as the decision, body as the rationale, and touched files as symbols — embedded for semantic search, zero LLM cost. Deduplicates by hash across runs.
- **`unlost replay git`**: New subcommand to ingest git history on demand (`unlost replay git --max-commits 500`).
- **Automatic git ingestion**: `unlost replay opencode`, `unlost replay claude`, and `unlost init` now automatically ingest git history after their main work completes. No extra step needed.

### Changed
- **Git capsule routing**: Git capsules are included in `brief` and `query` (where historical decisions are valuable) but excluded from `recall` (which stays focused on the conversational story) and from the trajectory controller's history window (which operates on live agent turns only).

## [0.6.5] - 2026-02-18

### Fixed
- **LLM Schema Compatibility**: Fixed invalid JSON schema for `extraction_mode` field in `IntentCapsule` that caused OpenAI-compatible APIs to reject requests with HTTP 400. The field was emitting `$ref` alongside sibling keywords (`description`, `default`), which is disallowed. Now uses an inline schema via `schemars(schema_with = ...)`.

## [0.6.4] - 2026-02-17

### Changed
- **CLI Replay Clarity**: Rephrased Hybrid Mode description and summary output to explicitly state that local indexing happens for all turns, while LLM analysis is reserved for pivotal moments.

## [0.6.3] - 2026-02-17

### Changed
- **CLI Replay Summary**: Rephrased the summary output to be more intuitive, distinguishing between local indexing and selective LLM analysis with explicit API savings percentage.

## [0.6.2] - 2026-02-17

### Fixed
- **Release Stability**: Synchronized `Cargo.lock` to ensure reproducible builds with `--locked`.

## [0.6.1] - 2026-02-17

### Fixed
- **Test Stability**: Fixed compilation errors in `src/types.rs` tests due to missing `extraction_mode` field in `IntentCapsule` initializers.

## [0.6.0] - 2026-02-17

### Added
- **Hybrid Replay Default**: Implemented research-backed tiered extraction. Replay now auto-detects "pivotal" turns (emotional friction, corrective keywords, high structural churn) for LLM analysis while always indexing raw text locally for maximum recall at minimum cost.
- **Selective Extraction Heuristics**: New `is_pivotal` sensor in `flow.rs` that identifies high-signal conversation branches using emotional valence, symbol churn, and message complexity.
- **Replay Statistical Summary**: The replay CLI now reports "pivotal moment" analysis percentages, providing transparency into LLM usage and signal density.
- **CLI Replay Refactor**: Replaced ambiguous `--no-llm` flag with a clear trinary choice: `--no-extraction` (zero-cost), default (Hybrid), and `--full-extraction` (high-fidelity).
- **Maintenance Tools**: Added `--clear` flag to `unlost replay` to safely wipe existing workspace database and deduplication trackers for a fresh backfill.

### Changed
- **Optimized Search recall**: Defaulted replay to local raw text embeddings for all turns, ensuring 88.5% Recall@5 (per internal research) without requiring any API calls for "routine" turns.
- **Improved Recall Recency**: Adjusted `unlost recall` selection logic to prioritize absolute recency (last 30 mins) and latest session context, preventing older replayed historical work from drowning out current progress.

## [0.5.0] - 2026-02-17

### Added
- **Trajectory-Aware Interventions**: Moved from raw percentages to descriptive severity labels (Significant, Strong, Acute) and plain-English diagnoses (e.g., "Grounding failure", "Repetitive stall").
- **Intervention Duration**: Track and display the build-up phase ("Intervened after Xm") to better reflect the trajectory slope.
- **Contextual Topics**: Automatically capture the conversation topic (user intent) during interventions and display it in `unlost recall`.
- **Intelligent Backfilling**: Added support for backfilling topics and diagnoses for historical intervention logs by matching timestamps against conversation history.
- **Improved Time Formatting**: Human-centric elapsed time representation (e.g., "1h 11m ago", "yesterday") in recall output.

### Changed
- **Cleaner Symbol Display**: Truncated and filtered symbol lists in recall to prioritize signal over noise (e.g., "src/main.rs and 22 others").
- **Enhanced Narrative Context**: The LLM generating the recall summary now receives detailed intervention metadata (duration, diagnosis, topic) to weave friction points into the workspace story.

## [0.4.2] - 2026-02-16

### Fixed
- Sync `Cargo.lock` to fix failed release workflow

## [0.4.1] - 2026-02-16

### Fixed
- Updated changelog to include 0.3.0 and 0.4.0 entries to fix release workflow

## [0.4.0] - 2026-02-16

### Added
- **Git Grounding**: Verify agent claims against actual commit history with `--git-grounding` flag during replay
- **Fluency Sensor**: Measure assistant verbosity vs user input to detect "Blind Acceptance" risk (Nature Scientific Reports alignment)
- **Cognitive Mirror Enhancements**: `unlost metrics` now shows average verbosity and context-load inflection diagnostics
- **Turn Key Deduplication**: Fixed state gap between live recording and transcript replay using persistent turn keys

### Changed
- Promoted Trajectory-based regulator framing across documentation and website
- Removed `internal/bench` from cargo workspace members

## [0.3.0] - 2026-02-15

### Added
- **TrajectoryController**: Proactive regulator with `Stable → Watch → Intervene` state machine
- **Basin Architecture**: Classification of friction into Loop (stalls), Spec (misunderstanding), and Drift (hallucination)
- **Codebase Grounding**: Integration with `unfault-core` for sub-second symbol graph validation
- **Temporal Awareness**: "Coffee pause" logic that decays controller state across inactivity to avoid misattributions
- **Symptom Channels**: Logic churn, instruction staticness, and grounding stall sensors
- First-class `unlost replay` command for transcript backfilling
- Semantic coloring for terminal output

### Changed
- Hid internal-only commands `serve` and `record`
- Enhanced `unlost metrics` with basin-specific breakdowns and high-cost window rankings

## [0.2.7] - 2026-02-13

### Added
- `unlost shim replay opencode` to backfill OpenCode message history into capsules
  - Discovers sessions for workspace from `~/.local/share/opencode/storage/`
  - Extracts user/assistant turn pairs with usage metadata
  - Parallel processing with spinner progress
- Cost warning before replay (both Claude and OpenCode) showing turn count and LLM model
  - Suggests cheaper alternatives for expensive models (gpt-4o-mini for OpenAI, claude-3-5-haiku for Anthropic)

## [0.2.6] - 2026-02-13

### Changed
- Enhanced documentation with real command output examples
- Updated `unlost recall` example showing file-specific narrative summaries
- Updated `unlost query` example demonstrating semantic search results

## [0.2.5] - 2026-02-13

### Fixed
- `unlost where` now shows the correct workspace ID from config instead of recomputing it
  - Fixes cases where manifest files (pyproject.toml, package.json, etc.) were added/removed after initial workspace registration
  - Ensures `where` output matches the actual data location used by other commands

## [0.2.4] - 2026-02-11

### Fixed
- Recall narrative now weights recency: emphasizes latest capsules when describing "recent work"
- Friction detection skips warnings when current user emotion is neutral/positive
- Friendly error message when OPENAI_API_KEY is missing instead of panic

## [0.2.3] - 2026-02-10

### Changed
- Conversational friction detection now triggers even when no symbols are extracted (e.g. "I'm confused")
- Treat explicit confusion as friction for stateless first-message nudges

### Fixed
- Added "upset" as a frustration signal to improve heuristic detection when the emotion model under-classifies

## [0.2.2] - 2026-02-10

### Added
- `unlost shim replay claude` to backfill Claude transcript history (with best-effort de-dupe)
- Stateless friction note for clearly frustrated first messages (no capsule history required)

### Changed
- Standardize naming on `claude` (CLI, shims, docs); keep `claudecode` as a compatibility alias
- Claude transcript ingestion now records user-only turns and includes bounded `tool_result` text

### Fixed
- Claude shim cursor logic that could skip most new transcript lines after the first `Stop`

## [0.2.1] - 2026-02-10

### Changed
- Release workflow now creates GitHub releases with notes extracted from this changelog

## [0.2.0] - 2026-02-10

Version alignment release - CLI and OpenCode plugin now share the same version number.

## [0.1.1] - 2026-02-09

### Added
- OpenCode plugin integration with friction detection
- Agent session ID tracking for multi-session workflows
- Async companion recording with usage-aware output
- Query and recall filter flags (`--session`, `--agent`, etc.)
- Windows support with `ort` copy-dylibs enabled
- Best-effort touched-path recording (`touched_paths`) to improve file association in capsules

### Changed
- OpenCode config switched to plugin-only + global install pattern
- Shim architecture extracted for companion flow separation
- README overhauled with agent orientation and failure modes
- Scoped `unlost recall <file>` now prioritizes semantic matches and backfills with recent capsules only if needed
- Optional recall workspace snapshot gated behind `UNLOST_RECALL_GIT_SNAPSHOT=1`

### Fixed
- Suppressed duplicate flush jobs
- Improved recall output formatting
- Recall relevance for file scopes by expanding semantic recall and reducing unrelated context injection
- Claude Code transcript ingestion now captures touched paths from tool/snapshot events so file edits show up in memory

## [0.1.0] - 2026-01-26

### Added
- Local mood metadata (ONNX) stored alongside capsules
- Recall prompt now includes mood and per-request metadata for richer storytelling

Initial public version.

- Recorder:
  - `unlost serve` multiplexed HTTP proxy for multiple workspaces via `/w/<workspace_id>/<provider>/...`
  - `unlost record` single-workspace proxy mode
- Memory:
  - Capsule extraction into `category/intent/decision/rationale/next_steps/symbols`
  - Local embeddings (fastembed) + LanceDB storage and query
- UX:
  - `unlost recall` and `unlost query` narrative outputs
  - `unlost inspect` for raw capsule inspection
  - `unlost init` seeds capsules from code graph + optional bounded git history

[0.7.0]: https://github.com/unfault/unlost/compare/v0.6.5...v0.7.0
[0.6.4]: https://github.com/unfault/unlost/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/unfault/unlost/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/unfault/unlost/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/unfault/unlost/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/unfault/unlost/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/unfault/unlost/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/unfault/unlost/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/unfault/unlost/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/unfault/unlost/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/unfault/unlost/compare/v0.2.7...v0.3.0
[0.2.7]: https://github.com/unfault/unlost/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/unfault/unlost/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/unfault/unlost/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/unfault/unlost/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/unfault/unlost/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/unfault/unlost/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/unfault/unlost/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/unfault/unlost/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/unfault/unlost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/unfault/unlost/releases/tag/v0.1.0
