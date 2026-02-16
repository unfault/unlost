# Changelog

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
