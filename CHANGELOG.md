# Changelog

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

[0.2.3]: https://github.com/unfault/unlost/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/unfault/unlost/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/unfault/unlost/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/unfault/unlost/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/unfault/unlost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/unfault/unlost/releases/tag/v0.1.0
