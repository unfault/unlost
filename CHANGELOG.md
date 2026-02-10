# Changelog

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

[0.2.0]: https://github.com/unfault/unlost/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/unfault/unlost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/unfault/unlost/releases/tag/v0.1.0
