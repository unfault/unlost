# Changelog

## [0.10.0] - 2026-02-24

### Added
- **`unlost interventions`**: New diagnostics command to show recent friction interventions applied to agents. Displays timestamp, building time (how long friction was building), severity/intensity score, cause/diagnosis, topic (user intent), symbols involved, user emotion, and symptom channels. Supports `--limit`, `--since`, `--until` filters.
- **`unlost challenge --deep`**: `challenge` is now concise by default — outputs only `THE DECISION`, `ALTERNATIVES` (2-3 options, no Cost/Evidence fields), and `VERDICT`. Pass `--deep` to get the full analysis with `UNKNOWNS` and `PROBES` sections.
- **Git provenance in capsules**: Each capsule now records `head_sha` (git HEAD at buffer-open time) and `commit_sha` (HEAD at flush time, when it has moved). Both fields are stored in LanceDB and displayed in `unlost inspect`, making it possible to correlate memory entries with exact commits.
- **HyPE questions surfaced in `inspect` and scan**: `unlost inspect` now displays the pre-generated HyPE questions stored alongside each capsule, so stored vectors can be verified. `scan_capsules` also now reads `questions_text` from LanceDB rather than returning an empty list.

### Fixed
- **Spurious spec interventions on short/meta inputs**: Three paths in the governor were firing alignment-check notes out of turn on messages like `"yes"` or short directives like `"Extend trace."`. (1) The ambient spec note now requires ≥4 words in intent and a non-empty decision before producing a check-in, preventing nonsense notes like `"my current understanding is 'yes'. Next I'll do ''"`. (2) The `north_star` selection in the high-debt spec branch now requires ≥6 words and excludes meta phrases (`casual`, `check-in`, `continue`, etc.), so `"User initiated a casual check-in"` can no longer appear as `"Original Goal"` and undermine the note. (3) The blind-acceptance intensity boost now skips known confirmation words (`yes`, `ok`, `sure`, `proceed`, `do it`, etc.) so decisive short inputs no longer inflate intensity toward an unwarranted intervention.
- **Windows stack overflow on startup**: The shim binary (`unlost shim opencode`) was crashing on Windows before writing the `{"ready":true}` signal because the default Windows main-thread stack (1 MiB) is too small for the deeply-nested tokio async state machine. Added `.cargo/config.toml` with `/STACK:8388608` for `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc` targets, matching the 8 MiB default on Linux/macOS.
- **Duplicate capsules on plugin restart**: The OpenCode plugin now computes a `turn_key` (`${userMessageId}:${assistantMessageId}`) and sends it with every record request, making the server-side deduplication guard reachable across restarts. Previously, restarting the plugin cleared in-memory dedup state and caused any re-surfaced exchange to be written to `capsules.jsonl` a second time.
- **Failure mode interventions are now session-scoped**: `evaluate_failure_modes` previously read the last 5 capsules from LanceDB with no session boundary awareness. A `Drift`, `RetrySpiral`, `Rediscovery`, or `FalseProgress` tag on the final capsule of a previous session would fire a system note on the very first message of the next session — with no actual friction to detect. The function now accepts a session ID and filters history to the current session only before evaluating. If no capsules from the current session exist yet, it returns `None`. Sessions with no known ID (e.g. the HTTP proxy path) retain the previous cross-session behaviour.

### Changed
- **HyPE-aligned retrieval for all commands**: Each command now frames its user query with a command-specific intent prefix before embedding, turning retrieval into a question-to-question match rather than a keyword-to-document match. This exploits the HyPE (Hypothetical Prompt Embeddings) questions already stored in `questions_text` at indexing time — without any extra LLM call at query time. Framing per command:
  - `recall`: *"What happened with \<target\>?"*
  - `brief`: *"Why is the current state of \<target\> the way it is?"*
  - `challenge`: *"Was the decision about \<target\> the right call?"*
  - `explore`: *"What are the alternatives and trade-offs for \<target\>?"*
  - `trace`: *"What sequence of decisions led to \<target\>?"*
  If the user's input already contains a `?`, the prefix is prepended as a soft bias rather than replacing the phrasing.
- **`trace` fan-out quality guard**: Fan-out (symbol-linked) capsules that carry no meaningful content — empty `intent` *and* empty `decision` — are now dropped before entering the causal chain. Previously all symbol-linked rows were admitted regardless of content, letting ghost/replay extractions pollute the chain.

## [0.9.0] - 2026-02-23

### Added
- **Git tag ingestion**: Git tags are now first-class capsules (`category: "GitTag"`). Each tag captures its name, dereferenced commit SHA, creator date, tag message, and the files touched by the tagged commit — so queries like "what changed between v0.8.0 and v0.9.0?" can be answered from memory. Deduplicates by tag name in `git/ingested_tags.txt`. Works for both annotated and lightweight tags.
- **Live changelog re-ingest on Stop hook**: The Claude shim now calls `ingest_changelog` at the end of every session. If `CHANGELOG.md` was in the session's touched paths (or has un-ingested versions), new entries are captured immediately without requiring a manual `unlost replay`. Zero-LLM cost, idempotent.
- **Live tag ingest on Stop hook**: The Claude shim also calls `ingest_git_tags` on every Stop hook, so tags created during a session are captured as boundary capsules before the next session begins.
- **OpenCode stdio shim session-end ingest**: The OpenCode stdio shim (`unlost shim opencode`) now runs `ingest_git_tags` and `ingest_changelog` when stdin closes (session end), giving it parity with the Claude Stop hook. Also drains the background worker before process exit to ensure no capsules are lost.

### Changed
- **`ingest_git_tags` wired into all batch paths**: `unlost init`, `unlost replay claude`, and `unlost replay opencode` now all call `ingest_git_tags` immediately after `ingest_git_commits`, so tag history is backfilled alongside commit history.

## [0.8.0] - 2026-02-23

### Added
- **`unlost explore`**: New command for forward-looking planning grounded in workspace memory. Given a scenario or goal (e.g. `unlost explore "should we keep lancedb or move to sqlite+fts?"`), retrieves the most relevant capsules via semantic search combined with an importance-scored full scan (failure modes, rationale, cross-session recurrence). Capsules are context — not a cage — so the LLM can reason beyond them while clearly labelling what comes from memory (`[memory]`) vs. external knowledge (`[outside]`). Output sections: CONTEXT FROM MEMORY, PATHS WORTH CONSIDERING, TENSIONS, QUESTIONS TO SIT WITH, IF YOU GO FURTHER.
- **`unlost challenge`**: New command to pressure-test a past decision or technology choice (e.g. `unlost challenge "lancedb"` or `unlost challenge "is our code currently properly organized?"`). Uses three evidence sources: (1) the live code graph via unfault-core (hotspots, dependency topology, routes, file list — ground truth even when capsules are thin), (2) changelog capsules (version history), and (3) conversational memory capsules (decisions, rationale, failure modes). Output sections: THE DECISION, ALTERNATIVES (as readable named cards with Upside/Downside/Cost/Evidence fields), VERDICT (keep if / change if), UNKNOWNS, PROBES.
- **`GraphContext` + `build_graph_context_for_workspace`**: New helper in `workspace.rs` that builds the full unfault-core code graph and extracts hotspots (centrality), hub dependencies, routes, and file paths in one call. Used by `challenge` to inject structural ground truth into the LLM prompt.

### Changed
- **Grouped help output**: `unlost --help` and `unlost` (no args) now display commands organised into four sections — **Memory** (`query`, `trace`, `recall`, `explore`, `challenge`, `brief`), **Workspace** (`init`, `reindex`, `clear`, `where`), **Setup** (`config`, `model`), and **Diagnostics** (`metrics`, `replay`, `inspect`) — instead of a single flat list. Implemented via a custom `help_template` on the root `Cli` struct (clap's `next_help_heading` derive attribute does not apply to struct-variant subcommands).
- **`explore` prompt redesign**: Rewritten to be genuinely open-ended and generative — a thinking partner, not an auditor. The LLM is instructed to use workspace memory as background and constraint, then think freely beyond it. Alternatives are labelled `[memory]` or `[outside]` so the user knows what is grounded and what is creative.
- **`challenge` alternatives format**: Replaced pipe-separated table (unreadable at terminal width) with named card format per alternative. Each card uses circled numbers (①②③④), with dimmed field labels (`Upside:`, `Downside:`, `Cost:`, `Evidence:`) and a blank line between cards for scannability.
- **Higher-signal recall selection**: `unlost recall` now filters low-signal capsules (e.g. replay/ghost extractions), scans a wider recent window to avoid crowd-out, and includes git commit capsules by default so the narrative stays anchored when conversational signal is thin.
- **Recall interventions controls**: Interventions can be hidden from output (`UNLOST_RECALL_HIDE_INTERVENTIONS=1`) and are excluded from the LLM narrative context by default unless explicitly enabled (`UNLOST_RECALL_INTERVENTIONS_IN_CONTEXT=1`).
- **Faster `reindex` rebuilds**: `unlost reindex` now batches embeddings and LanceDB inserts, clears the workspace DB directory in one operation, and shows in-place progress during rebuild.
- **Richer `trace --raw` output**: Raw trace printing now includes capsule source and best-effort references (e.g. `commit:<hash>` / `version:vX.Y.Z`) when available.

### Fixed
- **`render_structured` polish**: Space inserted between circled number and card title text (`①Keep` → `① Keep`). Probe lines changed from dim cyan (`\x1b[2;36m`, nearly invisible on dark backgrounds) to normal cyan (`\x1b[36m`). All prose, card field values, and probe lines now wrap at 80 columns via a new `wrap_ansi_line()` helper that measures visible width by skipping ANSI SGR escape sequences.
- **Safer `reindex` confirmation**: Confirmation prompt now reads a single line from stdin (instead of blocking on EOF), improving behavior in non-interactive environments.
- **Recall rendering clarity**: The narrative output now labels the final section as `Next steps (if any):` to avoid implying that every recap must produce action items.

## [0.7.1] - 2026-02-20

### Added
- **OpenCode skill generation**: `unlost config agent opencode` now automatically creates `.opencode/skills/unlost/SKILL.md` (per-project) or `~/.config/opencode/skills/unlost/SKILL.md` (with `--global`). The skill teaches OpenCode agents what unlost provides and how to use it. If the file already exists, the command prompts before overwriting.
- **Two-tier query guidance in skill**: The generated skill distinguishes fast-path commands (`unlost query --no-llm`, `unlost metrics`) — safe to run proactively with no LLM cost — from LLM-path commands (`unlost query`, `unlost recall`, `unlost brief`) — which should only run on explicit user request.

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

[0.10.0]: https://github.com/unfault/unlost/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/unfault/unlost/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/unfault/unlost/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/unfault/unlost/compare/v0.7.0...v0.7.1
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
