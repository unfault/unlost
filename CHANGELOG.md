# Changelog

## [Unreleased]

### Added

- **Source pointers**: Every capsule now carries an optional `source_pointer` field — an opaque URI pointing back to the system-of-record for that turn. Schemes: `claude+jsonl://...#turn=<uuid>`, `opencode+message://<session>/<msg_id>`, `copilot+events://...#offset=<bytes>`, `git+commit://<repo>#<sha>`, `git+tag://<repo>#<name>`, `changelog+version://<path>#<version>`. Stored as a new nullable column in LanceDB (`source_pointer`, additive migration following the `te_*` precedent), written to JSONL, and re-hydrated by `reindex`. All five shim paths populate the field: Claude hooks, OpenCode stdio plugin, OpenCode replay, Copilot events, git commit/tag ingestion, and changelog ingestion. The `query` and `inspect` commands print `source_ref` (human label) and `source_uri` (raw URI) footers for hits that carry a pointer. `resolve_source_label(uri)` in `workspace.rs` provides per-scheme rendering with 9 unit tests.
- **`recurrence_signal` channel**: New `SymptomChannels` field that measures how strongly the current user turn matches a dormant capsule not seen in the recent window. Drives the resurfacing basin in `TrajectoryController`. Stored as a per-turn EMA value in `SymptomChannels`; does not participate in the aggregate trajectory intensity.
- **Resurfacing basin** (`TrajectoryController::update_with_candidates`): When a dormant capsule scores `similarity × structural_weight ≥ 0.78` and the session has not yet had a standalone resurfacing, the controller emits a SYSTEM NOTE with the prior decision, its rationale, source URI, and — for cross-workspace matches — an `"in <project>"` clause. Modifier mode appends to an existing basin note instead of firing standalone. The old `update()` delegates to the new method with empty candidates for backward compatibility. Rename `check_friction → check_turn` throughout (`flow.rs`, all shims, comments) to reflect the expanded scope.
- **`resurfaced.rs`** — Global cooldown ledger at `~/.local/share/unlost/resurfaced.jsonl`, shared across all workspaces. `record()` appends a `(capsule_id, ts_ms)` entry; `load()` reads both the legacy per-workspace file and the global ledger; `is_cooling()` enforces a 30-day window. UUID-based capsule IDs make cross-workspace collisions safe.
- **`query_capsules_cross_workspace`** in `storage.rs`: Fans out an ANN query across the current workspace and every registered peer workspace. Tags each hit with `origin_workspace_id`. Merges results by distance ascending, capped to `total_limit`. Per-workspace failures are debug-logged and skipped. Used by both the per-turn recurrence channel in `flow.rs` and the new `thread` command.
- **Cross-workspace workspace labels**: `workspace_label(info)` and `workspace_label_by_id(id)` in `workspace.rs` derive human-readable project names from workspace root basenames. Used in SYSTEM NOTEs and the `thread` map.
- **`list_other_workspaces`** in `workspace.rs`: Returns all registered workspaces except the current one. Used by `query_capsules_cross_workspace` for the fan-out.
- **`record_resurfacing_emitted`** in `metrics.rs`: Logs a `ResurfacingEmitted` event to `metrics.jsonl` with `workspace_id`, `agent_session_id`, `matched_capsule_id`, `similarity`, `mode` (`"standalone"`/`"modifier"`), and `candidate_age_days`. Enables future success-metric computation (fraction of surfacings where the agent referenced the capsule).
- **`CapsuleHit.origin_workspace_id`**: New optional field on `CapsuleHit`. Set by cross-workspace retrieval to identify which workspace a hit came from; `None` for single-workspace queries. Used by the recurrence channel and the `thread` command renderer.
- **`unlost thread`**: New command that maps when a topic was explored over time, across all registered projects. Results are sorted oldest-to-newest, grouped by 4-hour session clusters; dormancy gaps >7d are rendered as `· · · N days dormant · · ·` markers. Each entry shows category, decision, rationale (first sentence), failure mode, first next step, up to 3 symbols, and source pointer. Optional LLM synthesis (default on) describes the intellectual arc — where the thinking started, how the framing shifted, where it currently stands — without framing any moment as "unresolved". Cross-workspace retrieval is the default. Supports `--since`, `--no-llm`, `--limit`, `--output plain`, `--llm-model`.
- **`docs/index.html`**: Added "The context you didn't know you needed" section describing the proactive recurrence channel surfacing dormant capsules before the developer asks.

### Changed

- **`check_friction` → `check_turn`**: Renamed throughout `flow.rs`, all shim call sites, and inline comments to reflect that this hook now does far more than friction detection (per-turn ANN retrieval, recurrence scoring, resurfacing injection).
- **`CopsuleHit` and `ResponseMeta` struct fields**: Both structs gained new nullable fields (`origin_workspace_id`, `source_pointer`). All construction sites — `recall.rs`, `brief.rs`, `init.rs`, `reindex.rs`, `recording.rs`, `resurfaced.rs` tests — updated with `None` defaults, keeping backward compat.

## [0.13.1] - 2026-03-03

### Fixed

- Previous release broke because release was created by agent as immutable
  before the workflow existed

## [0.13.0] - 2026-03-03

### Added
- **`TurnEval`**: Per-turn evaluation metadata computed on-the-fly at flush time with zero LLM calls. Each capsule now carries 12 agent-tuning (`tune`) dimensions — persisted governor `SymptomChannels` previously discarded after friction decisions — plus 5 developer coaching (`coach`) dimensions: `clarity`, `context_freshness` (cache ratio + frustration slope, captures compaction signal), `verification_rigor`, `decision_progress`, and `scope_discipline`. Behavioral flags (`session_heavy`, `session_too_long`, `retry_loop`, `blind_acceptance`, etc.) derived from thresholds on both dimensions. Stored in LanceDB (`te_*` columns with schema evolution), JSONL capsule log, and metrics. Displayed in `unlost inspect`.
- **`TurnEval.cost_acceleration`**: New coach dimension (0–1) measuring whether token spend is accelerating without corresponding `decision_progress`. Computed as the relative growth of `tokens_input` over a 3-turn rolling window, weighted by lack of progress. Emits `cost_spike` flag when > 0.5.
- **`unlost reflect`**: New command generating a structured coaching/diagnostics narrative from per-turn `TurnEval` telemetry — no raw transcript required. Three modes: `--mode coach` (developer collaboration habits), `--mode tune` (agent drift and failure patterns), `--mode both`. Supports `--session <id>` and `--since <duration>` scoping.
  - Every output opens with **NEXT ACTIONS** — 3–5 scannable bold imperatives before the full analysis.
  - `tune`/`both` modes include **SKILL ASSESSMENT**: audits installed agent skills (`.opencode/skills/`, `.claude/skills/`, etc.) against turn data (`helped / hurt / neutral`), then lists behavioural gaps to fill with "Look for skills that…" guidance derived from observed patterns. Infrastructure/observer skills (unlost, git-workflow, graph tools) are automatically excluded from the audit.
  - Rich ANSI renderer: mode-coloured section headers, score colouring (green/yellow/red), `(low confidence)` markers, dimmed turn references, `→` NEXT ACTIONS bullets, `◆` skill assessment bullets.
- **Outcome backfill**: At each checkpoint, `te_outcome_hint` (`progressed`/`stalled`/`regressed`/`unclear`) is retroactively set via deterministic lookahead heuristics and written back via LanceDB `UPDATE`.
- **`TurnEval` in all retrieval paths**: `query_capsules_lancedb`, `scan_capsules_lancedb`, and the fan-out path all populate `turn_eval` on `CapsuleHit` via a shared `read_turn_eval` helper.
- **`TurnEval` backfill on reindex**: `unlost reindex` automatically populates `TurnEval` for all capsule history. Post-v0.13 capsules restore full data from JSONL; pre-v0.13 capsules get coach dimensions computed from content + a rolling 8-turn history window (`v1-reindex` version marker).
- **Extended `verification_rigor` detection**: Static analysis and type-checker outputs now count as verification evidence: `clippy`, `mypy`, `tsc`, `pyright`, `eslint`, `ruff`, `biome`, `golangci`, plus failure patterns `type error`, `type mismatch`, `lint error`, `E0` (Rust), `TS` (TypeScript).

### Changed
- **`--mode diagnose` → `--mode tune`**: The agent-facing reflect persona is now `tune` throughout — CLI, prompts, inspect output, and comments — to make clear it targets agent behaviour improvement rather than generic diagnosis.

## [0.12.0] - 2026-02-27

### Added
- **GitHub Copilot CLI integration**: `unlost shim copilot` and `unlost config agent copilot` provide hooks-based integration with GitHub Copilot CLI. Session transcripts are read directly from `~/.copilot/session-state/<uuid>/events.jsonl`, giving access to full user and assistant text without synthesis. Session discovery uses `workspace.yaml` `created_at` proximity and `summary` cross-check at `sessionStart`, and `updated_at` proximity at `sessionEnd`. Installs `sessionStart`, `userPromptSubmitted`, and `sessionEnd` hooks via `.github/hooks/unlost.json`, and writes a Copilot-compatible skill to `.github/copilot/skills/unlost/`.

### Changed
- **`docs/index.html`**: Restructured landing page to prioritize Context Ownership and Memory over control. "How It Works" now follows the Memory Lifecycle (Record → Extract → Ground). Added "One Memory. Many Lenses" section to showcase `trace`, `challenge`, and `explore` as different views on the same grounded context. Moved "Cognitive Mirror" technical details to a dedicated deep-dive section at the bottom.

## [0.11.2] - 2026-02-26

### Fixed
- **Friction detection false positives**: Three targeted changes reduce spurious de-escalation interventions during productive back-and-forth discussions. (1) The `anger_streak` fast path now requires trajectory intensity >= Watch threshold (0.5) in addition to 2+ consecutive negative turns — pure emotion-classification noise can no longer trigger an intervention without corroborating behavioral evidence. (2) The go_emotions `disapproval` label is excluded from the anger streak counter, since it maps to intellectual disagreement rather than user upset; it still contributes to trajectory intensity via valence. (3) The heuristic override that mapped `neutral + 1 frustration signal → disapproval` is removed — a single matched keyword (e.g. `"broken"` in a technical description) is too weak a signal to override a neutral classification.

### Changed
- **`README`**: Reframed mission around ownership vs. authorship. The README now leads with the human engineer's perspective — accountability in incidents, reviews, and architecture decisions — rather than agent failure modes. Removed the babysitting tax framing and failure mode table from the lead; commands are now grouped by the moment you reach for them (understanding, deciding, handing off).

## [0.11.1] - 2026-02-25

### Changed
- **README**: Restructured for better readability, moving installation to top and collapsing technical details.


### Fixed
- **`unlost checkpoint` output**: Fixed narrative output not wrapping at 80 columns. Now uses `render_narrative` to ensure proper formatting and ANSI coloring.
- **Recall/inspect filtering bug**: Fixed an issue where `unlost recall` and `unlost inspect` with filters (e.g. `--emotion joy`) would return no results. The optimization to fetch only the most recent rows was calculating the offset based on the total row count instead of the filtered row count, often skipping all matching rows.

## [0.11.0] - 2026-02-24

### Added
- **`unlost config agent`**: Automatically install the `unlost-pr-comment` command when configuring the agent.
- **`/unlost-walkthrough` skill**: Installs a walkthrough skill for both OpenCode and Claude Code that guides users through recent changes step-by-step (with VSCode `code --goto` navigation).
- **"Under the Hood" section**: Added to both `README.md` and `docs/index.html` — a grouped inventory of every technique, algorithm, and strategy Unlost uses (trajectory sensing, emotion/NLP, retrieval/memory, storage/infrastructure), with "where it's used" context on the landing page.

### Changed
- **`unlost pr-comment` dual-audience comment**: The comment now serves both the author (staying close to code written by AI agents) and the reviewer. Voice changed from "you" to "we". New sections: "What we were navigating" (shared tradeoff framing with emotional signal woven in), "Ripple effects" (functional knock-on effects across commands/features, not just code imports), "Left open" (deferred decisions, open questions, unresolved next_steps from capsules), and "Re-read this" (1-2 linked file:function pointers to non-obvious logic). File references are now clickable GitHub blob links built from the head SHA and repo coordinates. "How To Verify" removed. Em-dashes banned from output. A blockquote hook at the top of every comment shows the decision count and explains what unlost does; if no decisions were found, it says so plainly and suggests a replay command. Fixed: `headRepositoryOwner` is now fetched as a top-level field (the nested `headRepository.owner.login` path was always empty).

### Fixed
- **`unlost inspect` capsule order**: Capsules are now displayed oldest-first (newest at end) instead of newest-first.
- **LanceDB timestamp filter crash**: Avoids a DataFusion interval planning error (`lhs:Null, rhs:Int64`) when applying `ts_ms` range filters on mixed-schema datasets; falls back to client-side time filtering and prints a repair command (`unlost reindex`).

## [0.10.0] - 2026-02-24

### Added
- **`unlost pr-comment`**: New command that posts a staff-engineer-style context comment on a
  GitHub PR (requires `gh` CLI). The comment explains what the changed code is, where the
  decisions that shaped it come from (drawing on recorded capsules and git history), and flags
  high-dependency files that may have wider impact ("Worth noting" section). Accepts a PR URL or
  number, an optional `--session-id` to scope the trace, and an optional `--from-commit` for
  diff bounds.
- **Stealth PR comment — OpenCode shim**: When the agent creates a GitHub PR via `gh pr create`
  (detected by scanning `bash` tool-call outputs for a GitHub PR URL), the OpenCode stdio shim
  automatically spawns `unlost pr-comment` in the background without blocking the agent.
- **Stealth PR comment — Claude shim**: Same stealth detection for the Claude Stop hook: assistant
  texts from each batch are scanned for a GitHub PR URL and `unlost pr-comment` is spawned if found.
- **`unlost trace --session-id`**: New flag to restrict the causal chain to capsules from a
  specific agent session, enabling per-session archaeology.
- **`unlost trace --from-commit` / `--to-commit`**: New flags to scope the trace to a commit
  range. Commit refs (branch names, SHAs, `HEAD`, etc.) are resolved to timestamps via
  `git log -1 --format=%ct`, then used as `since`/`until` filters on the capsule store.

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

[0.11.2]: https://github.com/unfault/unlost/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/unfault/unlost/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/unfault/unlost/compare/v0.10.0...v0.11.0
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
