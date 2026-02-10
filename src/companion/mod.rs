//! Companion module for agent integrations.
//!
//! This module provides the shared "flow" (check + record_turn pipeline) and
//! platform-specific shims that adapt external protocols to the internal events.
//!
//! # Architecture
//!
//! - `flow`: Core business logic for friction checking and conversation recording.
//!   Defines `CheckEvent`, `RecordTurnEvent`, and `Flow` which owns the chunking,
//!   background flush worker, and all heavy processing (LLM extract, embed, DB insert).
//!
//! - `shims`: Platform-specific adapters that translate external protocols into
//!   flow events. Each shim is responsible only for I/O and mapping.
//!   - `opencode_stdio`: JSON-RPC over stdin/stdout for OpenCode plugins.
//!   - `claude`: Claude hook JSON + transcript parsing.
//!   - (future) `codex`: OpenAI Codex integration.

pub(crate) mod flow;
pub(crate) mod shims;
