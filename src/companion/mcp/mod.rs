//! MCP (Model Context Protocol) server for unlost.
//!
//! Exposes 7 task-shaped tools for agents to query workspace memory without
//! going through the human-facing CLI (ANSI prose, LLM narration, etc.).
//!
//! All tools run on the no-LLM fast path. The calling agent already has a model;
//! we return structured, citable evidence and let the agent narrate.
//!
//! # Tools
//!
//! | Tool | Purpose | Write? |
//! |---|---|---|
//! | `unlost_recall` | Workspace memory lookup by query/file/symbol | No |
//! | `unlost_trace_decision` | Causal chain for a target | No |
//! | `unlost_challenge` | Prior rationale + alternatives against a proposal | No |
//! | `unlost_thread` | Cross-workspace topic history | No |
//! | `unlost_orient` | Recent touches + drift signal for files/symbols | No |
//! | `unlost_note` | Record a deliberate decision (opt-in) | Yes |
//! | `unlost_capsule_get` | Fetch full capsule by id for citation | No |

pub mod server;
pub mod types;

pub use server::UnlostMcpServer;
