//! Platform-specific shims for agent integrations.
//!
//! Each shim adapts an external protocol to the internal flow events.

pub(crate) mod claudecode;
pub(crate) mod opencode_stdio;
