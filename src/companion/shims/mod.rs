//! Platform-specific shims for agent integrations.
//!
//! Each shim adapts an external protocol to the internal flow events.

pub mod claude;
pub mod copilot;
pub mod opencode;
pub mod opencode_stdio;
