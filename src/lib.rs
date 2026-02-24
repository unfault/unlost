pub mod analysis;
pub mod changelog;
pub mod cli;
pub mod commands;
pub mod companion;
pub mod config;
pub mod constants;
pub mod embed;
pub mod emotion;
pub mod git;
pub mod governor;
pub mod http_proxy;
pub mod llm;
pub mod logging;
pub mod metrics;
pub mod narrative;
pub mod net;
pub mod recording;
pub mod storage;
pub mod storage_checkpoint;
pub mod types;
pub mod util;
pub mod workspace;

#[cfg(test)]
pub mod test_support;

pub use crate::llm::llm_extract;
pub use crate::types::IntentCapsule;
pub use crate::types::{
    CapsuleHit, InitCapsulesOutput, QueryNarrativeOutput, ResponseMeta, TrajectoryState,
};
pub use crate::workspace::{WorkspacePaths, now_ms, unlost_data_root, unlost_workspace_dir};
