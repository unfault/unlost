use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum EmotionType {
    Joy,
    Anger,
    Frustration,
    Sad,
    Confused,
    Neutral,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum ProviderType {
    Openai,
    Anthropic,
    Opencode,
}

#[derive(Debug, Parser)]
#[command(
    name = "unlost",
    version,
    about = "Local-first code memory (record, init, query)",
    help_template = "\
unlost {version}
{about}

{usage-heading} {usage}

Memory:
  query      Semantic search across recorded capsules
  trace      Trace the causal chain of decisions that led to the current state of a file, symbol, or concept
  recall     Recall the story so far (proactive overview)
  explore    Explore future paths grounded in your workspace memory
  challenge  Pressure-test a past decision or technology choice using your workspace memory
  brief      Get a staff engineer's debrief on this codebase — what matters, what bites, where to start

Workspace:
  init       Seed LanceDB from the current codebase (unfault-core graph)
  reindex    Rebuild LanceDB index from capsules.jsonl
  clear      Delete all generated data for the current workspace
  where      Show where the workspace's files are stored

Setup:
  config     Manage configuration (LLM provider, etc.)
  model      Manage local models (download, etc.)

Diagnostics:
  metrics    Show workspace metrics (local, derived from metrics.jsonl)
  replay     Replay/backfill agent transcripts into unlost
  inspect    Inspect stored capsules for this workspace

Options:
{options}
"
)]
pub struct Cli {
    /// Logging level for unlost (overrides RUST_LOG when set)
    #[arg(long, global = true, value_enum, alias = "log-level")]
    pub log: Option<LogLevel>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn as_tracing_str(self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum OutputFormat {
    /// Default terminal-friendly output (ANSI colors)
    Ansi,
    /// No ANSI colors (useful for piping)
    Plain,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Global recorder that multiplexes workspaces via base URL
    #[command(hide = true)]
    Serve {
        /// Bind address. Accepts either `port` or `ip:port`.
        /// Examples: `3000`, `127.0.0.1:3000`.
        #[arg(long, default_value = "127.0.0.1:3000")]
        bind: String,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Record live LLM conversations (captures and summarizes)
    #[command(alias = "proxy", hide = true)]
    Record {
        /// Bind address. Accepts either `port` or `ip:port`.
        /// Examples: `3000`, `0.0.0.0:3000`.
        #[arg(long, default_value = "3000")]
        bind: String,

        /// Upstream host (or set UNLOST_UPSTREAM_HOST)
        #[arg(long, env = "UNLOST_UPSTREAM_HOST")]
        upstream_host: String,

        /// Upstream port (or set UNLOST_UPSTREAM_PORT)
        #[arg(long, env = "UNLOST_UPSTREAM_PORT", default_value_t = 443)]
        upstream_port: u16,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Semantic search across recorded capsules
    Query {
        /// Query text
        query: Vec<String>,

        /// Max results
        #[arg(long, default_value_t = 5)]
        limit: usize,

        /// Filter results to a symbol
        #[arg(long)]
        symbol: Option<String>,

        /// Filter by user emotion (joy, anger, frustration, sad, confused, neutral)
        #[arg(long, value_enum)]
        emotion: Option<EmotionType>,

        /// Filter by upstream provider (openai, anthropic, opencode)
        #[arg(long, value_enum)]
        provider: Option<ProviderType>,

        /// Filter to capsules after this time (RFC3339 or relative: 1h, 1d, 1w, 1m, 1y)
        #[arg(long)]
        since: Option<String>,

        /// Filter to capsules before this time (RFC3339 or relative: 1h, 1d, 1w, 1m, 1y)
        #[arg(long)]
        until: Option<String>,

        /// Disable LLM narrative (prints raw matches)
        #[arg(long, default_value_t = false)]
        no_llm: bool,

        /// LLM model to use for query narrative
        #[arg(long)]
        llm_model: Option<String>,

        /// Print raw match facts after the narrative
        #[arg(long, default_value_t = false)]
        facts: bool,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Shortcut for `--output plain`
        #[arg(long, default_value_t = false)]
        plain: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,

        /// Path to capsules JSONL (fallback mode only). Defaults to the workspace's JSONL.
        #[arg(long, default_value = "")]
        file: String,
    },

    /// Trace the causal chain of decisions that led to the current state of a file, symbol, or concept
    Trace {
        /// File path, symbol name, or free-text question (e.g. "why is the timeout 30s?")
        target: Vec<String>,

        /// Max seed capsules from initial semantic search
        #[arg(long, default_value_t = 5)]
        seeds: usize,

        /// Max capsules per symbol fan-out
        #[arg(long, default_value_t = 8)]
        fan_out: usize,

        /// Similarity distance threshold (0.0–1.0); capsules above this are dropped
        #[arg(long, default_value_t = 0.65)]
        threshold: f32,

        /// Filter to capsules after this time (RFC3339 or relative: 1h, 1d, 1w, 1M, 1y)
        #[arg(long)]
        since: Option<String>,

        /// Filter to capsules before this time (RFC3339 or relative: 1h, 1d, 1w, 1M, 1y)
        #[arg(long)]
        until: Option<String>,

        /// LLM model to use for trace narrative
        #[arg(long)]
        llm_model: Option<String>,

        /// Disable LLM narrative (prints raw chain)
        #[arg(long, default_value_t = false)]
        no_llm: bool,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Shortcut for `--output plain`
        #[arg(long, default_value_t = false)]
        plain: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Get a staff engineer's debrief on this codebase — what matters, what bites, where to start
    Brief {
        /// Optional scope: file path, symbol, or concept to focus the brief on
        target: Vec<String>,

        /// LLM model to use for the brief
        #[arg(long)]
        llm_model: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Shortcut for `--output plain`
        #[arg(long, default_value_t = false)]
        plain: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Recall the story so far (proactive overview)
    Recall {
        /// Optional scope (file path or symbol/function name)
        target: Vec<String>,

        /// Max capsules to use
        #[arg(long, default_value_t = 40)]
        limit: usize,

        /// Filter by user emotion (joy, anger, frustration, sad, confused, neutral)
        #[arg(long, value_enum)]
        emotion: Option<EmotionType>,

        /// Filter by upstream provider (openai, anthropic, opencode)
        #[arg(long, value_enum)]
        provider: Option<ProviderType>,

        /// Filter to capsules after this time (RFC3339 or relative: 1h, 1d, 1w, 1m, 1y)
        #[arg(long)]
        since: Option<String>,

        /// Filter to capsules before this time (RFC3339 or relative: 1h, 1d, 1w, 1m, 1y)
        #[arg(long)]
        until: Option<String>,

        /// LLM model to use for recall narrative
        #[arg(long)]
        llm_model: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Shortcut for `--output plain`
        #[arg(long, default_value_t = false)]
        plain: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Explore future paths grounded in your workspace memory
    Explore {
        /// Scenario or goal to explore (e.g. "should we keep lancedb or move to sqlite+fts?")
        query: Vec<String>,

        /// LLM model to use for the exploration narrative
        #[arg(long)]
        llm_model: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Shortcut for `--output plain`
        #[arg(long, default_value_t = false)]
        plain: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Pressure-test a past decision or technology choice using your workspace memory
    Challenge {
        /// Decision or technology to challenge (e.g. "lancedb" or "was using fastembed the right call?")
        target: Vec<String>,

        /// Show full analysis: adds UNKNOWNS and PROBES sections (default: concise)
        #[arg(long, default_value_t = false)]
        deep: bool,

        /// LLM model to use for the challenge narrative
        #[arg(long)]
        llm_model: Option<String>,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Ansi)]
        output: OutputFormat,

        /// Shortcut for `--output plain`
        #[arg(long, default_value_t = false)]
        plain: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Show workspace metrics (local, derived from metrics.jsonl)
    Metrics {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,
    },

    /// Replay/backfill agent transcripts into unlost
    Replay {
        #[command(subcommand)]
        command: ReplayCommand,
    },

    /// Inspect stored capsules for this workspace
    Inspect {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Max rows to print
        #[arg(long, default_value_t = 20)]
        limit: usize,

        /// Filter by user emotion (joy, anger, frustration, sad, confused, neutral)
        #[arg(long, value_enum)]
        emotion: Option<EmotionType>,

        /// Filter by upstream provider (openai, anthropic, opencode)
        #[arg(long, value_enum)]
        provider: Option<ProviderType>,

        /// Filter to capsules after this time (RFC3339 or relative: 1h, 1d, 1w, 1m, 1y)
        #[arg(long)]
        since: Option<String>,

        /// Filter to capsules before this time (RFC3339 or relative: 1h, 1d, 1w, 1m, 1y)
        #[arg(long)]
        until: Option<String>,

        /// Optional Lance filter expression (DataFusion SQL)
        #[arg(long)]
        filter: Option<String>,
    },

    /// Seed LanceDB from the current codebase (unfault-core graph)
    Init {
        /// Root directory to scan
        #[arg(long, default_value = ".")]
        path: String,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,

        /// Max number of capsules to insert
        #[arg(long, default_value_t = 120)]
        max_capsules: usize,

        /// Disable LLM summaries for init
        #[arg(long, default_value_t = false)]
        no_llm: bool,

        /// Include recent git history (commit subjects + touched files) when available
        #[arg(long, default_value_t = true)]
        git_history: bool,

        /// Max commits to consider for git history (bounded)
        #[arg(long, default_value_t = 50)]
        git_commits: usize,

        /// Limit git history to a subdirectory (relative to repo root). Defaults to --path.
        #[arg(long)]
        git_path: Option<String>,

        /// LLM model to use for init summaries
        #[arg(long)]
        llm_model: Option<String>,

        /// Max LLM-generated capsules
        #[arg(long, default_value_t = 12)]
        llm_max_capsules: usize,
    },

    /// Manage local models (download, etc.)
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },

    /// Manage configuration (LLM provider, etc.)
    #[command(alias = "configure")]
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    /// Delete all generated data for the current workspace
    Clear {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Rebuild LanceDB index from capsules.jsonl
    Reindex {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Test emotion detection on a string (developer tool)
    #[command(hide = true)]
    Emotion {
        /// Text to classify
        text: String,
    },

    /// Agent integration shims (OpenCode, Claude Code, etc.)
    #[command(hide = true)]
    Shim {
        #[command(subcommand)]
        command: ShimCommand,
    },

    /// Show where the workspace's files are stored
    Where {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum ShimCommand {
    /// Run the OpenCode stdio shim (JSON-RPC over stdin/stdout)
    Opencode {
        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,

        /// Disable LLM extraction (fast, zero cost)
        #[arg(long, default_value_t = false)]
        no_extraction: bool,
    },

    /// Run the Claude hooks shim (reads hook JSON from stdin)
    #[command(alias = "claudecode")]
    Claude {
        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Replay/backfill agent transcripts into unlost
    Replay {
        #[command(subcommand)]
        command: ReplayCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum ReplayCommand {
    /// Replay a Claude transcript file into the current workspace
    #[command(alias = "claudecode")]
    Claude {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Claude transcript .jsonl file or directory path
        #[arg(long)]
        transcript_path: String,

        /// Claude session id (defaults to transcript filename stem)
        #[arg(long)]
        session_id: Option<String>,

        /// Force replay from beginning and overwrite cursor to EOF
        #[arg(long, default_value_t = true)]
        from_start: bool,

        /// Skip turns already replayed (best-effort)
        #[arg(long, default_value_t = true)]
        dedupe: bool,

        /// Disable LLM extraction (fast, zero cost)
        #[arg(long, default_value_t = false)]
        no_extraction: bool,

        /// Enable full LLM extraction for every turn (slow, expensive)
        #[arg(long, default_value_t = false)]
        full_extraction: bool,

        /// Clear existing database and replayed-tracking for this workspace before starting
        #[arg(long, default_value_t = false)]
        clear: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,

        /// Ground replayed turns with actual git logs (find corresponding commits)
        #[arg(long, default_value_t = false)]
        git_grounding: bool,
    },

    /// Ingest git commit history as capsules into the current workspace
    Git {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Max commits to ingest (most recent first, deduplicates on re-run)
        #[arg(long, default_value_t = 500)]
        max_commits: usize,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,
    },

    /// Replay OpenCode messages from disk storage into the current workspace
    Opencode {
        /// Workspace path (defaults to current directory)
        #[arg(long, default_value = ".")]
        path: String,

        /// Skip messages already replayed (best-effort)
        #[arg(long, default_value_t = true)]
        dedupe: bool,

        /// Disable LLM extraction (fast, zero cost)
        #[arg(long, default_value_t = false)]
        no_extraction: bool,

        /// Enable full LLM extraction for every turn (slow, expensive)
        #[arg(long, default_value_t = false)]
        full_extraction: bool,

        /// Clear existing database and replayed-tracking for this workspace before starting
        #[arg(long, default_value_t = false)]
        clear: bool,

        /// Embedding model (fastembed). Default: BAAI/bge-small-en-v1.5
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Embedding cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        embed_cache_dir: Option<String>,

        /// Ground replayed turns with actual git logs (find corresponding commits)
        #[arg(long, default_value_t = false)]
        git_grounding: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Manage LLM configuration for init/query narratives
    Llm {
        #[command(subcommand)]
        command: LlmCommand,
    },

    /// Configure an agent workspace to talk to unlost
    Agent {
        #[command(subcommand)]
        command: AgentCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum AgentCommand {
    /// Configure OpenCode to load the unlost plugin (stdio shim)
    Opencode {
        /// Workspace path (defaults to current directory; uses git toplevel)
        #[arg(long, default_value = ".")]
        path: String,

        /// npm package name to add
        #[arg(long, default_value = "@unfault/unlost-opencode")]
        plugin: String,

        /// Install globally in ~/.config/opencode/opencode.json instead of per-project
        #[arg(long)]
        global: bool,
    },

    /// Configure Claude hooks to use unlost
    #[command(alias = "claudecode")]
    Claude {
        /// Workspace path (defaults to current directory; uses git toplevel)
        #[arg(long, default_value = ".")]
        path: String,

        /// Install globally in ~/.claude/settings.json instead of per-project
        #[arg(long)]
        global: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum LlmCommand {
    /// Configure OpenAI as LLM provider
    Openai {
        /// OpenAI API key
        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: String,

        /// Default model to use
        #[arg(long, default_value = "gpt-4o-mini")]
        model: String,

        /// Optional base URL override (OpenAI-compatible)
        #[arg(long)]
        base_url: Option<String>,
    },

    /// Configure Anthropic as LLM provider
    Anthropic {
        /// Anthropic API key
        #[arg(long, env = "ANTHROPIC_API_KEY")]
        api_key: String,

        /// Default model to use
        #[arg(long, default_value = "claude-3-5-sonnet-20241022")]
        model: String,

        /// Optional base URL override
        #[arg(long)]
        base_url: Option<String>,
    },

    /// Configure local Ollama as LLM provider (OpenAI-compatible endpoint)
    Ollama {
        /// Ollama model name (e.g. llama3.2:3b)
        #[arg(long)]
        model: String,

        /// OpenAI-compatible base URL (default: http://127.0.0.1:11434/v1)
        #[arg(long, default_value = "http://127.0.0.1:11434/v1")]
        base_url: String,
    },

    /// Configure a custom OpenAI-compatible endpoint
    Custom {
        /// Base URL (e.g. https://my-endpoint/v1)
        #[arg(long)]
        base_url: String,

        /// API key (if required)
        #[arg(long)]
        api_key: Option<String>,

        /// Default model to use
        #[arg(long)]
        model: String,
    },

    /// Show current LLM configuration
    Show,

    /// Remove LLM configuration
    Remove,
}

#[derive(Debug, Subcommand)]
pub enum ModelCommand {
    /// Download embedding model files into the local cache
    Download {
        /// Embedding model (fastembed)
        #[arg(long, default_value = crate::constants::DEFAULT_EMBED_MODEL)]
        embed_model: String,

        /// Cache directory (defaults to XDG data dir)
        #[arg(long, env = "UNLOST_EMBED_CACHE_DIR")]
        cache_dir: Option<String>,

        /// Delete cache dir before downloading
        #[arg(long, default_value_t = false)]
        force: bool,
    },
}

pub fn parse_bind(s: &str) -> anyhow::Result<SocketAddr> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("bind cannot be empty");
    }

    // `:3000`
    if let Some(port_str) = s.strip_prefix(':') {
        let port: u16 = port_str.parse().context("invalid port")?;
        return Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port));
    }

    // `3000`
    if s.chars().all(|c| c.is_ascii_digit()) {
        let port: u16 = s.parse().context("invalid port")?;
        return Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port));
    }

    // `ip:port`
    s.parse().context("invalid bind address")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_as_tracing_str() {
        assert_eq!(LogLevel::Error.as_tracing_str(), "error");
        assert_eq!(LogLevel::Warn.as_tracing_str(), "warn");
        assert_eq!(LogLevel::Info.as_tracing_str(), "info");
        assert_eq!(LogLevel::Debug.as_tracing_str(), "debug");
        assert_eq!(LogLevel::Trace.as_tracing_str(), "trace");
    }

    #[test]
    fn test_parse_bind() {
        // Test port-only formats
        let addr = parse_bind("3000").unwrap();
        assert_eq!(addr.port(), 3000);
        assert_eq!(addr.ip(), std::net::IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

        let addr = parse_bind(":3000").unwrap();
        assert_eq!(addr.port(), 3000);
        assert_eq!(addr.ip(), std::net::IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

        // Test IP:port format
        let addr = parse_bind("127.0.0.1:3000").unwrap();
        assert_eq!(addr.port(), 3000);
        assert_eq!(addr.ip(), std::net::IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));

        let addr = parse_bind("0.0.0.0:8080").unwrap();
        assert_eq!(addr.port(), 8080);
        assert_eq!(addr.ip(), std::net::IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

        // Test IPv6
        let addr = parse_bind("[::1]:3000").unwrap();
        assert_eq!(addr.port(), 3000);

        // Test error cases
        assert!(parse_bind("").is_err());
        assert!(parse_bind("   ").is_err());
        assert!(parse_bind("invalid").is_err());
        assert!(parse_bind("127.0.0.1").is_err());
        assert!(parse_bind("127.0.0.1:invalid").is_err());
        assert!(parse_bind("99999").is_err()); // Port out of range
    }

    #[test]
    fn test_output_format_equality() {
        assert_eq!(OutputFormat::Ansi, OutputFormat::Ansi);
        assert_eq!(OutputFormat::Plain, OutputFormat::Plain);
        assert_ne!(OutputFormat::Ansi, OutputFormat::Plain);
    }
}
