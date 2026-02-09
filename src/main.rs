use clap::{CommandFactory, Parser};

mod analysis;
mod cli;
mod commands;
mod companion;
mod config;
mod constants;
mod embed;
mod emotion;
mod governor;
mod http_proxy;
mod llm;
mod logging;
mod metrics;
mod narrative;
mod net;
mod recording;
mod storage;
mod types;
mod util;
mod workspace;

#[cfg(test)]
mod test_support;

pub(crate) use crate::llm::llm_extract;
pub(crate) use crate::types::{CapsuleHit, InitCapsulesOutput, QueryNarrativeOutput, ResponseMeta};
pub use crate::types::IntentCapsule;
pub(crate) use crate::workspace::{now_ms, unlost_data_root, unlost_workspace_dir, WorkspacePaths};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = crate::cli::Cli::parse();

    if cli.command.is_none() {
        crate::cli::Cli::command().print_help()?;
        println!(
            "\n\nTry:\n- unlost config agent opencode --path .\n- unlost config agent claudecode --global\n- unlost config llm anthropic --model claude-3-5-sonnet-20241022\n- unlost init --path .\n- unlost recall\n- unlost query \"what are the routes available?\"\n"
        );
        return Ok(());
    }

    // Default to "info" for shim (we want to see friction/emotion logs),
    // "warn" for everything else
    let is_shim = matches!(&cli.command, Some(crate::cli::Command::Shim { .. }));
    let default_level = if is_shim { "info" } else { "warn" };
    let log_level = cli
        .log
        .map(|l| l.as_tracing_str().to_string())
        .unwrap_or_else(|| default_level.to_string());
    let filter = crate::logging::create_filter(&log_level);

    // Determine logging mode based on command type:
    // - Shim commands use file-only (stdout/stderr used for protocol)
    // - Long-running server commands log to both file and stderr
    // - Short-lived commands just use stderr
    let is_long_running = matches!(
        &cli.command,
        Some(crate::cli::Command::Serve { .. }) | Some(crate::cli::Command::Record { .. })
    );

    // Keep the guard alive for the duration of main()
    let _log_guard = if is_shim {
        Some(crate::logging::init_logging_file_only(filter))
    } else if is_long_running {
        Some(crate::logging::init_logging(filter))
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
        None
    };

    // rustls 0.23 requires selecting a process-level CryptoProvider.
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    match cli.command.unwrap() {
        crate::cli::Command::Serve {
            bind,
            embed_model,
            embed_cache_dir,
        } => {
            crate::commands::serve::run(bind, embed_model, embed_cache_dir).await?;
        }
        crate::cli::Command::Record {
            bind,
            upstream_host,
            upstream_port,
            embed_model,
            embed_cache_dir,
        } => {
            crate::commands::record::run(
                bind,
                upstream_host,
                upstream_port,
                embed_model,
                embed_cache_dir,
            )
            .await?;
        }
        crate::cli::Command::Query {
            query,
            limit,
            symbol,
            emotion,
            provider,
            since,
            until,
            no_llm,
            llm_model,
            facts,
            output,
            plain,
            embed_model,
            embed_cache_dir,
            file,
        } => {
            let output = if plain {
                crate::cli::OutputFormat::Plain
            } else {
                output
            };
            crate::commands::query::run(
                query,
                limit,
                symbol,
                emotion,
                provider,
                since,
                until,
                no_llm,
                llm_model,
                facts,
                output,
                embed_model,
                embed_cache_dir,
                file,
            )
            .await?;
        }
        crate::cli::Command::Recall {
            target,
            limit,
            emotion,
            provider,
            since,
            until,
            llm_model,
            output,
            plain,
            embed_model,
            embed_cache_dir,
        } => {
            let output = if plain {
                crate::cli::OutputFormat::Plain
            } else {
                output
            };
            crate::commands::recall::run(
                target,
                limit,
                emotion,
                provider,
                since,
                until,
                llm_model,
                output,
                embed_model,
                embed_cache_dir,
            )
            .await?;
        }
        crate::cli::Command::Metrics { path } => {
            crate::commands::metrics::run(path)?;
        }
        crate::cli::Command::Inspect { path, limit, emotion, provider, since, until, filter } => {
            crate::commands::inspect::run(path, limit, emotion, provider, since, until, filter).await?;
        }
        crate::cli::Command::Init {
            path,
            embed_model,
            embed_cache_dir,
            max_capsules,
            no_llm,
            git_history,
            git_commits,
            git_path,
            llm_model,
            llm_max_capsules,
        } => {
            crate::commands::init::run(
                path,
                embed_model,
                embed_cache_dir,
                max_capsules,
                no_llm,
                git_history,
                git_commits,
                git_path,
                llm_model,
                llm_max_capsules,
            )
            .await?;
        }
        crate::cli::Command::Model { command } => {
            crate::commands::model::run(command).await?;
        }
        crate::cli::Command::Config { command } => {
            crate::commands::config::run(command)?;
        }
        crate::cli::Command::Clear { path, yes } => {
            crate::commands::clear::run(path, yes)?;
        }
        crate::cli::Command::Reindex { path, yes } => {
            crate::commands::reindex::run(path, yes).await?;
        }
        crate::cli::Command::Emotion { text } => {
            crate::commands::emotion::run(text).await?;
        }
        crate::cli::Command::Shim { command } => match command {
            crate::cli::ShimCommand::Opencode {
                embed_model,
                embed_cache_dir,
            } => {
                crate::companion::shims::opencode_stdio::run(embed_model, embed_cache_dir).await?;
            }
            crate::cli::ShimCommand::Claudecode {
                embed_model,
                embed_cache_dir,
            } => {
                crate::companion::shims::claudecode::run(embed_model, embed_cache_dir).await?;
            }
        },
        crate::cli::Command::Where { path } => {
            crate::commands::where_cmd::run(path)?;
        }
    }

    Ok(())
}
