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
            "\n\nTry:\n- unlost serve --bind 127.0.0.1:3000\n- unlost configure agent opencode --path . --server http://127.0.0.1:3000\n- unlost config llm anthropic --model claude-3-5-sonnet-20241022\n- unlost init --path .\n- unlost recall\n- unlost query \"what are the routes available?\"\n"
        );
        return Ok(());
    }

    let filter = if let Some(level) = cli.log {
        // Keep dependency noise low unless user opts in via RUST_LOG.
        tracing_subscriber::EnvFilter::new(format!(
            "unlost={},lance=warn,lancedb=warn",
            level.as_tracing_str()
        ))
    } else {
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"))
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

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
        },
    }

    Ok(())
}
