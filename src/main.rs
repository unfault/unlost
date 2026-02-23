#![allow(clippy::collapsible_if)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::too_many_arguments)]

use clap::{CommandFactory, Parser};
use unlost::cli::{Cli, Command, OutputFormat, ReplayCommand, ShimCommand};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.command.is_none() {
        Cli::command().print_help()?;
        println!("\nTry:");
        println!("- unlost config agent opencode --path .");
        println!("- unlost config agent claude --global");
        println!("- unlost config llm anthropic --model claude-3-5-sonnet-20241022");
        println!("- unlost init --path .");
        println!("- unlost recall");
        println!("- unlost query \"what are the routes available?\"");
        println!();
        return Ok(());
    }

    // Default to "info" for shim (we want to see friction/emotion logs),
    // "warn" for everything else
    let is_shim = matches!(
        &cli.command,
        Some(Command::Shim { .. }) | Some(Command::Replay { .. })
    );
    let default_level = if is_shim { "info" } else { "warn" };
    let log_level = cli
        .log
        .map(|l: unlost::cli::LogLevel| l.as_tracing_str().to_string())
        .unwrap_or_else(|| default_level.to_string());
    let filter = unlost::logging::create_filter(&log_level);

    // Determine logging mode based on command type:
    // - Shim commands use file-only (stdout/stderr used for protocol)
    // - Long-running server commands log to both file and stderr
    // - Short-lived commands just use stderr
    let is_long_running = matches!(
        &cli.command,
        Some(Command::Serve { .. }) | Some(Command::Record { .. })
    );

    // Keep the guard alive for the duration of main()
    let _log_guard = if is_shim {
        Some(unlost::logging::init_logging_file_only(filter))
    } else if is_long_running {
        Some(unlost::logging::init_logging(filter))
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
        None
    };

    // rustls 0.23 requires selecting a process-level CryptoProvider.
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    match cli.command.unwrap() {
        Command::Serve {
            bind,
            embed_model,
            embed_cache_dir,
        } => {
            unlost::commands::serve::run(bind, embed_model, embed_cache_dir).await?;
        }
        Command::Record {
            bind,
            upstream_host,
            upstream_port,
            embed_model,
            embed_cache_dir,
        } => {
            unlost::commands::record::run(
                bind,
                upstream_host,
                upstream_port,
                embed_model,
                embed_cache_dir,
            )
            .await?;
        }
        Command::Query {
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
            let output = if plain { OutputFormat::Plain } else { output };
            unlost::commands::query::run(
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
        Command::Trace {
            target,
            seeds,
            fan_out,
            threshold,
            since,
            until,
            llm_model,
            no_llm,
            output,
            plain,
            embed_model,
            embed_cache_dir,
        } => {
            let output = if plain { OutputFormat::Plain } else { output };
            unlost::commands::trace::run(
                target,
                seeds,
                fan_out,
                threshold,
                since,
                until,
                no_llm,
                llm_model,
                output,
                embed_model,
                embed_cache_dir,
            )
            .await?;
        }
        Command::Brief {
            target,
            llm_model,
            output,
            plain,
            embed_model,
            embed_cache_dir,
        } => {
            let output = if plain { OutputFormat::Plain } else { output };
            unlost::commands::brief::run(target, llm_model, output, embed_model, embed_cache_dir)
                .await?;
        }
        Command::Explore {
            query,
            llm_model,
            output,
            plain,
            embed_model,
            embed_cache_dir,
        } => {
            let output = if plain { OutputFormat::Plain } else { output };
            unlost::commands::explore::run(query, llm_model, output, embed_model, embed_cache_dir)
                .await?;
        }
        Command::Challenge {
            target,
            deep,
            llm_model,
            output,
            plain,
            embed_model,
            embed_cache_dir,
        } => {
            let output = if plain { OutputFormat::Plain } else { output };
            unlost::commands::challenge::run(
                target, deep, llm_model, output, embed_model, embed_cache_dir,
            )
            .await?;
        }
        Command::Recall {
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
            let output = if plain { OutputFormat::Plain } else { output };
            unlost::commands::recall::run(
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
        Command::Metrics { path } => {
            unlost::commands::metrics::run(path)?;
        }
        Command::Inspect {
            path,
            limit,
            emotion,
            provider,
            since,
            until,
            filter,
        } => {
            unlost::commands::inspect::run(path, limit, emotion, provider, since, until, filter)
                .await?;
        }
        Command::Init {
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
            unlost::commands::init::run(
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
        Command::Model { command } => {
            unlost::commands::model::run(command).await?;
        }
        Command::Config { command } => {
            unlost::commands::config::run(command)?;
        }
        Command::Clear { path, yes } => {
            unlost::commands::clear::run(path, yes)?;
        }
        Command::Reindex { path, yes } => {
            unlost::commands::reindex::run(path, yes).await?;
        }
        Command::Replay { command } => {
            handle_replay(command).await?;
        }
        Command::Emotion { text } => {
            unlost::commands::emotion::run(text).await?;
        }
        Command::Shim { command } => match command {
            ShimCommand::Opencode {
                embed_model,
                embed_cache_dir,
            } => {
                unlost::companion::shims::opencode_stdio::run(embed_model, embed_cache_dir).await?;
            }
            ShimCommand::Claude {
                embed_model,
                embed_cache_dir,
            } => {
                unlost::companion::shims::claude::run(embed_model, embed_cache_dir).await?;
            }
            ShimCommand::Replay { command } => {
                handle_replay(command).await?;
            }
        },
        Command::Where { path } => {
            unlost::commands::where_cmd::run(path)?;
        }
    }

    Ok(())
}


async fn handle_replay(command: ReplayCommand) -> anyhow::Result<()> {
    match command {
        ReplayCommand::Git {
            path,
            max_commits,
            embed_model,
            embed_cache_dir,
        } => {
            unlost::git::replay_git(path, max_commits, embed_model, embed_cache_dir).await?;
        }
        ReplayCommand::Claude {
            path,
            transcript_path,
            session_id,
            from_start,
            dedupe,
            no_extraction,
            full_extraction,
            clear,
            embed_model,
            embed_cache_dir,
            git_grounding,
        } => {
            let mode = if no_extraction {
                unlost::types::ExtractionMode::None
            } else if full_extraction {
                unlost::types::ExtractionMode::Full
            } else {
                unlost::types::ExtractionMode::Hybrid
            };

            unlost::companion::shims::claude::replay(
                path,
                transcript_path,
                session_id,
                from_start,
                dedupe,
                clear,
                mode,
                embed_model,
                embed_cache_dir,
                git_grounding,
            )
            .await?;
        }
        ReplayCommand::Opencode {
            path,
            dedupe,
            no_extraction,
            full_extraction,
            clear,
            embed_model,
            embed_cache_dir,
            git_grounding,
        } => {
            let mode = if no_extraction {
                unlost::types::ExtractionMode::None
            } else if full_extraction {
                unlost::types::ExtractionMode::Full
            } else {
                unlost::types::ExtractionMode::Hybrid
            };

            unlost::companion::shims::opencode::replay(
                path,
                dedupe,
                clear,
                mode,
                embed_model,
                embed_cache_dir,
                git_grounding,
            )
            .await?;
        }
    }
    Ok(())
}
