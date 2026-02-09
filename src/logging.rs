//! Global unlost logging with daily rotation.
//!
//! Logs are stored in the unlost data directory (`~/.unlost/logs/`) as daily rotating files.
//! All unlost instances share the same log files, making debugging easier.

use tracing_appender::rolling::{RollingFileAppender, Rotation};

/// A guard that must be kept alive for the duration of the program to ensure
/// logs are flushed. When dropped, the non-blocking writer will flush remaining logs.
pub struct LogGuard {
    _guard: tracing_appender::non_blocking::WorkerGuard,
}

/// Initialize logging to both stderr and a file in the unlost data directory.
/// Returns a guard that must be kept alive for the duration of logging.
///
/// Log files are created at `~/.unlost/logs/unlost.YYYY-MM-DD.log`.
pub fn init_logging(filter: tracing_subscriber::EnvFilter) -> LogGuard {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let log_dir = crate::unlost_data_root().join("logs");
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "unlost.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(true),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false),
        )
        .init();

    LogGuard { _guard: guard }
}

/// Initialize logging to file only (no stderr).
/// Useful for shims/companions where stdout/stderr are used for protocol communication.
pub fn init_logging_file_only(filter: tracing_subscriber::EnvFilter) -> LogGuard {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let log_dir = crate::unlost_data_root().join("logs");
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "unlost.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false),
        )
        .init();

    LogGuard { _guard: guard }
}

/// Create a filter for unlost logs at the given level, keeping dependencies quiet.
pub fn create_filter(level: &str) -> tracing_subscriber::EnvFilter {
    tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(format!("unlost={},lance=warn,lancedb=warn", level))
    })
}
