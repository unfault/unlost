pub(crate) fn escape_sql_string(s: &str) -> String {
    s.replace("'", "''")
}

pub(crate) fn scope_filter_expr(scope: &str) -> Option<String> {
    let s = scope.trim();
    if s.is_empty() {
        return None;
    }
    let s = escape_sql_string(s);
    Some(format!("array_contains(symbols, '{s}')"))
}

pub(crate) fn parse_time_filter(s: &str) -> anyhow::Result<Option<i64>> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(None);
    }

    let now = chrono::Utc::now();
    let ms = match s {
        _ if s.ends_with('s') => {
            let secs: i64 = s[..s.len() - 1].parse()?;
            now.timestamp_millis() - secs * 1000
        }
        _ if s.ends_with('m') => {
            let mins: i64 = s[..s.len() - 1].parse()?;
            now.timestamp_millis() - mins * 60 * 1000
        }
        _ if s.ends_with('h') => {
            let hours: i64 = s[..s.len() - 1].parse()?;
            now.timestamp_millis() - hours * 60 * 60 * 1000
        }
        _ if s.ends_with('d') => {
            let days: i64 = s[..s.len() - 1].parse()?;
            now.timestamp_millis() - days * 24 * 60 * 60 * 1000
        }
        _ if s.ends_with('w') => {
            let weeks: i64 = s[..s.len() - 1].parse()?;
            now.timestamp_millis() - weeks * 7 * 24 * 60 * 60 * 1000
        }
        _ if s.ends_with('M') => {
            let months: i64 = s[..s.len() - 1].parse()?;
            now.timestamp_millis() - months * 30 * 24 * 60 * 60 * 1000
        }
        _ if s.ends_with('y') => {
            let years: i64 = s[..s.len() - 1].parse()?;
            now.timestamp_millis() - years * 365 * 24 * 60 * 60 * 1000
        }
        _ => {
            let dt = chrono::DateTime::parse_from_rfc3339(s)?;
            dt.timestamp_millis()
        }
    };
    Ok(Some(ms))
}

pub(crate) fn strip_llm_boilerplate(mut s: String) -> String {
    let lower = s.to_ascii_lowercase();
    if let Some(i) = lower
        .find("<system-reminder")
        .or_else(|| lower.find("<system"))
        .or_else(|| lower.find("<commentary"))
        .or_else(|| lower.find("<tool"))
    {
        s.truncate(i);
    }
    s
}

pub(crate) fn wrap_plain_text(input: &str, width: usize) -> String {
    // Plain text wrapping intended for terminal/piping. It preserves list markers
    // with a hanging indent so wrapped bullets stay readable.
    if width < 10 {
        return input.to_string();
    }

    fn split_list_prefix(s: &str) -> Option<(&str, &str)> {
        // Returns (marker, rest) for markdown-ish list lines.
        // Assumes `s` is left-trimmed.
        if let Some(rest) = s.strip_prefix("- ") {
            return Some(("- ", rest));
        }
        if let Some(rest) = s.strip_prefix("* ") {
            return Some(("* ", rest));
        }
        if let Some(rest) = s.strip_prefix("+ ") {
            return Some(("+ ", rest));
        }
        if let Some(rest) = s.strip_prefix("> ") {
            return Some(("> ", rest));
        }

        // Numbered list: 1. foo  /  1) foo
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == 0 {
            return None;
        }
        if i + 1 < bytes.len() && (bytes[i] == b'.' || bytes[i] == b')') && bytes[i + 1] == b' ' {
            let (marker, rest) = s.split_at(i + 2);
            return Some((marker, rest));
        }
        None
    }

    let mut out = String::with_capacity(input.len() + input.len() / 10);
    let mut prev_blank = false;
    for (li, line) in input.lines().enumerate() {
        if li > 0 {
            out.push('\n');
        }

        if line.trim().is_empty() {
            // Keep at most one consecutive blank line.
            if prev_blank {
                continue;
            }
            prev_blank = true;
            continue;
        }
        prev_blank = false;

        if line.len() <= width {
            out.push_str(line.trim_end());
            continue;
        }

        let indent_len = line.chars().take_while(|c| c.is_ascii_whitespace()).count();
        let indent = " ".repeat(indent_len);
        let trimmed = line.trim_start();
        let (marker, rest) = split_list_prefix(trimmed).unwrap_or(("", trimmed));
        let hanging_indent = " ".repeat(indent_len + marker.len());

        let mut cur = String::new();
        cur.push_str(&indent);
        cur.push_str(marker);
        let mut cur_len = indent_len + marker.len();
        let base_len = cur_len;

        for word in rest.split_whitespace() {
            let wlen = word.len();
            let needs_space = cur_len > base_len;
            let add_len = wlen + if needs_space { 1 } else { 0 };

            if cur_len + add_len > width && cur_len > base_len {
                out.push_str(cur.trim_end());
                out.push('\n');
                cur.clear();
                cur.push_str(&hanging_indent);
                cur_len = hanging_indent.len();
            }

            if cur_len > base_len {
                cur.push(' ');
                cur_len += 1;
            }
            cur.push_str(word);
            cur_len += wlen;
        }

        out.push_str(cur.trim_end());
    }
    out
}

fn normalize_file_token(raw: &str) -> Option<String> {
    fn allowed(ch: char) -> bool {
        ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-')
    }

    let mut s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        return None;
    }

    // Trim leading/trailing non-path punctuation.
    while let Some(ch) = s.chars().next() {
        if allowed(ch) {
            break;
        }
        s = &s[ch.len_utf8()..];
    }
    while let Some(ch) = s.chars().rev().next() {
        if allowed(ch) {
            break;
        }
        s = &s[..s.len() - ch.len_utf8()];
    }

    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Normalize slashes and leading ./
    let s = s.replace('\\', "/");
    let s = s.strip_prefix("./").unwrap_or(&s).to_string();
    if s.is_empty() || s == "/" {
        return None;
    }

    // Require at least one alnum so we don't ingest junk like "/".
    if !s.chars().any(|c| c.is_ascii_alphanumeric()) {
        return None;
    }

    // Treat absolute-looking tokens as relative by stripping the leading slash.
    let s = s.strip_prefix('/').unwrap_or(&s).to_string();
    if s.is_empty() {
        return None;
    }

    let has_sep = s.contains('/');
    let last_seg = s.rsplit('/').next().unwrap_or(s.as_str());
    let has_ext = last_seg.rsplit_once('.').is_some_and(|(_, ext)| {
        let ext = ext.trim();
        !ext.is_empty() && ext.len() <= 8 && ext.chars().all(|c| c.is_ascii_alphanumeric())
    });

    // Allow: clear file paths, or common repo directories.
    if has_ext {
        return Some(s);
    }

    if has_sep {
        const PREFIXES: [&str; 8] = [
            "src/",
            "docs/",
            "agents/",
            "tests/",
            "examples/",
            "crates/",
            ".github/",
            "scripts/",
        ];
        if PREFIXES.iter().any(|p| s.starts_with(p)) {
            return Some(s);
        }
    }

    None
}

pub(crate) fn extract_touched_paths_from_exchange_input(input: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // 1) Prefer explicit section injected by the companion flow.
    let mut in_touched = false;
    for line in input.lines() {
        let l = line.trim_end();
        if !in_touched {
            if l.trim() == "Touched paths:" {
                in_touched = true;
            }
            continue;
        }

        if l.trim().is_empty() {
            break;
        }
        if let Some(tok) = normalize_file_token(l) {
            if seen.insert(tok.clone()) {
                out.push(tok);
            }
        }
        if out.len() >= 64 {
            break;
        }
    }

    // 2) Also scan for inline mentions in the conversation slice.
    if out.len() < 64 {
        // Backtick-enclosed tokens are often paths.
        let mut cur = String::new();
        let mut in_ticks = false;
        for ch in input.chars() {
            if ch == '`' {
                if in_ticks {
                    if let Some(tok) = normalize_file_token(&cur) {
                        if seen.insert(tok.clone()) {
                            out.push(tok);
                        }
                    }
                    cur.clear();
                }
                in_ticks = !in_ticks;
                continue;
            }
            if in_ticks {
                if cur.len() < 512 {
                    cur.push(ch);
                }
            }
        }

        // NOTE: we intentionally avoid scanning arbitrary whitespace tokens to reduce false positives.
    }

    out
}

pub(crate) fn augment_capsule_symbols_from_input(capsule: &mut crate::IntentCapsule, input: &str) {
    let extracted = extract_touched_paths_from_exchange_input(input);
    if extracted.is_empty() {
        return;
    }

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut merged: Vec<String> = Vec::new();

    for s in capsule.symbols.iter() {
        if seen.insert(s.clone()) {
            merged.push(s.clone());
        }
    }
    for s in extracted {
        if seen.insert(s.clone()) {
            merged.push(s);
        }
    }

    merged.truncate(32);
    capsule.symbols = merged;
}

pub(crate) fn format_elapsed_time(ts_ms: i64, now_ms: i64) -> String {
    let diff_ms = now_ms - ts_ms;
    if diff_ms < 0 {
        return "in the future".to_string();
    }

    let secs = diff_ms / 1000;
    if secs < 60 {
        return "just now".to_string();
    }

    let mins = secs / 60;
    if mins < 60 {
        return format!("{}m ago", mins);
    }

    let hours = mins / 60;
    let rem_mins = mins % 60;
    if hours < 24 {
        if rem_mins == 0 {
            return format!("{}h ago", hours);
        }
        return format!("{}h {}m ago", hours, rem_mins);
    }

    let days = hours / 24;
    if days == 1 {
        return "yesterday".to_string();
    }
    format!("{}d ago", days)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_elapsed_time() {
        let now = 1000000000000;
        assert_eq!(format_elapsed_time(now - 30000, now), "just now");
        assert_eq!(format_elapsed_time(now - 120000, now), "2m ago");
        assert_eq!(format_elapsed_time(now - 3600000, now), "1h ago");
        assert_eq!(format_elapsed_time(now - 3660000, now), "1h 1m ago");
        assert_eq!(format_elapsed_time(now - 86400000, now), "yesterday");
        assert_eq!(format_elapsed_time(now - 172800000, now), "2d ago");
    }

    #[test]
    fn test_escape_sql_string() {
        assert_eq!(escape_sql_string("abc"), "abc");
        assert_eq!(escape_sql_string("a'b"), "a''b");
        assert_eq!(escape_sql_string("''"), "''''");
    }

    #[test]
    fn test_scope_filter_expr() {
        assert_eq!(scope_filter_expr(""), None);
        assert_eq!(scope_filter_expr("   "), None);
        assert_eq!(
            scope_filter_expr("MySymbol"),
            Some("array_contains(symbols, 'MySymbol')".to_string())
        );
        assert_eq!(
            scope_filter_expr("a'b"),
            Some("array_contains(symbols, 'a''b')".to_string())
        );
    }

    #[test]
    fn test_parse_time_filter_relative() {
        let now = chrono::Utc::now().timestamp_millis();
        assert!(parse_time_filter("1s").unwrap().unwrap() < now);
        assert!(parse_time_filter("5m").unwrap().unwrap() < now);
        assert!(parse_time_filter("2h").unwrap().unwrap() < now);
        assert!(parse_time_filter("1d").unwrap().unwrap() < now);
        assert!(parse_time_filter("1w").unwrap().unwrap() < now);
        assert!(parse_time_filter("1M").unwrap().unwrap() < now);
        assert!(parse_time_filter("1y").unwrap().unwrap() < now);
    }

    #[test]
    fn test_parse_time_filter_empty() {
        assert_eq!(parse_time_filter("").unwrap(), None);
        assert_eq!(parse_time_filter("   ").unwrap(), None);
    }

    #[test]
    fn test_strip_llm_boilerplate() {
        assert_eq!(strip_llm_boilerplate("ok".to_string()), "ok");
        assert_eq!(
            strip_llm_boilerplate("hello\n<system>nope".to_string()),
            "hello\n"
        );
        assert_eq!(
            strip_llm_boilerplate("prefix<tool-call>content".to_string()),
            "prefix"
        );
        assert_eq!(
            strip_llm_boilerplate("content<commentary>comment".to_string()),
            "content"
        );
        assert_eq!(
            strip_llm_boilerplate("mixed<SYSTEM>system".to_string()),
            "mixed"
        );
        assert_eq!(
            strip_llm_boilerplate("no boilerplate here".to_string()),
            "no boilerplate here"
        );
    }
}
