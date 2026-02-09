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

fn looks_like_file_token(mut s: &str) -> Option<&str> {
    s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Trim common punctuation/bracketing around tokens.
    while let Some(ch) = s.chars().next() {
        if ch.is_ascii_punctuation() && ch != '/' && ch != '.' && ch != '_' && ch != '-' {
            s = &s[ch.len_utf8()..];
            continue;
        }
        break;
    }
    while let Some(ch) = s.chars().rev().next() {
        if ch.is_ascii_punctuation() && ch != '/' && ch != '.' && ch != '_' && ch != '-' {
            s = &s[..s.len() - ch.len_utf8()];
            continue;
        }
        break;
    }

    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        return None;
    }

    // Heuristics: paths or common filename extensions.
    if s.contains('/') || s.starts_with("./") {
        return Some(s);
    }

    const EXTS: [&str; 18] = [
        ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".md", ".toml", ".json", ".yml",
        ".yaml", ".html", ".css", ".scss", ".png", ".svg", ".sh",
    ];
    if EXTS.iter().any(|e| s.ends_with(e)) {
        return Some(s);
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
        if let Some(tok) = looks_like_file_token(l) {
            let tok = tok.trim_start_matches("./");
            if !tok.is_empty() && seen.insert(tok.to_string()) {
                out.push(tok.to_string());
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
                    if let Some(tok) = looks_like_file_token(&cur) {
                        let tok = tok.trim_start_matches("./");
                        if !tok.is_empty() && seen.insert(tok.to_string()) {
                            out.push(tok.to_string());
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

        // Whitespace tokens (best-effort).
        for raw in input.split_whitespace() {
            if out.len() >= 64 {
                break;
            }
            if let Some(tok) = looks_like_file_token(raw) {
                let tok = tok.trim_start_matches("./");
                if !tok.is_empty() && seen.insert(tok.to_string()) {
                    out.push(tok.to_string());
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
