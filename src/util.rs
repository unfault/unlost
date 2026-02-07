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
