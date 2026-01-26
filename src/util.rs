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

pub(crate) fn strip_llm_boilerplate(mut s: String) -> String {
    // Defensive: if the LLM includes any leaked system/tool boilerplate, strip it.
    // Do a case-insensitive search for common tag prefixes.
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
