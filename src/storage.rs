use anyhow::Context;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array, ListArray,
    RecordBatch, RecordBatchIterator, StringArray,
    builder::{ListBuilder, StringBuilder},
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use futures_util::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::index::{Index, scalar::LabelListIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;
use uuid::Uuid;

pub(crate) const CAPSULES_TABLE: &str = "capsules_v4";

fn capsules_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("ts_ms", DataType::Int64, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("upstream_host", DataType::Utf8, false),
        Field::new("request_path", DataType::Utf8, false),
        Field::new("http_status", DataType::Int32, false),
        Field::new("conn_id", DataType::Int64, false),
        Field::new("exchange_seq", DataType::Int64, false),
        Field::new("agent_session_id", DataType::Utf8, true),
        // Best-effort usage fields (mostly from agent plugins)
        Field::new("agent_provider_id", DataType::Utf8, true),
        Field::new("agent_model_id", DataType::Utf8, true),
        Field::new("agent_cost", DataType::Float64, true),
        Field::new("tokens_input", DataType::Int64, true),
        Field::new("tokens_output", DataType::Int64, true),
        Field::new("tokens_reasoning", DataType::Int64, true),
        Field::new("tokens_cache_read", DataType::Int64, true),
        Field::new("tokens_cache_write", DataType::Int64, true),
        Field::new("user_emotion", DataType::Utf8, true),
        Field::new("user_emotion_conf", DataType::Float32, true),
        Field::new("user_valence", DataType::Float32, true),
        Field::new("user_intensity", DataType::Float32, true),
        Field::new("assistant_emotion", DataType::Utf8, true),
        Field::new("assistant_emotion_conf", DataType::Float32, true),
        Field::new("assistant_valence", DataType::Float32, true),
        Field::new("assistant_intensity", DataType::Float32, true),
        Field::new("category", DataType::Utf8, false),
        Field::new("intent", DataType::Utf8, false),
        Field::new("decision", DataType::Utf8, false),
        Field::new("rationale", DataType::Utf8, false),
        Field::new(
            "next_steps",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new(
            "symbols",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
            false,
        ),
        // HyPE: pre-generated questions this capsule answers; stored as a joined string for FTS.
        Field::new("questions_text", DataType::Utf8, true),
        // Git provenance: HEAD SHA when the buffer opened (always present in git repos).
        Field::new("head_sha", DataType::Utf8, true),
        // Git provenance: SHA of the commit that landed during this turn (sparse).
        Field::new("commit_sha", DataType::Utf8, true),
    ]))
}

pub(crate) async fn ensure_capsules_table(db: &Connection) -> anyhow::Result<lancedb::Table> {
    match db.open_table(CAPSULES_TABLE).execute().await {
        Ok(t) => {
            // Best-effort schema evolution: older installs may not have usage columns.
            if let Ok(schema) = t.schema().await {
                let existing: std::collections::HashSet<&str> =
                    schema.fields().iter().map(|f| f.name().as_str()).collect();
                let mut exprs: Vec<(String, String)> = Vec::new();

                let add_str = |name: &str, exprs: &mut Vec<(String, String)>| {
                    if !existing.contains(name) {
                        exprs.push((name.to_string(), "CAST(NULL AS VARCHAR)".to_string()));
                    }
                };
                let add_i64 = |name: &str, exprs: &mut Vec<(String, String)>| {
                    if !existing.contains(name) {
                        exprs.push((name.to_string(), "CAST(NULL AS BIGINT)".to_string()));
                    }
                };
                let add_f64 = |name: &str, exprs: &mut Vec<(String, String)>| {
                    if !existing.contains(name) {
                        exprs.push((name.to_string(), "CAST(NULL AS DOUBLE)".to_string()));
                    }
                };

                add_str("agent_provider_id", &mut exprs);
                add_str("agent_model_id", &mut exprs);
                add_f64("agent_cost", &mut exprs);
                add_i64("tokens_input", &mut exprs);
                add_i64("tokens_output", &mut exprs);
                add_i64("tokens_reasoning", &mut exprs);
                add_i64("tokens_cache_read", &mut exprs);
                add_i64("tokens_cache_write", &mut exprs);
                add_str("questions_text", &mut exprs);
                add_str("head_sha", &mut exprs);
                add_str("commit_sha", &mut exprs);

                if !exprs.is_empty() {
                    let _ = t
                        .add_columns(
                            lancedb::table::NewColumnTransform::SqlExpressions(exprs),
                            None,
                        )
                        .await;
                }
            }

            Ok(t)
        }
        Err(_) => {
            tracing::info!(table = CAPSULES_TABLE, "creating lancedb table");
            let schema = capsules_schema();

            let id = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let ts_ms = Arc::new(Int64Array::from_iter_values(std::iter::empty::<i64>()));
            let source = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let upstream_host = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let request_path = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let http_status = Arc::new(Int32Array::from_iter_values(std::iter::empty::<i32>()));
            let conn_id = Arc::new(Int64Array::from_iter_values(std::iter::empty::<i64>()));
            let exchange_seq = Arc::new(Int64Array::from_iter_values(std::iter::empty::<i64>()));
            let agent_session_id =
                Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));

            let agent_provider_id =
                Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));
            let agent_model_id =
                Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));
            let agent_cost = Arc::new(Float64Array::from_iter(std::iter::empty::<Option<f64>>()));
            let tokens_input = Arc::new(Int64Array::from_iter(std::iter::empty::<Option<i64>>()));
            let tokens_output = Arc::new(Int64Array::from_iter(std::iter::empty::<Option<i64>>()));
            let tokens_reasoning =
                Arc::new(Int64Array::from_iter(std::iter::empty::<Option<i64>>()));
            let tokens_cache_read =
                Arc::new(Int64Array::from_iter(std::iter::empty::<Option<i64>>()));
            let tokens_cache_write =
                Arc::new(Int64Array::from_iter(std::iter::empty::<Option<i64>>()));

            let user_emotion = Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));
            let user_emotion_conf =
                Arc::new(Float32Array::from_iter(std::iter::empty::<Option<f32>>()));
            let user_valence = Arc::new(Float32Array::from_iter(std::iter::empty::<Option<f32>>()));
            let user_intensity =
                Arc::new(Float32Array::from_iter(std::iter::empty::<Option<f32>>()));
            let assistant_emotion =
                Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));
            let assistant_emotion_conf =
                Arc::new(Float32Array::from_iter(std::iter::empty::<Option<f32>>()));
            let assistant_valence =
                Arc::new(Float32Array::from_iter(std::iter::empty::<Option<f32>>()));
            let assistant_intensity =
                Arc::new(Float32Array::from_iter(std::iter::empty::<Option<f32>>()));

            let category = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let intent = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let decision = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));
            let rationale = Arc::new(StringArray::from_iter_values(std::iter::empty::<&str>()));

            let mut next_steps_builder = ListBuilder::new(StringBuilder::new());
            let next_steps = Arc::new(next_steps_builder.finish());

            let mut symbols_builder = ListBuilder::new(StringBuilder::new());
            let symbols = Arc::new(symbols_builder.finish());

            let embedding = Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    std::iter::empty::<Option<Vec<Option<f32>>>>(),
                    384,
                ),
            );

            let questions_text =
                Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));
            let head_sha =
                Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));
            let commit_sha =
                Arc::new(StringArray::from_iter(std::iter::empty::<Option<&str>>()));

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    id,
                    ts_ms,
                    source,
                    upstream_host,
                    request_path,
                    http_status,
                    conn_id,
                    exchange_seq,
                    agent_session_id,
                    agent_provider_id,
                    agent_model_id,
                    agent_cost,
                    tokens_input,
                    tokens_output,
                    tokens_reasoning,
                    tokens_cache_read,
                    tokens_cache_write,
                    user_emotion,
                    user_emotion_conf,
                    user_valence,
                    user_intensity,
                    assistant_emotion,
                    assistant_emotion_conf,
                    assistant_valence,
                    assistant_intensity,
                    category,
                    intent,
                    decision,
                    rationale,
                    next_steps,
                    symbols,
                    embedding,
                    questions_text,
                    head_sha,
                    commit_sha,
                ],
            )
            .context("failed to build empty schema batch")?;

            let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
            let table = db
                .create_table(CAPSULES_TABLE, Box::new(batches))
                .execute()
                .await
                .with_context(|| format!("failed to create {CAPSULES_TABLE}"))?;

            table
                .create_index(&["embedding"], Index::Auto)
                .execute()
                .await
                .ok();
            table
                .create_index(
                    &["symbols"],
                    Index::LabelList(LabelListIndexBuilder::default()),
                )
                .execute()
                .await
                .ok();
            table
                .create_index(&["ts_ms"], Index::Auto)
                .execute()
                .await
                .ok();

            Ok(table)
        }
    }
}

/// Open the capsules table from an existing connection. Returns an error if the
/// table doesn't exist yet (workspace not initialised).
pub(crate) async fn open_capsules_table(
    db: &Connection,
) -> anyhow::Result<lancedb::Table> {
    db.open_table(CAPSULES_TABLE)
        .execute()
        .await
        .map_err(|e| anyhow::anyhow!("capsules table not found: {e}"))
}

/// Convert a slice of Arrow RecordBatches from the capsules table into
/// `CapsuleHit` values. Shared by the checkpoint module to avoid duplicating
/// the full inline conversion loop.
pub(crate) fn record_batches_to_hits(
    batches: &[RecordBatch],
    _workspace_id: &str,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    let mut out: Vec<crate::CapsuleHit> = Vec::new();
    let limit = usize::MAX;

    for batch in batches {
        let schema = batch.schema();
        let idx = |name: &str| schema.index_of(name).ok();
        let col_str = |name: &str| -> Option<&StringArray> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>())
        };
        let col_f32 = |name: &str| -> Option<&Float32Array> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<Float32Array>())
        };

        let read_emotion = |row: usize,
                            label: Option<&StringArray>,
                            conf: Option<&Float32Array>,
                            val: Option<&Float32Array>,
                            inten: Option<&Float32Array>|
         -> Option<crate::emotion::EmotionMeta> {
            let label = label
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            if label.trim().is_empty() {
                return None;
            }
            let confidence = conf
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let valence = val
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let intensity = inten
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            Some(crate::emotion::EmotionMeta {
                label: label.to_string(),
                valence,
                intensity,
                confidence,
            })
        };

        let id_col = col_str("id");
        let ts_ms_col =
            idx("ts_ms").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let conn_id_col =
            idx("conn_id").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let exchange_seq_col =
            idx("exchange_seq").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let http_status_col =
            idx("http_status").and_then(|i| batch.column(i).as_any().downcast_ref::<Int32Array>());
        let source = col_str("source");
        let intent = col_str("intent");
        let decision = col_str("decision");
        let rationale = col_str("rationale");
        let category = col_str("category");
        let upstream_host = col_str("upstream_host");
        let request_path = col_str("request_path");
        let agent_session_id_col = col_str("agent_session_id");
        let agent_provider_id_col = col_str("agent_provider_id");
        let agent_model_id_col = col_str("agent_model_id");
        let agent_cost_col =
            idx("agent_cost").and_then(|i| batch.column(i).as_any().downcast_ref::<Float64Array>());
        let tokens_input_col =
            idx("tokens_input").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_output_col =
            idx("tokens_output").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_reasoning_col =
            idx("tokens_reasoning").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_cache_read_col =
            idx("tokens_cache_read").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_cache_write_col =
            idx("tokens_cache_write").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let user_emotion_label = col_str("user_emotion");
        let user_emotion_conf = col_f32("user_emotion_conf");
        let user_valence = col_f32("user_valence");
        let user_intensity = col_f32("user_intensity");
        let assistant_emotion_label = col_str("assistant_emotion");
        let assistant_emotion_conf = col_f32("assistant_emotion_conf");
        let assistant_valence = col_f32("assistant_valence");
        let assistant_intensity = col_f32("assistant_intensity");
        let next_steps_col =
            idx("next_steps").and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
        let symbols_col =
            idx("symbols").and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
        let head_sha_col = col_str("head_sha");
        let commit_sha_col = col_str("commit_sha");

        for row in 0..batch.num_rows() {
            if out.len() >= limit {
                break;
            }
            let id = id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("")
                .to_string();
            if id.is_empty() {
                continue;
            }
            let ts_ms = ts_ms_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let conn_id = conn_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let exchange_seq = exchange_seq_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let http_status = http_status_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let cat = category
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let src = source
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let up = upstream_host
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let path = request_path
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let agent_session = agent_session_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let agent_provider_id = agent_provider_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let agent_model_id = agent_model_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let agent_cost =
                agent_cost_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_input =
                tokens_input_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_output =
                tokens_output_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_reasoning =
                tokens_reasoning_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_cache_read =
                tokens_cache_read_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_cache_write =
                tokens_cache_write_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let usage = if agent_provider_id.is_some()
                || agent_model_id.is_some()
                || agent_cost.is_some()
                || tokens_input.is_some()
                || tokens_output.is_some()
                || tokens_reasoning.is_some()
                || tokens_cache_read.is_some()
                || tokens_cache_write.is_some()
            {
                Some(crate::types::UsageMeta {
                    provider_id: agent_provider_id,
                    model_id: agent_model_id,
                    cost: agent_cost,
                    tokens_input,
                    tokens_output,
                    tokens_reasoning,
                    tokens_cache_read,
                    tokens_cache_write,
                })
            } else {
                None
            };
            let user_emotion = read_emotion(
                row,
                user_emotion_label,
                user_emotion_conf,
                user_valence,
                user_intensity,
            );
            let assistant_emotion = read_emotion(
                row,
                assistant_emotion_label,
                assistant_emotion_conf,
                assistant_valence,
                assistant_intensity,
            );
            let int_text = intent
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("")
                .to_string();
            let dec_text = decision
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("")
                .to_string();
            let rat_text = rationale
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("")
                .to_string();
            let read_string_list = |col: Option<&ListArray>| -> Vec<String> {
                let col = match col {
                    Some(c) => c,
                    None => return vec![],
                };
                if col.is_null(row) {
                    return vec![];
                }
                let list = col.value(row);
                let str_arr = match list.as_any().downcast_ref::<StringArray>() {
                    Some(a) => a,
                    None => return vec![],
                };
                (0..str_arr.len())
                    .filter_map(|i| {
                        (!str_arr.is_null(i)).then(|| str_arr.value(i).to_string())
                    })
                    .filter(|s| !s.trim().is_empty())
                    .collect()
            };
            let next_steps_vec = read_string_list(next_steps_col);
            let symbols_vec = read_string_list(symbols_col);
            let head_sha = head_sha_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let commit_sha = commit_sha_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));

            let cap = crate::types::IntentCapsule {
                category: cat.to_string(),
                intent: int_text,
                decision: dec_text,
                rationale: rat_text,
                next_steps: next_steps_vec,
                symbols: symbols_vec,
                user_symbols: vec![],
                failure_mode: crate::types::FailureMode::None,
                failure_signals: None,
                extraction_mode: crate::types::ExtractionMode::default(),
                questions: vec![],
            };
            let meta = crate::types::ResponseMeta {
                source: src.to_string(),
                upstream_host: up.to_string(),
                request_path: path.to_string(),
                http_status: http_status as u16,
                agent_session_id: agent_session,
                usage,
            };
            out.push(crate::CapsuleHit {
                id,
                ts_ms,
                conn_id,
                exchange_seq,
                capsule: cap,
                meta,
                distance: 0.0,
                user_emotion,
                assistant_emotion,
                head_sha,
                commit_sha,
            });
        }
    }
    Ok(out)
}

pub(crate) async fn query_capsules_lancedb(
    query_text: &str,
    limit: usize,
    symbol: Option<&str>,
    emotion: Option<&str>,
    provider: Option<&str>,
    since: Option<i64>,
    until: Option<i64>,
    embedder: crate::embed::Embedder,
    ws: &crate::WorkspacePaths,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;

    let table = match db.open_table(CAPSULES_TABLE).execute().await {
        Ok(t) => t,
        Err(_) => anyhow::bail!("capsules table not found (workspace_id={})", ws.id),
    };

    let q_embedding = crate::embed::embed_text(&embedder, query_text).await?;

    let mut q = table
        .query()
        .nearest_to(q_embedding.as_slice())?
        .column("embedding")
        .limit(limit);

    let mut filters: Vec<String> = Vec::new();

    if let Some(sym) = symbol {
        let sym = crate::util::escape_sql_string(sym);
        filters.push(format!("array_contains(symbols, '{sym}')"));
    }

    if let Some(emotion) = emotion {
        let emotion = crate::util::escape_sql_string(emotion);
        filters.push(format!(
            "user_emotion = '{emotion}' OR assistant_emotion = '{emotion}'"
        ));
    }

    if let Some(provider) = provider {
        let provider_host = match provider {
            "openai" => "api.openai.com",
            "anthropic" => "api.anthropic.com",
            "opencode" => "opencode.ai",
            _ => provider,
        };
        filters.push(format!("upstream_host = '{provider_host}'"));
    }

    if let Some(since_ms) = since {
        filters.push(format!("ts_ms >= {since_ms}"));
    }

    if let Some(until_ms) = until {
        filters.push(format!("ts_ms <= {until_ms}"));
    }

    if !filters.is_empty() {
        let combined = filters.join(" AND ");
        q = q.only_if(combined);
    }

    let batches = q.execute().await?.try_collect::<Vec<_>>().await?;
    if batches.is_empty() {
        return Ok(vec![]);
    }

    let mut out: Vec<crate::CapsuleHit> = Vec::new();

    for batch in batches {
        let schema = batch.schema();
        let idx = |name: &str| schema.index_of(name).ok();
        let col_str = |name: &str| -> Option<&StringArray> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>())
        };

        let col_f32 = |name: &str| -> Option<&Float32Array> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<Float32Array>())
        };

        let read_emotion = |row: usize,
                            label: Option<&StringArray>,
                            conf: Option<&Float32Array>,
                            val: Option<&Float32Array>,
                            inten: Option<&Float32Array>|
         -> Option<crate::emotion::EmotionMeta> {
            let label = label
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            if label.trim().is_empty() {
                return None;
            }
            let confidence = conf
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let valence = val
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let intensity = inten
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            Some(crate::emotion::EmotionMeta {
                label: label.to_string(),
                valence,
                intensity,
                confidence,
            })
        };

        let id_col = col_str("id");
        let ts_ms_col =
            idx("ts_ms").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let conn_id_col =
            idx("conn_id").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let exchange_seq_col =
            idx("exchange_seq").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let http_status_col =
            idx("http_status").and_then(|i| batch.column(i).as_any().downcast_ref::<Int32Array>());
        let source = col_str("source");
        let intent = col_str("intent");
        let decision = col_str("decision");
        let rationale = col_str("rationale");
        let category = col_str("category");
        let upstream_host = col_str("upstream_host");
        let request_path = col_str("request_path");
        let agent_session_id_col = col_str("agent_session_id");

        let agent_provider_id_col = col_str("agent_provider_id");
        let agent_model_id_col = col_str("agent_model_id");
        let agent_cost_col =
            idx("agent_cost").and_then(|i| batch.column(i).as_any().downcast_ref::<Float64Array>());
        let tokens_input_col =
            idx("tokens_input").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_output_col = idx("tokens_output")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_reasoning_col = idx("tokens_reasoning")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_cache_read_col = idx("tokens_cache_read")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_cache_write_col = idx("tokens_cache_write")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());

        let user_emotion_label = col_str("user_emotion");
        let user_emotion_conf = col_f32("user_emotion_conf");
        let user_valence = col_f32("user_valence");
        let user_intensity = col_f32("user_intensity");
        let assistant_emotion_label = col_str("assistant_emotion");
        let assistant_emotion_conf = col_f32("assistant_emotion_conf");
        let assistant_valence = col_f32("assistant_valence");
        let assistant_intensity = col_f32("assistant_intensity");

        let distance = idx("_distance").and_then(|i| {
            batch
                .column(i)
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
        });

        let next_steps =
            idx("next_steps").and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
        let symbols =
            idx("symbols").and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
        let head_sha_col = col_str("head_sha");
        let commit_sha_col = col_str("commit_sha");

        for row in 0..batch.num_rows() {
            if out.len() >= limit {
                break;
            }

            let dist = distance
                .and_then(|d| (!d.is_null(row)).then(|| d.value(row)))
                .unwrap_or_default();
            let id = id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let ts_ms = ts_ms_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let conn_id = conn_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let exchange_seq = exchange_seq_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let http_status = http_status_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let cat = category
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let src = source
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let up = upstream_host
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let path = request_path
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let agent_session = agent_session_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));

            let agent_provider_id = agent_provider_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let agent_model_id = agent_model_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let agent_cost = agent_cost_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_input =
                tokens_input_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_output =
                tokens_output_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_reasoning =
                tokens_reasoning_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_cache_read =
                tokens_cache_read_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_cache_write =
                tokens_cache_write_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));

            let usage = if agent_provider_id.is_some()
                || agent_model_id.is_some()
                || agent_cost.is_some()
                || tokens_input.is_some()
                || tokens_output.is_some()
                || tokens_reasoning.is_some()
                || tokens_cache_read.is_some()
                || tokens_cache_write.is_some()
            {
                Some(crate::types::UsageMeta {
                    provider_id: agent_provider_id,
                    model_id: agent_model_id,
                    cost: agent_cost,
                    tokens_input,
                    tokens_output,
                    tokens_reasoning,
                    tokens_cache_read,
                    tokens_cache_write,
                })
            } else {
                None
            };

            let i_text = intent
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let d_text = decision
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let r_text = rationale
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");

            let mut syms: Vec<String> = Vec::new();
            if let Some(sym_arr) = symbols {
                if !sym_arr.is_null(row) {
                    let values = sym_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        syms = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            let mut steps: Vec<String> = Vec::new();
            if let Some(ns_arr) = next_steps {
                if !ns_arr.is_null(row) {
                    let values = ns_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        steps = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            out.push(crate::CapsuleHit {
                id: id.to_string(),
                ts_ms,
                conn_id,
                exchange_seq,
                distance: dist,
                user_emotion: read_emotion(
                    row,
                    user_emotion_label,
                    user_emotion_conf,
                    user_valence,
                    user_intensity,
                ),
                assistant_emotion: read_emotion(
                    row,
                    assistant_emotion_label,
                    assistant_emotion_conf,
                    assistant_valence,
                    assistant_intensity,
                ),
                capsule: crate::IntentCapsule {
                    category: cat.to_string(),
                    intent: i_text.to_string(),
                    decision: d_text.to_string(),
                    rationale: r_text.to_string(),
                    next_steps: steps,
                    symbols: syms,
                    user_symbols: vec![], // Not stored in DB yet
                    // Existing capsules in DB don't have failure_mode yet
                    failure_mode: crate::types::FailureMode::None,
                    failure_signals: None,
                    extraction_mode: crate::types::ExtractionMode::None,
                    questions: vec![],
                },
                meta: crate::ResponseMeta {
                    source: src.to_string(),
                    upstream_host: up.to_string(),
                    request_path: path.to_string(),
                    http_status: (http_status.max(0) as u16),
                    agent_session_id: agent_session,
                    usage,
                },
                head_sha: head_sha_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string())),
                commit_sha: commit_sha_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string())),
            });
        }
    }

    Ok(out)
}

pub(crate) async fn scan_capsules_lancedb(
    ws: &crate::WorkspacePaths,
    limit: usize,
    symbol: Option<&str>,
    emotion: Option<&str>,
    provider: Option<&str>,
    since: Option<i64>,
    until: Option<i64>,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    scan_capsules_lancedb_impl(ws, limit, symbol, emotion, provider, since, until, false).await
}

/// Like `scan_capsules_lancedb` but returns the most recent rows first.
/// Uses offset to skip to near the end of the table (assuming append-only insertion),
/// fetches those rows, then sorts by ts_ms descending.
pub(crate) async fn scan_capsules_lancedb_recent(
    ws: &crate::WorkspacePaths,
    limit: usize,
    symbol: Option<&str>,
    emotion: Option<&str>,
    provider: Option<&str>,
    since: Option<i64>,
    until: Option<i64>,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    scan_capsules_lancedb_impl(ws, limit, symbol, emotion, provider, since, until, true).await
}

async fn scan_capsules_lancedb_impl(
    ws: &crate::WorkspacePaths,
    limit: usize,
    symbol: Option<&str>,
    emotion: Option<&str>,
    provider: Option<&str>,
    since: Option<i64>,
    until: Option<i64>,
    recent_first: bool,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = match db.open_table(CAPSULES_TABLE).execute().await {
        Ok(t) => t,
        Err(_) => return Ok(vec![]),
    };

    let mut q = table.query();

    // For recent_first, we skip to near the end of the table and fetch more rows to account for filtering
    if recent_first {
        let total = table.count_rows(None).await.unwrap_or(0);
        let fetch_count = limit * 3; // fetch extra to handle potential filter reduction
        if total > fetch_count {
            q = q.offset(total - fetch_count);
        }
        q = q.limit(fetch_count);
    } else {
        q = q.limit(limit);
    }

    let mut filters: Vec<String> = Vec::new();

    if let Some(sym) = symbol {
        let sym = crate::util::escape_sql_string(sym);
        filters.push(format!("array_contains(symbols, '{sym}')"));
    }

    if let Some(emotion) = emotion {
        let emotion = crate::util::escape_sql_string(emotion);
        filters.push(format!(
            "user_emotion = '{emotion}' OR assistant_emotion = '{emotion}'"
        ));
    }

    if let Some(provider) = provider {
        let provider_host = match provider {
            "openai" => "api.openai.com",
            "anthropic" => "api.anthropic.com",
            "opencode" => "opencode.ai",
            _ => provider,
        };
        filters.push(format!("upstream_host = '{provider_host}'"));
    }

    if let Some(since_ms) = since {
        filters.push(format!("ts_ms >= {since_ms}"));
    }

    if let Some(until_ms) = until {
        filters.push(format!("ts_ms <= {until_ms}"));
    }

    if !filters.is_empty() {
        let combined = filters.join(" AND ");
        q = q.only_if(combined);
    }

    let batches = q.execute().await?.try_collect::<Vec<_>>().await?;
    if batches.is_empty() {
        return Ok(vec![]);
    }

    let mut out: Vec<crate::CapsuleHit> = Vec::new();

    for batch in batches {
        let schema = batch.schema();
        let idx = |name: &str| schema.index_of(name).ok();
        let col_str = |name: &str| -> Option<&StringArray> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>())
        };

        let col_f32 = |name: &str| -> Option<&Float32Array> {
            idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<Float32Array>())
        };

        let read_emotion = |row: usize,
                            label: Option<&StringArray>,
                            conf: Option<&Float32Array>,
                            val: Option<&Float32Array>,
                            inten: Option<&Float32Array>|
         -> Option<crate::emotion::EmotionMeta> {
            let label = label
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            if label.trim().is_empty() {
                return None;
            }
            let confidence = conf
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let valence = val
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let intensity = inten
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            Some(crate::emotion::EmotionMeta {
                label: label.to_string(),
                valence,
                intensity,
                confidence,
            })
        };

        let id_col = col_str("id");
        let ts_ms_col =
            idx("ts_ms").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let conn_id_col =
            idx("conn_id").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let exchange_seq_col =
            idx("exchange_seq").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let http_status_col =
            idx("http_status").and_then(|i| batch.column(i).as_any().downcast_ref::<Int32Array>());

        let source = col_str("source");
        let intent = col_str("intent");
        let decision = col_str("decision");
        let rationale = col_str("rationale");
        let category = col_str("category");
        let upstream_host = col_str("upstream_host");
        let request_path = col_str("request_path");
        let agent_session_id_col = col_str("agent_session_id");

        let agent_provider_id_col = col_str("agent_provider_id");
        let agent_model_id_col = col_str("agent_model_id");
        let agent_cost_col =
            idx("agent_cost").and_then(|i| batch.column(i).as_any().downcast_ref::<Float64Array>());
        let tokens_input_col =
            idx("tokens_input").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_output_col = idx("tokens_output")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_reasoning_col = idx("tokens_reasoning")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_cache_read_col = idx("tokens_cache_read")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
        let tokens_cache_write_col = idx("tokens_cache_write")
            .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());

        let user_emotion_label = col_str("user_emotion");
        let user_emotion_conf = col_f32("user_emotion_conf");
        let user_valence = col_f32("user_valence");
        let user_intensity = col_f32("user_intensity");
        let assistant_emotion_label = col_str("assistant_emotion");
        let assistant_emotion_conf = col_f32("assistant_emotion_conf");
        let assistant_valence = col_f32("assistant_valence");
        let assistant_intensity = col_f32("assistant_intensity");

        let next_steps =
            idx("next_steps").and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
        let symbols =
            idx("symbols").and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
        let questions_text_col = col_str("questions_text");
        let head_sha_col = col_str("head_sha");
        let commit_sha_col = col_str("commit_sha");

        for row in 0..batch.num_rows() {
            // Skip early-exit when recent_first since we need all rows to sort
            if !recent_first && out.len() >= limit {
                break;
            }
            let cat = category
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let up = upstream_host
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let path = request_path
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let src = source
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let id = id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let ts_ms = ts_ms_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let conn_id = conn_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let exchange_seq = exchange_seq_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let http_status = http_status_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or_default();
            let agent_session = agent_session_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));

            let agent_provider_id = agent_provider_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let agent_model_id = agent_model_id_col
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
            let agent_cost = agent_cost_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_input =
                tokens_input_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_output =
                tokens_output_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_reasoning =
                tokens_reasoning_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_cache_read =
                tokens_cache_read_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));
            let tokens_cache_write =
                tokens_cache_write_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row)));

            let usage = if agent_provider_id.is_some()
                || agent_model_id.is_some()
                || agent_cost.is_some()
                || tokens_input.is_some()
                || tokens_output.is_some()
                || tokens_reasoning.is_some()
                || tokens_cache_read.is_some()
                || tokens_cache_write.is_some()
            {
                Some(crate::types::UsageMeta {
                    provider_id: agent_provider_id,
                    model_id: agent_model_id,
                    cost: agent_cost,
                    tokens_input,
                    tokens_output,
                    tokens_reasoning,
                    tokens_cache_read,
                    tokens_cache_write,
                })
            } else {
                None
            };
            let i_text = intent
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let d_text = decision
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");
            let r_text = rationale
                .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                .unwrap_or("");

            let mut syms: Vec<String> = Vec::new();
            if let Some(sym_arr) = symbols {
                if !sym_arr.is_null(row) {
                    let values = sym_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        syms = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            let mut steps: Vec<String> = Vec::new();
            if let Some(ns_arr) = next_steps {
                if !ns_arr.is_null(row) {
                    let values = ns_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        steps = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
            }

            out.push(crate::CapsuleHit {
                id: id.to_string(),
                ts_ms,
                conn_id,
                exchange_seq,
                distance: 0.0,
                user_emotion: read_emotion(
                    row,
                    user_emotion_label,
                    user_emotion_conf,
                    user_valence,
                    user_intensity,
                ),
                assistant_emotion: read_emotion(
                    row,
                    assistant_emotion_label,
                    assistant_emotion_conf,
                    assistant_valence,
                    assistant_intensity,
                ),
                capsule: crate::IntentCapsule {
                    category: cat.to_string(),
                    intent: i_text.to_string(),
                    decision: d_text.to_string(),
                    rationale: r_text.to_string(),
                    next_steps: steps,
                    symbols: syms,
                    user_symbols: vec![], // Not stored in DB yet
                    // Existing capsules in DB don't have failure_mode yet
                    failure_mode: crate::types::FailureMode::None,
                    failure_signals: None,
                    extraction_mode: crate::types::ExtractionMode::None,
                    questions: questions_text_col
                        .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                        .map(|s| {
                            s.split('\n')
                                .filter(|q| !q.is_empty())
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default(),
                },
                meta: crate::ResponseMeta {
                    source: src.to_string(),
                    upstream_host: up.to_string(),
                    request_path: path.to_string(),
                    http_status: (http_status.max(0) as u16),
                    agent_session_id: agent_session,
                    usage,
                },
                head_sha: head_sha_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string())),
                commit_sha: commit_sha_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string())),
            });
        }
    }

    if recent_first {
        out.sort_by(|a, b| a.ts_ms.cmp(&b.ts_ms));
        out.truncate(limit);
    }

    Ok(out)
}

/// The retrieval intent each command has when it calls `query_capsules_lancedb`.
///
/// Each variant maps to a question-style prefix that is prepended to the user's raw
/// target/query string before embedding.  This exploits HyPE (Hypothetical Prompt
/// Embeddings): at indexing time we stored pre-generated questions in `questions_text`
/// and embedded capsule content *alongside* those questions.  By framing the query in
/// the same question style as the stored prompts, retrieval becomes a
/// question-to-question match rather than a keyword-to-document match, which yields
/// higher precision without any extra LLM call at query time.
///
/// Rules:
///   - If `target` is already phrased as a question (contains '?') we leave it as-is
///     and only add the intent prefix to bias the embedding.
///   - If `target` is empty the prefix alone is used so the ANN still finds a useful
///     seed neighbourhood.
#[derive(Debug, Clone, Copy)]
pub(crate) enum QueryIntent {
    /// `recall` — chronological story of what happened
    Recall,
    /// `brief` — current state and rationale (staff-engineer debrief)
    Brief,
    /// `challenge` — pressure-test a past decision
    Challenge,
    /// `explore` — forward-looking alternatives and trade-offs
    Explore,
    /// `trace` — causal chain leading to the current state
    Trace,
}

/// Frame a raw user query/target with a command-specific question prefix so that the
/// resulting embedding aligns with the HyPE question vectors stored at indexing time.
///
/// Returns the framed string ready to pass directly to `embed_text`.
pub(crate) fn frame_query_for_command(target: &str, intent: QueryIntent) -> String {
    let target = target.trim();
    let prefix = match intent {
        QueryIntent::Recall => "What happened with",
        QueryIntent::Brief => "Why is the current state of",
        QueryIntent::Challenge => "Was the decision about",
        QueryIntent::Explore => "What are the alternatives and trade-offs for",
        QueryIntent::Trace => "What sequence of decisions led to",
    };

    if target.is_empty() {
        // No user target — use the prefix alone as an intent signal
        return prefix.to_string();
    }

    // If already a question, prepend the intent prefix as a soft bias
    if target.contains('?') {
        return format!("{prefix}: {target}");
    }

    // Build a natural question from the prefix + target
    match intent {
        QueryIntent::Brief => format!("{prefix} {target} the way it is?"),
        QueryIntent::Challenge => format!("{prefix} {target} the right call?"),
        QueryIntent::Trace | QueryIntent::Recall | QueryIntent::Explore => {
            format!("{prefix} {target}?")
        }
    }
}

/// Build the canonical embed text for a capsule.
///
/// Includes category, failure mode, top symbols, and the structured intent/decision/rationale
/// so that the embedding encodes semantic trajectory rather than just point-in-time wording.
///
/// `prior_decision` carries the most recent decision from the same insertion sequence
/// (e.g. the previous capsule's decision text). This encodes causal continuity into the
/// vector — capsules that are part of the same work thread end up closer in embedding space
/// even when the session ID is reused across unrelated work.
pub(crate) fn capsule_embed_text_with_prior(
    c: &crate::IntentCapsule,
    prior_decision: Option<&str>,
) -> String {
    let mut s = String::new();

    // Category grounds the semantic domain (e.g. "Debugging" vs "Architecture")
    if !c.category.trim().is_empty() && c.category.trim() != "unknown" {
        s.push_str("category: ");
        s.push_str(c.category.trim());
        s.push('\n');
    }

    // Failure mode is high-signal for causal chains — pain points cluster
    if c.failure_mode != crate::types::FailureMode::None {
        let fm = match c.failure_mode {
            crate::types::FailureMode::Drift => "drift",
            crate::types::FailureMode::Rediscovery => "rediscovery",
            crate::types::FailureMode::DecisionConflict => "decision_conflict",
            crate::types::FailureMode::RetrySpiral => "retry_spiral",
            crate::types::FailureMode::FalseProgress => "false_progress",
            crate::types::FailureMode::UnboundedHorizon => "unbounded_horizon",
            crate::types::FailureMode::None => "",
        };
        if !fm.is_empty() {
            s.push_str("failure_mode: ");
            s.push_str(fm);
            s.push('\n');
        }
    }

    // Top symbols anchor the capsule to concrete code locations
    let syms: Vec<&str> = c.symbols.iter().take(5).map(|s| s.as_str()).collect();
    if !syms.is_empty() {
        s.push_str("symbols: ");
        s.push_str(&syms.join(", "));
        s.push('\n');
    }

    // Prior decision encodes causal continuity: where did we come from?
    if let Some(prior) = prior_decision {
        let prior = prior.trim();
        if !prior.is_empty() {
            // Truncate to keep the embed text focused
            let prior = if prior.len() > 120 {
                &prior[..120]
            } else {
                prior
            };
            s.push_str("prior: ");
            s.push_str(prior);
            s.push('\n');
        }
    }

    if !c.intent.trim().is_empty() {
        s.push_str("intent: ");
        s.push_str(c.intent.trim());
        s.push('\n');
    }
    if !c.decision.trim().is_empty() {
        s.push_str("decision: ");
        s.push_str(c.decision.trim());
        s.push('\n');
    }
    if !c.rationale.trim().is_empty() {
        s.push_str("rationale: ");
        s.push_str(c.rationale.trim());
        s.push('\n');
    }
    s
}

/// Build a causal chain of capsules anchored to a query.
///
/// Algorithm:
/// 1. Run ANN vector search to get a seed set of semantically relevant capsules.
/// 2. For each seed, fan out to capsules that share symbols (existing LabelList index).
/// 3. Optionally filter to capsules older than the most-recent seed (`backwards_only`).
/// 4. Apply a similarity threshold to stop the chain before it goes off-topic.
/// 5. Deduplicate by id and sort ascending by ts_ms so the chain reads chronologically.
///
/// This surfaces the causal path of decisions that led to the current state of a file or
/// concept — even across different agent sessions and non-contiguous time windows.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn trace_capsules_lancedb(
    query: &str,
    seed_limit: usize,
    fan_out_per_seed: usize,
    distance_threshold: f32,
    since_ms: Option<i64>,
    until_ms: Option<i64>,
    session_id: Option<&str>,
    embedder: crate::embed::Embedder,
    ws: &crate::WorkspacePaths,
) -> anyhow::Result<Vec<crate::CapsuleHit>> {
    // Step 1: seed set via vector ANN
    let seeds = query_capsules_lancedb(
        query, seed_limit, None, None, None, since_ms, until_ms, embedder, ws,
    )
    .await?;

    // If a session_id filter is active, restrict seeds to that session.
    let seeds: Vec<crate::CapsuleHit> = if let Some(sid) = session_id {
        seeds
            .into_iter()
            .filter(|h| {
                h.meta
                    .agent_session_id
                    .as_deref()
                    .map(|s| s == sid)
                    .unwrap_or(false)
            })
            .collect()
    } else {
        seeds
    };

    if seeds.is_empty() {
        return Ok(vec![]);
    }

    // Step 2: collect all unique symbols from seeds
    let all_symbols: std::collections::HashSet<String> = seeds
        .iter()
        .flat_map(|h| h.capsule.symbols.iter().cloned())
        .collect();

    // The causal chain looks backwards from the most recent seed
    let newest_seed_ts = seeds.iter().map(|h| h.ts_ms).max().unwrap_or(i64::MAX);

    let db = lancedb::connect(ws.db_dir.to_string_lossy().as_ref())
        .execute()
        .await?;
    let table = match db.open_table(CAPSULES_TABLE).execute().await {
        Ok(t) => t,
        Err(_) => return Ok(seeds),
    };

    let mut all_hits: std::collections::HashMap<String, crate::CapsuleHit> =
        std::collections::HashMap::new();

    // Add seeds first (they pass the threshold by definition)
    for h in seeds {
        all_hits.insert(h.id.clone(), h);
    }

    // Step 3: fan out — for each symbol, fetch capsules that touch it
    for sym in all_symbols.iter().take(12) {
        let sym_escaped = crate::util::escape_sql_string(sym);
        let mut filter_parts = vec![format!("array_contains(symbols, '{sym_escaped}')")];

        // Backwards in time: only include capsules older than the newest seed
        filter_parts.push(format!("ts_ms <= {newest_seed_ts}"));
        if let Some(since) = since_ms {
            filter_parts.push(format!("ts_ms >= {since}"));
        }
        if let Some(until) = until_ms {
            filter_parts.push(format!("ts_ms <= {until}"));
        }
        if let Some(sid) = session_id {
            let sid_escaped = crate::util::escape_sql_string(sid);
            filter_parts.push(format!("agent_session_id = '{sid_escaped}'"));
        }

        let filter = filter_parts.join(" AND ");
        let batches = match table
            .query()
            .only_if(filter)
            .limit(fan_out_per_seed)
            .execute()
            .await
        {
            Ok(s) => s.try_collect::<Vec<_>>().await.unwrap_or_default(),
            Err(_) => continue,
        };

        // Parse rows — reuse the scan row parser via a mini inline parse
        for batch in &batches {
            let schema = batch.schema();
            let idx = |name: &str| schema.index_of(name).ok();
            let col_str = |name: &str| -> Option<&StringArray> {
                idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<StringArray>())
            };
            let col_f32 = |name: &str| -> Option<&Float32Array> {
                idx(name).and_then(|i| batch.column(i).as_any().downcast_ref::<Float32Array>())
            };

            let id_col = col_str("id");
            let ts_ms_col =
                idx("ts_ms").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
            let conn_id_col =
                idx("conn_id").and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
            let exchange_seq_col = idx("exchange_seq")
                .and_then(|i| batch.column(i).as_any().downcast_ref::<Int64Array>());
            let http_status_col = idx("http_status")
                .and_then(|i| batch.column(i).as_any().downcast_ref::<Int32Array>());
            let source = col_str("source");
            let intent = col_str("intent");
            let decision = col_str("decision");
            let rationale = col_str("rationale");
            let category = col_str("category");
            let upstream_host = col_str("upstream_host");
            let request_path = col_str("request_path");
            let agent_session_id_col = col_str("agent_session_id");
            let user_emotion_label = col_str("user_emotion");
            let user_emotion_conf = col_f32("user_emotion_conf");
            let user_valence = col_f32("user_valence");
            let user_intensity = col_f32("user_intensity");
            let assistant_emotion_label = col_str("assistant_emotion");
            let assistant_emotion_conf = col_f32("assistant_emotion_conf");
            let assistant_valence = col_f32("assistant_valence");
            let assistant_intensity = col_f32("assistant_intensity");
            let next_steps_col = idx("next_steps")
                .and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
            let symbols_col =
                idx("symbols").and_then(|i| batch.column(i).as_any().downcast_ref::<ListArray>());
            let head_sha_col = col_str("head_sha");
            let commit_sha_col = col_str("commit_sha");

            for row in 0..batch.num_rows() {
                let id = match id_col.and_then(|a| (!a.is_null(row)).then(|| a.value(row))) {
                    Some(v) if !v.is_empty() => v.to_string(),
                    _ => continue,
                };
                if all_hits.contains_key(&id) {
                    continue;
                }

                let ts_ms = ts_ms_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or_default();
                let conn_id = conn_id_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or_default();
                let exchange_seq = exchange_seq_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or_default();
                let http_status = http_status_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or_default();
                let cat = category
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or("");
                let src = source
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or("");
                let up = upstream_host
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or("");
                let path = request_path
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or("");
                let agent_session = agent_session_id_col
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string()));
                let i_text = intent
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or("");
                let d_text = decision
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or("");
                let r_text = rationale
                    .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                    .unwrap_or("");

                let read_emotion_local = |label: Option<&StringArray>,
                                          conf: Option<&Float32Array>,
                                          val: Option<&Float32Array>,
                                          inten: Option<&Float32Array>|
                 -> Option<crate::emotion::EmotionMeta> {
                    let lbl = label
                        .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                        .unwrap_or("");
                    if lbl.trim().is_empty() {
                        return None;
                    }
                    Some(crate::emotion::EmotionMeta {
                        label: lbl.to_string(),
                        confidence: conf
                            .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                            .unwrap_or_default(),
                        valence: val
                            .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                            .unwrap_or_default(),
                        intensity: inten
                            .and_then(|a| (!a.is_null(row)).then(|| a.value(row)))
                            .unwrap_or_default(),
                    })
                };

                let mut syms: Vec<String> = Vec::new();
                if let Some(sym_arr) = symbols_col
                    && !sym_arr.is_null(row)
                {
                    let values = sym_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        syms = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }
                let mut steps: Vec<String> = Vec::new();
                if let Some(ns_arr) = next_steps_col
                    && !ns_arr.is_null(row)
                {
                    let values = ns_arr.value(row);
                    if let Some(sa) = values.as_any().downcast_ref::<StringArray>() {
                        steps = (0..sa.len())
                            .filter(|&i| !sa.is_null(i))
                            .map(|i| sa.value(i).to_string())
                            .collect();
                    }
                }

                // Fan-out hits are linked by symbol, not by semantic score. We can't
                // compute a real embedding distance here without an extra embed call, so
                // we use content quality as a proxy guard: drop capsules that carry no
                // meaningful signal (empty intent *and* empty decision — ghost extractions).
                // Valid capsules are admitted with distance = threshold * 0.9 to indicate
                // they are symbol-linked rather than semantically ranked.
                if i_text.trim().is_empty() && d_text.trim().is_empty() {
                    continue;
                }
                let fan_distance = distance_threshold * 0.9;

                all_hits.insert(
                    id.clone(),
                    crate::CapsuleHit {
                        id,
                        ts_ms,
                        conn_id,
                        exchange_seq,
                        distance: fan_distance,
                        user_emotion: read_emotion_local(
                            user_emotion_label,
                            user_emotion_conf,
                            user_valence,
                            user_intensity,
                        ),
                        assistant_emotion: read_emotion_local(
                            assistant_emotion_label,
                            assistant_emotion_conf,
                            assistant_valence,
                            assistant_intensity,
                        ),
                        capsule: crate::IntentCapsule {
                            category: cat.to_string(),
                            intent: i_text.to_string(),
                            decision: d_text.to_string(),
                            rationale: r_text.to_string(),
                            next_steps: steps,
                            symbols: syms,
                            user_symbols: vec![],
                            failure_mode: crate::types::FailureMode::None,
                            failure_signals: None,
                            extraction_mode: crate::types::ExtractionMode::None,
                            questions: vec![],
                        },
                        meta: crate::ResponseMeta {
                            source: src.to_string(),
                            upstream_host: up.to_string(),
                            request_path: path.to_string(),
                            http_status: http_status.max(0) as u16,
                            agent_session_id: agent_session,
                            usage: None,
                        },
                        head_sha: head_sha_col
                            .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string())),
                        commit_sha: commit_sha_col
                            .and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string())),
                    },
                );

                if all_hits.len() >= seed_limit * fan_out_per_seed {
                    break;
                }
            }
        }
    }

    // Sort chronologically (oldest first) — this IS the causal chain order
    let mut chain: Vec<crate::CapsuleHit> = all_hits.into_values().collect();
    chain.sort_by_key(|h| h.ts_ms);
    Ok(chain)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn insert_capsule_row(
    db: &Connection,
    embedder: &crate::embed::Embedder,
    conn_id: u64,
    exchange_seq: u64,
    ts_ms: i64,
    meta: &crate::ResponseMeta,
    user_emotion: Option<&crate::emotion::EmotionMeta>,
    assistant_emotion: Option<&crate::emotion::EmotionMeta>,
    capsule: &crate::IntentCapsule,
    // Prior decision text from the preceding capsule in the same session/sequence.
    // Encodes causal continuity into the embedding so work threads cluster in vector space.
    prior_decision: Option<&str>,
    // git HEAD SHA when the buffer opened (short, 7-char). None if not a git repo.
    head_sha: Option<&str>,
    // git SHA of the commit that landed during this turn, if detected. Sparse.
    commit_sha: Option<&str>,
) -> anyhow::Result<()> {
    tracing::info!(
        conn_id,
        exchange_seq,
        ts_ms,
        has_usage = meta.usage.is_some(),
        decision_bytes = capsule.decision.len(),
        has_prior = prior_decision.is_some(),
        "insert_capsule_row called"
    );
    tracing::debug!(
        conn_id,
        exchange_seq,
        decision_bytes = capsule.decision.len(),
        symbols = capsule.symbols.len(),
        "inserting capsule"
    );
    let table = ensure_capsules_table(db).await?;
    let schema = capsules_schema();

    let text_to_embed = capsule_embed_text_with_prior(capsule, prior_decision);
    let embedding = crate::embed::embed_text(embedder, &text_to_embed).await?;
    if embedding.len() != 384 {
        anyhow::bail!("embedding dimension mismatch: {}", embedding.len());
    }

    let id = Uuid::new_v4().to_string();

    let id_arr = Arc::new(StringArray::from(vec![id.as_str()]));
    let ts_ms_arr = Arc::new(Int64Array::from(vec![ts_ms]));
    let source_arr = Arc::new(StringArray::from(vec![meta.source.as_str()]));
    let upstream_host_arr = Arc::new(StringArray::from(vec![meta.upstream_host.as_str()]));
    let request_path_arr = Arc::new(StringArray::from(vec![meta.request_path.as_str()]));
    let http_status_arr = Arc::new(Int32Array::from(vec![meta.http_status as i32]));
    let conn_id_arr = Arc::new(Int64Array::from(vec![conn_id as i64]));
    let exchange_seq_arr = Arc::new(Int64Array::from(vec![exchange_seq as i64]));
    let agent_session_id_arr = Arc::new(StringArray::from(vec![meta.agent_session_id.as_deref()]));

    let agent_provider_id_arr = Arc::new(StringArray::from(vec![
        meta.usage.as_ref().and_then(|u| u.provider_id.as_deref()),
    ]));
    let agent_model_id_arr = Arc::new(StringArray::from(vec![
        meta.usage.as_ref().and_then(|u| u.model_id.as_deref()),
    ]));
    let agent_cost_arr = Arc::new(Float64Array::from(vec![
        meta.usage.as_ref().and_then(|u| u.cost),
    ]));
    let tokens_input_arr = Arc::new(Int64Array::from(vec![
        meta.usage.as_ref().and_then(|u| u.tokens_input),
    ]));
    let tokens_output_arr = Arc::new(Int64Array::from(vec![
        meta.usage.as_ref().and_then(|u| u.tokens_output),
    ]));
    let tokens_reasoning_arr = Arc::new(Int64Array::from(vec![
        meta.usage.as_ref().and_then(|u| u.tokens_reasoning),
    ]));
    let tokens_cache_read_arr = Arc::new(Int64Array::from(vec![
        meta.usage.as_ref().and_then(|u| u.tokens_cache_read),
    ]));
    let tokens_cache_write_arr = Arc::new(Int64Array::from(vec![
        meta.usage.as_ref().and_then(|u| u.tokens_cache_write),
    ]));

    let user_emotion_arr = Arc::new(StringArray::from(vec![
        user_emotion.map(|e| e.label.as_str()),
    ]));
    let user_emotion_conf_arr =
        Arc::new(Float32Array::from(vec![user_emotion.map(|e| e.confidence)]));
    let user_valence_arr = Arc::new(Float32Array::from(vec![user_emotion.map(|e| e.valence)]));
    let user_intensity_arr = Arc::new(Float32Array::from(vec![user_emotion.map(|e| e.intensity)]));

    let assistant_emotion_arr = Arc::new(StringArray::from(vec![
        assistant_emotion.map(|e| e.label.as_str()),
    ]));
    let assistant_emotion_conf_arr = Arc::new(Float32Array::from(vec![
        assistant_emotion.map(|e| e.confidence),
    ]));
    let assistant_valence_arr = Arc::new(Float32Array::from(vec![
        assistant_emotion.map(|e| e.valence),
    ]));
    let assistant_intensity_arr = Arc::new(Float32Array::from(vec![
        assistant_emotion.map(|e| e.intensity),
    ]));
    let category_arr = Arc::new(StringArray::from(vec![capsule.category.as_str()]));
    let intent_arr = Arc::new(StringArray::from(vec![capsule.intent.as_str()]));
    let decision_arr = Arc::new(StringArray::from(vec![capsule.decision.as_str()]));
    let rationale_arr = Arc::new(StringArray::from(vec![capsule.rationale.as_str()]));

    let mut next_steps_builder = ListBuilder::new(StringBuilder::new());
    for step in &capsule.next_steps {
        next_steps_builder.values().append_value(step);
    }
    next_steps_builder.append(true);
    let next_steps_arr = Arc::new(next_steps_builder.finish());

    let mut symbols_builder = ListBuilder::new(StringBuilder::new());
    for sym in &capsule.symbols {
        symbols_builder.values().append_value(sym);
    }
    symbols_builder.append(true);
    let symbols_arr = Arc::new(symbols_builder.finish());

    let embedding_arr = Arc::new(
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            std::iter::once(Some(embedding.into_iter().map(Some).collect::<Vec<_>>())),
            384,
        ),
    );

    // HyPE: join questions into a single text field for search/display
    let questions_joined = if capsule.questions.is_empty() {
        None
    } else {
        Some(capsule.questions.join("\n"))
    };
    let questions_text_arr = Arc::new(StringArray::from(vec![questions_joined.as_deref()]));
    let head_sha_arr = Arc::new(StringArray::from(vec![head_sha]));
    let commit_sha_arr = Arc::new(StringArray::from(vec![commit_sha]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            id_arr,
            ts_ms_arr,
            source_arr,
            upstream_host_arr,
            request_path_arr,
            http_status_arr,
            conn_id_arr,
            exchange_seq_arr,
            agent_session_id_arr,
            agent_provider_id_arr,
            agent_model_id_arr,
            agent_cost_arr,
            tokens_input_arr,
            tokens_output_arr,
            tokens_reasoning_arr,
            tokens_cache_read_arr,
            tokens_cache_write_arr,
            user_emotion_arr,
            user_emotion_conf_arr,
            user_valence_arr,
            user_intensity_arr,
            assistant_emotion_arr,
            assistant_emotion_conf_arr,
            assistant_valence_arr,
            assistant_intensity_arr,
            category_arr,
            intent_arr,
            decision_arr,
            rationale_arr,
            next_steps_arr,
            symbols_arr,
            embedding_arr,
            questions_text_arr,
            head_sha_arr,
            commit_sha_arr,
        ],
    )
    .context("failed to build insert batch")?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
    table
        .add(batches)
        .execute()
        .await
        .context("lancedb insert failed")?;
    Ok(())
}

/// A pre-assembled row ready for batch insertion.
pub(crate) struct CapsuleRow {
    pub conn_id: u64,
    pub exchange_seq: u64,
    pub ts_ms: i64,
    pub meta: crate::ResponseMeta,
    pub capsule: crate::IntentCapsule,
    /// Pre-computed embedding vector (len must be 384).
    pub embedding: Vec<f32>,
    /// git HEAD SHA when the buffer opened. None for reindexed/replayed rows.
    pub head_sha: Option<String>,
    /// git SHA of the commit that landed during this turn, if detected.
    pub commit_sha: Option<String>,
}

/// Insert a batch of capsule rows in a single LanceDB write.
/// Callers are responsible for computing embeddings up front
/// (e.g. via `crate::embed::embed_texts_batch`).
pub(crate) async fn insert_capsule_batch(
    table: &lancedb::Table,
    rows: &[CapsuleRow],
) -> anyhow::Result<()> {
    if rows.is_empty() {
        return Ok(());
    }

    let schema = capsules_schema();
    let n = rows.len();

    let mut ids: Vec<String> = Vec::with_capacity(n);
    let mut ts_ms_vec: Vec<i64> = Vec::with_capacity(n);
    let mut source_vec: Vec<String> = Vec::with_capacity(n);
    let mut upstream_host_vec: Vec<String> = Vec::with_capacity(n);
    let mut request_path_vec: Vec<String> = Vec::with_capacity(n);
    let mut http_status_vec: Vec<i32> = Vec::with_capacity(n);
    let mut conn_id_vec: Vec<i64> = Vec::with_capacity(n);
    let mut exchange_seq_vec: Vec<i64> = Vec::with_capacity(n);
    let mut agent_session_id_vec: Vec<Option<String>> = Vec::with_capacity(n);
    let mut agent_provider_id_vec: Vec<Option<String>> = Vec::with_capacity(n);
    let mut agent_model_id_vec: Vec<Option<String>> = Vec::with_capacity(n);
    let mut agent_cost_vec: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut tokens_input_vec: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut tokens_output_vec: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut tokens_reasoning_vec: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut tokens_cache_read_vec: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut tokens_cache_write_vec: Vec<Option<i64>> = Vec::with_capacity(n);
    // emotions: all None during reindex (not stored in JSONL)
    let mut category_vec: Vec<String> = Vec::with_capacity(n);
    let mut intent_vec: Vec<String> = Vec::with_capacity(n);
    let mut decision_vec: Vec<String> = Vec::with_capacity(n);
    let mut rationale_vec: Vec<String> = Vec::with_capacity(n);
    let mut next_steps_builder = ListBuilder::new(StringBuilder::new());
    let mut symbols_builder = ListBuilder::new(StringBuilder::new());
    let mut questions_text_vec: Vec<Option<String>> = Vec::with_capacity(n);
    let mut head_sha_vec: Vec<Option<String>> = Vec::with_capacity(n);
    let mut commit_sha_vec: Vec<Option<String>> = Vec::with_capacity(n);
    // Flat embedding storage: n * 384 f32 values
    let mut embeddings_flat: Vec<Option<Vec<Option<f32>>>> = Vec::with_capacity(n);

    for row in rows {
        ids.push(Uuid::new_v4().to_string());
        ts_ms_vec.push(row.ts_ms);
        source_vec.push(row.meta.source.clone());
        upstream_host_vec.push(row.meta.upstream_host.clone());
        request_path_vec.push(row.meta.request_path.clone());
        http_status_vec.push(row.meta.http_status as i32);
        conn_id_vec.push(row.conn_id as i64);
        exchange_seq_vec.push(row.exchange_seq as i64);
        agent_session_id_vec.push(row.meta.agent_session_id.clone());
        agent_provider_id_vec.push(row.meta.usage.as_ref().and_then(|u| u.provider_id.clone()));
        agent_model_id_vec.push(row.meta.usage.as_ref().and_then(|u| u.model_id.clone()));
        agent_cost_vec.push(row.meta.usage.as_ref().and_then(|u| u.cost));
        tokens_input_vec.push(row.meta.usage.as_ref().and_then(|u| u.tokens_input));
        tokens_output_vec.push(row.meta.usage.as_ref().and_then(|u| u.tokens_output));
        tokens_reasoning_vec.push(row.meta.usage.as_ref().and_then(|u| u.tokens_reasoning));
        tokens_cache_read_vec.push(row.meta.usage.as_ref().and_then(|u| u.tokens_cache_read));
        tokens_cache_write_vec.push(row.meta.usage.as_ref().and_then(|u| u.tokens_cache_write));
        category_vec.push(row.capsule.category.clone());
        intent_vec.push(row.capsule.intent.clone());
        decision_vec.push(row.capsule.decision.clone());
        rationale_vec.push(row.capsule.rationale.clone());

        for step in &row.capsule.next_steps {
            next_steps_builder.values().append_value(step);
        }
        next_steps_builder.append(true);

        for sym in &row.capsule.symbols {
            symbols_builder.values().append_value(sym);
        }
        symbols_builder.append(true);

        questions_text_vec.push(if row.capsule.questions.is_empty() {
            None
        } else {
            Some(row.capsule.questions.join("\n"))
        });
        head_sha_vec.push(row.head_sha.clone());
        commit_sha_vec.push(row.commit_sha.clone());

        if row.embedding.len() != 384 {
            anyhow::bail!(
                "embedding dimension mismatch in batch: got {}",
                row.embedding.len()
            );
        }
        embeddings_flat.push(Some(row.embedding.iter().map(|&v| Some(v)).collect()));
    }

    let null_f32: Vec<Option<f32>> = vec![None; n];
    let null_str: Vec<Option<&str>> = vec![None; n];

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(ts_ms_vec)),
            Arc::new(StringArray::from(
                source_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                upstream_host_vec
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                request_path_vec
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(http_status_vec)),
            Arc::new(Int64Array::from(conn_id_vec)),
            Arc::new(Int64Array::from(exchange_seq_vec)),
            Arc::new(StringArray::from(
                agent_session_id_vec
                    .iter()
                    .map(|o| o.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                agent_provider_id_vec
                    .iter()
                    .map(|o| o.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                agent_model_id_vec
                    .iter()
                    .map(|o| o.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(agent_cost_vec)),
            Arc::new(Int64Array::from(tokens_input_vec)),
            Arc::new(Int64Array::from(tokens_output_vec)),
            Arc::new(Int64Array::from(tokens_reasoning_vec)),
            Arc::new(Int64Array::from(tokens_cache_read_vec)),
            Arc::new(Int64Array::from(tokens_cache_write_vec)),
            // emotions: all null during reindex
            Arc::new(StringArray::from(null_str.clone())),
            Arc::new(Float32Array::from(null_f32.clone())),
            Arc::new(Float32Array::from(null_f32.clone())),
            Arc::new(Float32Array::from(null_f32.clone())),
            Arc::new(StringArray::from(null_str.clone())),
            Arc::new(Float32Array::from(null_f32.clone())),
            Arc::new(Float32Array::from(null_f32.clone())),
            Arc::new(Float32Array::from(null_f32.clone())),
            Arc::new(StringArray::from(
                category_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                intent_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                decision_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rationale_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(next_steps_builder.finish()),
            Arc::new(symbols_builder.finish()),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(embeddings_flat, 384),
            ),
            Arc::new(StringArray::from(
                questions_text_vec
                    .iter()
                    .map(|o| o.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                head_sha_vec
                    .iter()
                    .map(|o| o.as_deref())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                commit_sha_vec
                    .iter()
                    .map(|o| o.as_deref())
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .context("failed to build batch insert RecordBatch")?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
    table
        .add(batches)
        .execute()
        .await
        .context("lancedb batch insert failed")?;
    Ok(())
}
