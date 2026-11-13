//! Parse tool-call outputs exported by parser-conformance with the Rust
//! tool parsers, so the results can be compared with the Python parsers.
//!
//! Reads JSON lines from stdin with `parser`, `tokenizer_file`, `text` and
//! `tools` (OpenAI format), and writes one JSON line per input with the Rust
//! result or the error. Text goes through the tokenizer and the incremental
//! decoder, as the frontend feeds it, then through one `parse_into` call and
//! `finish`.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use vllm_chat::ToolParserFactory;
use vllm_parser::tool::{Tool, ToolParserOutput};
use vllm_tokenizer::{DynTokenizer, HuggingFaceTokenizer};

fn tools(value: &Value) -> Vec<Tool> {
    let Some(list) = value.as_array() else {
        return Vec::new();
    };
    list.iter()
        .map(|tool| {
            let function = &tool["function"];
            Tool {
                name: function["name"].as_str().unwrap_or_default().to_string(),
                description: function["description"].as_str().map(str::to_string),
                parameters: function.get("parameters").cloned().unwrap_or_else(|| json!({})),
                strict: None,
                defer_loading: None,
            }
        })
        .collect()
}

fn parse(row: &Value, tokenizer: DynTokenizer) -> Result<ToolParserOutput, String> {
    let name = row["parser"].as_str().ok_or("missing parser")?;
    let text = row["text"].as_str().ok_or("missing text")?;
    let mut parser = ToolParserFactory::global()
        .create(name, &tools(&row["tools"]), tokenizer.clone())
        .map_err(|e| format!("create: {e}"))?;
    let ids = tokenizer.encode(text, false).map_err(|e| format!("encode: {e}"))?;
    let mut decoder = tokenizer.create_decode_stream(&[], false, 0);
    for &id in &ids {
        decoder.push_token(id).map_err(|e| format!("decode: {e}"))?;
    }
    let decoded = decoder.flush(None).map_err(|e| format!("decode: {e}"))?.1;
    let mut output = ToolParserOutput::default();
    parser.parse_into(decoded, &mut output).map_err(|e| format!("parse: {e}"))?;
    output.append(parser.finish().map_err(|e| format!("finish: {e}"))?);
    Ok(output.coalesce())
}

fn main() {
    let mut tokenizers: HashMap<String, DynTokenizer> = HashMap::new();
    let mut out = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let row: Value = serde_json::from_str(&line.expect("read stdin")).expect("JSON line");
        let file = row["tokenizer_file"].as_str().expect("tokenizer_file").to_string();
        let tokenizer = tokenizers
            .entry(file.clone())
            .or_insert_with(|| {
                Arc::new(HuggingFaceTokenizer::new(Path::new(&file)).expect("load tokenizer"))
            })
            .clone();
        let result = match parse(&row, tokenizer) {
            Ok(output) => json!({
                "content": output.normal_text(),
                "calls": output
                    .calls()
                    .iter()
                    .map(|c| json!({"name": c.name, "arguments": c.arguments}))
                    .collect::<Vec<_>>(),
            }),
            Err(error) => json!({ "error": error }),
        };
        let reply = json!({ "id": row["id"], "parser": row["parser"], "rust": result });
        writeln!(out, "{reply}").expect("write stdout");
    }
}
