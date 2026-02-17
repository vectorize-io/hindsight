use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::api::{ApiClient, RecallRequest, ReflectRequest, MemoryItem, RetainRequest};
use crate::config;
use crate::output::{self, OutputFormat};
use crate::ui;

// Import types from generated client
use hindsight_client::types::{Budget, ChunkIncludeOptions, IncludeOptions, TagsMatch};
use serde::Deserialize;
use serde_json;

// Local types for serde_json::Value deserialization
#[derive(Debug, Deserialize)]
struct MemoryUnitDetail {
    id: String,
    text: String,
    #[serde(rename = "type")]
    type_: Option<String>,
    document_id: Option<String>,
    context: Option<String>,
    occurred_start: Option<String>,
    occurred_end: Option<String>,
    entities: Option<Vec<EntityRef>>,
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct EntityRef {
    id: String,
    name: String,
}

// Helper function to parse budget string to Budget enum
fn parse_budget(budget: &str) -> Budget {
    match budget.to_lowercase().as_str() {
        "low" => Budget::Low,
        "high" => Budget::High,
        _ => Budget::Mid, // Default to mid
    }
}

/// List memory units with pagination and optional filters
pub fn list(
    client: &ApiClient,
    bank_id: &str,
    type_filter: Option<String>,
    query: Option<String>,
    limit: i64,
    offset: i64,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching memories..."))
    } else {
        None
    };

    let response = client.list_memories(
        bank_id,
        type_filter.as_deref(),
        query.as_deref(),
        Some(limit),
        Some(offset),
        verbose,
    );

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Memories: {} (showing {}-{})", bank_id, offset + 1, offset + result.items.len() as i64));

                if result.items.is_empty() {
                    println!("  {}", ui::dim("No memories found."));
                } else {
                    for item in &result.items {
                        let fact_type = item.get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let type_t = match fact_type {
                            "world" => 0.0,
                            "experience" => 0.5,
                            "opinion" => 1.0,
                            _ => 0.5,
                        };

                        let id = item.get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");

                        println!(
                            "  {} {}",
                            ui::gradient(&format!("[{}]", fact_type.to_uppercase()), type_t),
                            ui::dim(id)
                        );

                        // Truncate text if too long
                        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                            let text_preview: String = text.chars().take(100).collect();
                            let ellipsis = if text.len() > 100 { "..." } else { "" };
                            println!("    {}{}", text_preview, ellipsis);
                        }

                        if let Some(doc_id) = item.get("document_id").and_then(|v| v.as_str()) {
                            println!("    {} {}", ui::dim("doc:"), ui::dim(doc_id));
                        }
                        println!();
                    }

                    println!("  {} {} total", ui::dim("Total:"), result.total);
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Get a specific memory unit by ID
pub fn get(
    client: &ApiClient,
    bank_id: &str,
    memory_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching memory..."))
    } else {
        None
    };

    let response = client.get_memory(bank_id, memory_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(value) => {
            if output_format == OutputFormat::Pretty {
                let result: MemoryUnitDetail = serde_json::from_value(value)
                    .with_context(|| "Failed to parse memory response")?;

                let fact_type = result.type_.as_deref().unwrap_or("unknown");
                let type_t = match fact_type {
                    "world" => 0.0,
                    "experience" => 0.5,
                    "opinion" => 1.0,
                    _ => 0.5,
                };

                ui::print_section_header(&format!("Memory: {}", memory_id));

                println!("  {} {}", ui::dim("Type:"), ui::gradient(&fact_type.to_uppercase(), type_t));
                println!("  {} {}", ui::dim("ID:"), result.id);

                if let Some(doc_id) = &result.document_id {
                    println!("  {} {}", ui::dim("Document:"), doc_id);
                }

                if let Some(context) = &result.context {
                    println!("  {} {}", ui::dim("Context:"), context);
                }

                println!();
                println!("{}", ui::gradient_text("─── Content ───"));
                println!();
                println!("{}", result.text);

                // Show temporal info if available
                if result.occurred_start.is_some() || result.occurred_end.is_some() {
                    println!();
                    println!("{}", ui::gradient_text("─── Temporal ───"));
                    if let Some(start) = &result.occurred_start {
                        println!("  {} {}", ui::dim("Start:"), start);
                    }
                    if let Some(end) = &result.occurred_end {
                        println!("  {} {}", ui::dim("End:"), end);
                    }
                }

                // Show entities if available
                if let Some(entities) = &result.entities {
                    if !entities.is_empty() {
                        println!();
                        println!("{}", ui::gradient_text("─── Entities ───"));
                        for entity in entities {
                            println!("  • {} ({})", entity.name, entity.id);
                        }
                    }
                }

                // Show tags if available
                if let Some(tags) = &result.tags {
                    if !tags.is_empty() {
                        println!();
                        println!("{}", ui::gradient_text("─── Tags ───"));
                        println!("  {}", tags.join(", "));
                    }
                }

                println!();
            } else {
                output::print_output(&value, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// Helper function to check if a file is supported by the file converter (markitdown)
fn is_supported_file(path: &std::path::Path) -> bool {
    const SUPPORTED_EXTENSIONS: &[&str] = &[
        // Documents
        "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
        // Images (OCR)
        "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff",
        // Web / markup
        "html", "htm",
        // Text / data
        "txt", "md", "csv", "json", "yaml", "yml", "toml", "xml", "rst", "adoc", "log",
        // Audio (transcription)
        "mp3", "wav", "ogg", "flac",
    ];
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn recall(
    client: &ApiClient,
    agent_id: &str,
    query: String,
    fact_type: Vec<String>,
    budget: String,
    max_tokens: i64,
    trace: bool,
    include_chunks: bool,
    chunk_max_tokens: i64,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Recalling memories..."))
    } else {
        None
    };

    // Build include options if chunks are requested
    let include = if include_chunks {
        Some(IncludeOptions {
            chunks: Some(ChunkIncludeOptions {
                max_tokens: chunk_max_tokens,
            }),
            entities: None,
        })
    } else {
        None
    };

    let request = RecallRequest {
        query,
        types: if fact_type.is_empty() { None } else { Some(fact_type) },
        budget: Some(parse_budget(&budget)),
        max_tokens,
        trace,
        query_timestamp: None,
        include,
        tags: None,
        tags_match: TagsMatch::Any,
    };

    let response = client.recall(agent_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_search_results(&result, trace, include_chunks);
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn reflect(
    client: &ApiClient,
    agent_id: &str,
    query: String,
    budget: String,
    context: Option<String>,
    max_tokens: Option<i64>,
    schema_path: Option<PathBuf>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Reflecting..."))
    } else {
        None
    };

    // Load and parse schema if provided
    let response_schema = if let Some(path) = schema_path {
        let schema_content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read schema file: {}", path.display()))?;
        let schema: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&schema_content)
            .with_context(|| format!("Failed to parse JSON schema from: {}", path.display()))?;
        Some(schema)
    } else {
        None
    };

    let request = ReflectRequest {
        query,
        budget: Some(parse_budget(&budget)),
        context,
        max_tokens: max_tokens.unwrap_or(4096),
        include: None,
        response_schema,
        tags: None,
        tags_match: TagsMatch::Any,
    };

    let response = client.reflect(agent_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_think_response(&result);
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn retain(
    client: &ApiClient,
    agent_id: &str,
    content: String,
    doc_id: Option<String>,
    context: Option<String>,
    r#async: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let doc_id = doc_id.unwrap_or_else(config::generate_doc_id);

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Retaining memory..."))
    } else {
        None
    };

    let item = MemoryItem {
        content: content.clone(),
        context,
        metadata: None,
        timestamp: None,
        document_id: Some(doc_id.clone()),
        entities: None,
        tags: None,
    };

    let request = RetainRequest {
        items: vec![item],
        async_: r#async,
        document_tags: None,
    };

    let response = client.retain(agent_id, &request, r#async, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!(
                    "Memory retained successfully (document: {})",
                    doc_id
                ));
                if result.is_async {
                    println!("  Status: queued for background processing");
                    println!("  Items: {}", result.items_count);
                } else {
                    println!("  Stored count: {}", result.items_count);
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn retain_files(
    client: &ApiClient,
    agent_id: &str,
    path: PathBuf,
    recursive: bool,
    context: Option<String>,
    r#async: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Path does not exist: {}", path.display());
    }

    let mut file_paths = Vec::new();

    if path.is_file() {
        file_paths.push(path);
    } else if path.is_dir() {
        if recursive {
            for entry in WalkDir::new(&path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let file_path = entry.path();
                if is_supported_file(file_path) {
                    file_paths.push(file_path.to_path_buf());
                }
            }
        } else {
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.is_file() && is_supported_file(&file_path) {
                    file_paths.push(file_path);
                }
            }
        }
    }

    if file_paths.is_empty() {
        ui::print_warning("No supported files found. Supported formats: pdf, docx, pptx, xlsx, jpg, png, html, txt, md, csv, mp3, wav, and more.");
        return Ok(());
    }

    ui::print_info(&format!("Found {} file(s) to import", file_paths.len()));

    // Batch files (max 10 per request)
    const BATCH_SIZE: usize = 10;
    let batches: Vec<&[PathBuf]> = file_paths.chunks(BATCH_SIZE).collect();
    let mut all_operation_ids: Vec<String> = Vec::new();

    let pb = ui::create_progress_bar(file_paths.len() as u64, "Uploading files");

    for batch in &batches {
        let mut file_data: Vec<(String, Vec<u8>)> = Vec::new();
        for file_path in *batch {
            let filename = file_path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "file".to_string());
            let content = fs::read(file_path)
                .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
            file_data.push((filename, content));
            pb.inc(1);
        }

        let result = client.file_retain(agent_id, file_data, context.clone(), None, verbose)?;
        all_operation_ids.extend(result.operation_ids);
    }

    pb.finish_with_message("Files uploaded");

    if r#async {
        if output_format == OutputFormat::Pretty {
            ui::print_success("Files queued for processing");
            println!("  Files: {}", file_paths.len());
            for op_id in &all_operation_ids {
                println!("  Operation ID: {}", op_id);
            }
        } else {
            let result = serde_json::json!({ "operation_ids": all_operation_ids });
            output::print_output(&result, output_format)?;
        }
    } else {
        // Poll all operations until they complete
        let poll_spinner = if output_format == OutputFormat::Pretty {
            Some(ui::create_spinner("Processing files..."))
        } else {
            None
        };

        let mut failed = Vec::new();
        for op_id in &all_operation_ids {
            let (success, error_msg) = client.poll_operation(agent_id, op_id, verbose)?;
            if !success {
                failed.push(error_msg.unwrap_or_else(|| "Unknown error".to_string()));
            }
        }

        if let Some(mut sp) = poll_spinner {
            sp.finish();
        }

        if failed.is_empty() {
            if output_format == OutputFormat::Pretty {
                ui::print_success("Files retained successfully");
                println!("  Files processed: {}", file_paths.len());
            } else {
                let result = serde_json::json!({
                    "success": true,
                    "files_count": file_paths.len(),
                    "operation_ids": all_operation_ids,
                });
                output::print_output(&result, output_format)?;
            }
        } else {
            for msg in &failed {
                if output_format == OutputFormat::Pretty {
                    ui::print_error(&format!("Retain operation failed: {}", msg));
                }
            }
            anyhow::bail!("{} operation(s) failed", failed.len());
        }
    }

    Ok(())
}

pub fn delete(
    client: &ApiClient,
    agent_id: &str,
    unit_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting memory unit..."))
    } else {
        None
    };

    let response = client.delete_memory(agent_id, unit_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                if result.success {
                    ui::print_success("Memory unit deleted successfully");
                } else {
                    ui::print_error("Failed to delete memory unit");
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn clear(
    client: &ApiClient,
    agent_id: &str,
    fact_type: Option<String>,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // Confirmation prompt unless -y flag is used
    if !yes && output_format == OutputFormat::Pretty {
        let message = if let Some(ft) = &fact_type {
            format!(
                "Are you sure you want to clear all '{}' memories for bank '{}'? This cannot be undone.",
                ft, agent_id
            )
        } else {
            format!(
                "Are you sure you want to clear ALL memories for bank '{}'? This cannot be undone.",
                agent_id
            )
        };

        let confirmed = ui::prompt_confirmation(&message)?;

        if !confirmed {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner_msg = if let Some(ft) = &fact_type {
        format!("Clearing {} memories...", ft)
    } else {
        "Clearing all memories...".to_string()
    };

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner(&spinner_msg))
    } else {
        None
    };

    let response = client.clear_memories(agent_id, fact_type.as_deref(), verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                if result.success {
                    let msg = if fact_type.is_some() {
                        "Memories cleared successfully"
                    } else {
                        "All memories cleared successfully"
                    };
                    ui::print_success(msg);
                } else {
                    ui::print_error("Failed to clear memories");
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_is_text_file_supported_extensions() {
        let supported = [
            "file.txt", "file.md", "file.json", "file.yaml", "file.yml",
            "file.toml", "file.xml", "file.csv", "file.log", "file.rst", "file.adoc",
        ];
        for filename in supported {
            assert!(
                is_text_file(Path::new(filename)),
                "{} should be recognized as a text file",
                filename
            );
        }
    }

    #[test]
    fn test_is_text_file_case_insensitive() {
        assert!(is_text_file(Path::new("file.JSON")));
        assert!(is_text_file(Path::new("file.TXT")));
        assert!(is_text_file(Path::new("file.Md")));
        assert!(is_text_file(Path::new("file.YAML")));
    }

    #[test]
    fn test_is_text_file_unsupported_extensions() {
        let unsupported = [
            "file.pdf", "file.doc", "file.docx", "file.png", "file.jpg",
            "file.exe", "file.bin", "file.zip", "file.tar", "file.gz",
        ];
        for filename in unsupported {
            assert!(
                !is_text_file(Path::new(filename)),
                "{} should NOT be recognized as a text file",
                filename
            );
        }
    }

    #[test]
    fn test_is_text_file_no_extension() {
        assert!(!is_text_file(Path::new("README")));
        assert!(!is_text_file(Path::new("Makefile")));
        assert!(!is_text_file(Path::new(".gitignore")));
    }

    #[test]
    fn test_is_text_file_with_path() {
        assert!(is_text_file(Path::new("/some/path/to/file.json")));
        assert!(is_text_file(Path::new("../relative/path/file.md")));
        assert!(!is_text_file(Path::new("/path/to/image.png")));
    }

    #[test]
    fn test_parse_budget_valid_values() {
        assert!(matches!(parse_budget("low"), Budget::Low));
        assert!(matches!(parse_budget("mid"), Budget::Mid));
        assert!(matches!(parse_budget("high"), Budget::High));
    }

    #[test]
    fn test_parse_budget_case_insensitive() {
        assert!(matches!(parse_budget("LOW"), Budget::Low));
        assert!(matches!(parse_budget("MID"), Budget::Mid));
        assert!(matches!(parse_budget("HIGH"), Budget::High));
        assert!(matches!(parse_budget("Low"), Budget::Low));
        assert!(matches!(parse_budget("High"), Budget::High));
    }

    #[test]
    fn test_parse_budget_defaults_to_mid() {
        assert!(matches!(parse_budget("invalid"), Budget::Mid));
        assert!(matches!(parse_budget(""), Budget::Mid));
        assert!(matches!(parse_budget("unknown"), Budget::Mid));
    }
}
