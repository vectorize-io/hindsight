use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

use crate::api::{ApiClient, RecallRequest, ReflectRequest, MemoryItem, RetainRequest};
use crate::config;
use crate::output::{self, OutputFormat};
use crate::ui;

// Import types from generated client
use hindsight_client::types::{Budget, ChunkIncludeOptions, IncludeOptions};

// Helper function to parse budget string to Budget enum
fn parse_budget(budget: &str) -> Budget {
    match budget.to_lowercase().as_str() {
        "low" => Budget::Low,
        "high" => Budget::High,
        _ => Budget::Mid, // Default to mid
    }
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
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Reflecting..."))
    } else {
        None
    };

    let request = ReflectRequest {
        query,
        budget: Some(parse_budget(&budget)),
        context,
        include: None,
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
    };

    let request = RetainRequest {
        items: vec![item],
        async_: r#async,
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

    let mut files = Vec::new();

    if path.is_file() {
        files.push(path);
    } else if path.is_dir() {
        if recursive {
            for entry in WalkDir::new(&path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "txt" || ext == "md" {
                        files.push(path.to_path_buf());
                    }
                }
            }
        } else {
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "txt" || ext == "md" {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    if files.is_empty() {
        ui::print_warning("No .txt or .md files found");
        return Ok(());
    }

    ui::print_info(&format!("Found {} files to import", files.len()));

    let pb = ui::create_progress_bar(files.len() as u64, "Processing files");

    let mut items = Vec::new();

    for file_path in &files {
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        let doc_id = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(config::generate_doc_id);

        items.push(MemoryItem {
            content,
            context: context.clone(),
            metadata: None,
            timestamp: None,
            document_id: Some(doc_id),
        });

        pb.inc(1);
    }

    pb.finish_with_message("Files processed");

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Retaining memories..."))
    } else {
        None
    };

    let request = RetainRequest {
        items,
        async_: r#async,
    };

    let response = client.retain(agent_id, &request, r#async, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success("Files retained successfully");
                if result.is_async {
                    println!("  Status: queued for background processing");
                    println!("  Items: {}", result.items_count);
                } else {
                    println!("  Total units created: {}", result.items_count);
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
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
