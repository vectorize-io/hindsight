use anyhow::Result;
use chrono::{Duration as ChronoDuration, NaiveDate, Utc};
use std::collections::BTreeMap;
use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

pub fn list(
    client: &ApiClient,
    agent_id: &str,
    query: Option<String>,
    date: Option<String>,
    limit: i32,
    offset: i32,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // If date filter is provided, use the date-aware listing
    if date.is_some() {
        return list_with_date(client, agent_id, date.as_deref(), verbose, output_format);
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching documents..."))
    } else {
        None
    };

    let response = client.list_documents(agent_id, query.as_deref(), Some(limit), Some(offset), verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(docs_response) => {
            if output_format == OutputFormat::Pretty {
                ui::print_info(&format!("Documents for bank '{}' (total: {})", agent_id, docs_response.total));
                for doc in &docs_response.items {
                    let id = doc.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let created = doc.get("created_at").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let updated = doc.get("updated_at").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let text_len = doc.get("text_length").and_then(|v| v.as_i64()).unwrap_or(0);
                    let mem_count = doc.get("memory_unit_count").and_then(|v| v.as_i64()).unwrap_or(0);

                    println!("\n  Document ID: {}", id);
                    println!("    Created: {}", created);
                    println!("    Updated: {}", updated);
                    println!("    Text Length: {}", text_len);
                    println!("    Memory Units: {}", mem_count);
                }
            } else {
                output::print_output(&docs_response, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

/// List documents with date filtering
fn list_with_date(
    client: &ApiClient,
    bank_id: &str,
    date_filter: Option<&str>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching all documents..."))
    } else {
        None
    };

    // Fetch all documents with pagination
    let all_docs = fetch_all_documents(client, bank_id, verbose)?;

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    // Parse the date filter
    let target_date = parse_date_filter(date_filter)?;

    // Filter and group documents by date
    let mut by_date: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
    let mut filtered_count = 0;

    for doc in all_docs {
        let created_at = doc.get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Parse the date part (YYYY-MM-DD) from created_at
        let doc_date = created_at.split('T').next().unwrap_or("");

        // Apply date filter if specified
        if let Some(ref target) = target_date {
            let target_str = target.format("%Y-%m-%d").to_string();
            if doc_date != target_str {
                continue;
            }
        }

        filtered_count += 1;
        by_date.entry(doc_date.to_string()).or_default().push(doc);
    }

    // Output
    if output_format == OutputFormat::Pretty {
        let filter_desc = match date_filter {
            None | Some("yesterday") => "yesterday".to_string(),
            Some("today") => "today".to_string(),
            Some("all") => "all dates".to_string(),
            Some(d) => d.to_string(),
        };

        ui::print_info(&format!(
            "Documents for bank '{}' (filter: {}, showing: {})",
            bank_id, filter_desc, filtered_count
        ));
        println!();

        // Show documents grouped by date (reverse order - newest first)
        for (date_str, docs) in by_date.iter().rev() {
            println!("  {} ({} documents)", date_str, docs.len());
            for doc in docs {
                let id = doc.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                let mem_count = doc.get("memory_unit_count").and_then(|v| v.as_i64()).unwrap_or(0);
                println!("    - {} ({} memories)", id, mem_count);
            }
            println!();
        }
    } else {
        // JSON/YAML output - convert to a list structure
        let output: Vec<serde_json::Value> = by_date.values().flatten().cloned().collect();
        output::print_output(&output, output_format)?;
    }

    Ok(())
}

/// Fetch all documents with pagination
fn fetch_all_documents(
    client: &ApiClient,
    bank_id: &str,
    verbose: bool,
) -> Result<Vec<serde_json::Value>> {
    let mut all_docs = Vec::new();
    let mut offset = 0;
    let limit = 500;

    loop {
        let response = client.list_documents(bank_id, None, Some(limit), Some(offset), verbose)?;

        if response.items.is_empty() {
            break;
        }

        // Convert Map<String, Value> to Value for each item
        for item in response.items {
            all_docs.push(serde_json::Value::Object(item));
        }

        offset += limit;

        // Check if we've fetched everything
        if all_docs.len() >= response.total as usize {
            break;
        }
    }

    Ok(all_docs)
}

/// Parse date filter string into a NaiveDate
fn parse_date_filter(filter: Option<&str>) -> Result<Option<NaiveDate>> {
    match filter {
        None | Some("yesterday") => {
            // Default to yesterday
            Ok(Some(Utc::now().date_naive() - ChronoDuration::days(1)))
        }
        Some("today") => Ok(Some(Utc::now().date_naive())),
        Some("all") => Ok(None), // No filtering
        Some(date_str) => {
            // Try to parse as YYYY-MM-DD
            NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .map(Some)
                .map_err(|e| anyhow::anyhow!("Invalid date format '{}': {}. Use YYYY-MM-DD, 'yesterday', 'today', or 'all'", date_str, e))
        }
    }
}

pub fn get(
    client: &ApiClient,
    agent_id: &str,
    document_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching document..."))
    } else {
        None
    };

    let response = client.get_document(agent_id, document_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(doc) => {
            if output_format == OutputFormat::Pretty {
                ui::print_info(&format!("Document: {}", doc.id));
                println!("  Bank ID: {}", doc.bank_id);
                println!("  Created: {}", doc.created_at);
                println!("  Updated: {}", doc.updated_at);
                println!("  Memory Units: {}", doc.memory_unit_count);
                println!("\n  Text:\n{}", doc.original_text);
            } else {
                output::print_output(&doc, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn delete(
    client: &ApiClient,
    agent_id: &str,
    document_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting document..."))
    } else {
        None
    };

    let response = client.delete_document(agent_id, document_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                if result.success {
                    ui::print_success("Document deleted successfully");
                } else {
                    ui::print_error("Failed to delete document");
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}
