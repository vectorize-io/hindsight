use anyhow::Result;
use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

pub fn list(
    client: &ApiClient,
    agent_id: &str,
    query: Option<String>,
    limit: i32,
    offset: i32,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
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
