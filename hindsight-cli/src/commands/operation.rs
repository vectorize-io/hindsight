use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;
use anyhow::Result;

fn operation_summary_lines(op: &crate::api::Operation) -> Vec<String> {
    let mut lines = vec![
        format!("\n  Operation ID: {}", op.id),
        format!("    Type: {}", op.task_type),
        format!("    Status: {}", op.status),
        format!("    Items: {}", op.items_count),
    ];
    if let Some(filename) = &op.filename {
        lines.push(format!("    Filename: {}", filename));
    }
    if let Some(doc_id) = &op.document_id {
        lines.push(format!("    Document ID: {}", doc_id));
    }
    lines
}

pub fn list(
    client: &ApiClient,
    agent_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching operations..."))
    } else {
        None
    };

    let response = client.list_operations(agent_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(ops_response) => {
            if output_format == OutputFormat::Pretty {
                if ops_response.operations.is_empty() {
                    ui::print_info("No operations found");
                } else {
                    ui::print_info(&format!(
                        "Found {} operation(s)",
                        ops_response.operations.len()
                    ));
                    for op in &ops_response.operations {
                        for line in operation_summary_lines(op) {
                            println!("{line}");
                        }
                    }
                }
            } else {
                output::print_output(&ops_response, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::operation_summary_lines;
    use crate::api::Operation;

    fn operation(filename: Option<&str>) -> Operation {
        Operation {
            id: "op-1".to_string(),
            task_type: "retain".to_string(),
            items_count: 2,
            filename: filename.map(str::to_string),
            document_id: Some("doc-1".to_string()),
            created_at: "2024-01-15T10:00:00Z".to_string(),
            status: "completed".to_string(),
            error_message: None,
        }
    }

    #[test]
    fn summary_lines_include_filename_when_present() {
        let lines = operation_summary_lines(&operation(Some("notes.md")));
        assert!(lines.iter().any(|line| line == "    Filename: notes.md"));
        assert!(lines.iter().any(|line| line == "    Document ID: doc-1"));
    }

    #[test]
    fn summary_lines_skip_filename_when_absent() {
        let lines = operation_summary_lines(&operation(None));
        assert!(!lines.iter().any(|line| line.starts_with("    Filename:")));
        assert!(lines.iter().any(|line| line == "    Document ID: doc-1"));
    }
}

/// Get the status of a specific operation
pub fn get(
    client: &ApiClient,
    agent_id: &str,
    operation_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching operation status..."))
    } else {
        None
    };

    let response = client.get_operation(agent_id, operation_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Operation: {}", operation_id));

                use hindsight_client::types::Status;
                let status_str = match &result.status {
                    Status::Completed => ui::gradient_start("completed"),
                    Status::Pending => ui::gradient_mid("pending"),
                    Status::Processing => ui::gradient_mid("processing"),
                    Status::Failed => ui::gradient_end("failed"),
                    Status::Cancelled => ui::dim("cancelled"),
                    Status::NotFound => ui::gradient_end("not_found"),
                };

                println!("  {} {}", ui::dim("Status:"), status_str);

                if let Some(error) = &result.error_message {
                    println!("  {} {}", ui::dim("Error:"), ui::gradient_end(error));
                }

                println!();
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

pub fn cancel(
    client: &ApiClient,
    agent_id: &str,
    operation_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Cancelling operation..."))
    } else {
        None
    };

    let response = client.cancel_operation(agent_id, operation_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                if result.success {
                    ui::print_success("Operation cancelled successfully");
                } else {
                    ui::print_error("Failed to cancel operation");
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Retry a failed async operation
pub fn retry(
    client: &ApiClient,
    agent_id: &str,
    operation_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Retrying operation..."))
    } else {
        None
    };

    let response = client.retry_operation(agent_id, operation_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_success(&format!("Operation '{}' retried", operation_id));
        let json = serde_json::to_value(&result)?;
        println!(
            "  {}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}

/// Permanently delete a terminal async operation
pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    operation_id: &str,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if !yes && output_format == OutputFormat::Pretty {
        let message = format!(
            "Are you sure you want to permanently delete operation '{}'? This cannot be undone.",
            operation_id
        );
        if !ui::prompt_confirmation(&message)? {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting operation..."))
    } else {
        None
    };

    let response = client.delete_operation(bank_id, operation_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        if result.success {
            ui::print_success(&result.message);
        } else {
            ui::print_error(&result.message);
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}
