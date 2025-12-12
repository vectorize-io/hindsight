use anyhow::Result;
use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

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
                    ui::print_info(&format!("Found {} operation(s)", ops_response.operations.len()));
                    for op in &ops_response.operations {
                        println!("\n  Operation ID: {}", op.id);
                        println!("    Type: {}", op.task_type);
                        println!("    Status: {}", op.status);
                        println!("    Items: {}", op.items_count);
                        if let Some(doc_id) = &op.document_id {
                            println!("    Document ID: {}", doc_id);
                        }
                    }
                }
            } else {
                output::print_output(&ops_response, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
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
        Err(e) => Err(e)
    }
}
