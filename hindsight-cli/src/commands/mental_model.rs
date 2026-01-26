//! Mental model commands for managing user-curated summaries.

use anyhow::Result;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

use hindsight_client::types;

/// List mental models for a bank
pub fn list(
    client: &ApiClient,
    bank_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching mental models..."))
    } else {
        None
    };

    let response = client.list_mental_models(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Mental Models: {}", bank_id));

                if result.items.is_empty() {
                    println!("  {}", ui::dim("No mental models found."));
                } else {
                    for mental_model in &result.items {
                        println!(
                            "  {} {}",
                            ui::gradient_start(&mental_model.id),
                            mental_model.name
                        );

                        // Show content preview
                        let preview: String = mental_model.content.chars().take(80).collect();
                        let ellipsis = if mental_model.content.len() > 80 { "..." } else { "" };
                        println!("    {}{}", ui::dim(&preview), ellipsis);

                        println!();
                    }
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Get a specific mental model
pub fn get(
    client: &ApiClient,
    bank_id: &str,
    mental_model_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching mental model..."))
    } else {
        None
    };

    let response = client.get_mental_model(bank_id, mental_model_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(mental_model) => {
            if output_format == OutputFormat::Pretty {
                print_mental_model_detail(&mental_model);
            } else {
                output::print_output(&mental_model, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Create a new mental model
pub fn create(
    client: &ApiClient,
    bank_id: &str,
    name: &str,
    source_query: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating mental model..."))
    } else {
        None
    };

    let request = types::CreateMentalModelRequest {
        name: name.to_string(),
        source_query: source_query.to_string(),
        max_tokens: 2048,
        tags: vec![],
    };

    let response = client.create_mental_model(bank_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Mental model created, operation_id: {}", result.operation_id));
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Update a mental model
pub fn update(
    client: &ApiClient,
    bank_id: &str,
    mental_model_id: &str,
    name: Option<String>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if name.is_none() {
        anyhow::bail!("--name must be provided");
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating mental model..."))
    } else {
        None
    };

    let request = types::UpdateMentalModelRequest { name };

    let response = client.update_mental_model(bank_id, mental_model_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(mental_model) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Mental model '{}' updated successfully", mental_model_id));
                println!();
                print_mental_model_detail(&mental_model);
            } else {
                output::print_output(&mental_model, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Delete a mental model
pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    mental_model_id: &str,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // Confirmation prompt unless -y flag is used
    if !yes && output_format == OutputFormat::Pretty {
        let message = format!(
            "Are you sure you want to delete mental model '{}'? This cannot be undone.",
            mental_model_id
        );

        let confirmed = ui::prompt_confirmation(&message)?;

        if !confirmed {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting mental model..."))
    } else {
        None
    };

    let response = client.delete_mental_model(bank_id, mental_model_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(_) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Mental model '{}' deleted successfully", mental_model_id));
            } else {
                println!("{{\"success\": true}}");
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Refresh a mental model
pub fn refresh(
    client: &ApiClient,
    bank_id: &str,
    mental_model_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Submitting mental model refresh..."))
    } else {
        None
    };

    let response = client.refresh_mental_model(bank_id, mental_model_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(operation) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!(
                    "Mental model refresh submitted. Operation ID: {}",
                    operation.operation_id
                ));
                println!("  {} {}", ui::dim("Status:"), operation.status);
                println!();
                println!("{}", ui::dim("Use 'hindsight operations get' to check the operation status."));
            } else {
                output::print_output(&operation, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// Helper function to print mental model details
fn print_mental_model_detail(mental_model: &types::MentalModelResponse) {
    ui::print_section_header(&mental_model.name);

    println!("  {} {}", ui::dim("ID:"), ui::gradient_start(&mental_model.id));
    println!("  {} {}", ui::dim("Source Query:"), &mental_model.source_query);

    println!();
    println!("{}", ui::gradient_text("─── Content ───"));
    println!();
    println!("{}", &mental_model.content);
    println!();
}
