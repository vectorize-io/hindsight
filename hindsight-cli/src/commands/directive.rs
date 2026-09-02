//! Directive commands for managing behavioral rules.

use anyhow::Result;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

use hindsight_client::types;

/// List directives for a bank
pub fn list(
    client: &ApiClient,
    bank_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching directives..."))
    } else {
        None
    };

    let response = client.list_directives(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Directives: {}", bank_id));

                if result.items.is_empty() {
                    println!("  {}", ui::dim("No directives found."));
                } else {
                    for directive in &result.items {
                        let status = if directive.is_active {
                            ui::gradient_start("active")
                        } else {
                            ui::dim("inactive")
                        };
                        println!(
                            "  {} {} [{}]",
                            ui::gradient_start(&directive.id),
                            directive.name,
                            status
                        );

                        // Show content preview
                        let preview: String = directive.content.chars().take(80).collect();
                        let ellipsis = if directive.content.len() > 80 {
                            "..."
                        } else {
                            ""
                        };
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

/// Get a specific directive
pub fn get(
    client: &ApiClient,
    bank_id: &str,
    directive_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching directive..."))
    } else {
        None
    };

    let response = client.get_directive(bank_id, directive_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(directive) => {
            if output_format == OutputFormat::Pretty {
                print_directive_detail(&directive);
            } else {
                output::print_output(&directive, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Create a new directive
#[allow(clippy::too_many_arguments)]
pub fn create(
    client: &ApiClient,
    bank_id: &str,
    name: &str,
    content: &str,
    priority: i64,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating directive..."))
    } else {
        None
    };

    let request = types::CreateDirectiveRequest {
        name: name.to_string(),
        content: content.to_string(),
        is_active: true,
        priority,
        tags: vec![],
    };

    let response = client.create_directive(bank_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(directive) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!(
                    "Directive '{}' created successfully",
                    directive.id
                ));
                println!();
                print_directive_detail(&directive);
            } else {
                output::print_output(&directive, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Update a directive
#[allow(clippy::too_many_arguments)]
pub fn update(
    client: &ApiClient,
    bank_id: &str,
    directive_id: &str,
    name: Option<String>,
    content: Option<String>,
    is_active: Option<bool>,
    priority: Option<i64>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if name.is_none() && content.is_none() && is_active.is_none() && priority.is_none() {
        anyhow::bail!(
            "At least one of --name, --content, --is-active, or --priority must be provided"
        );
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating directive..."))
    } else {
        None
    };

    let request = types::UpdateDirectiveRequest {
        name,
        content,
        is_active,
        priority,
        tags: None,
    };

    let response = client.update_directive(bank_id, directive_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(directive) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!(
                    "Directive '{}' updated successfully",
                    directive_id
                ));
                println!();
                print_directive_detail(&directive);
            } else {
                output::print_output(&directive, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Delete a directive
pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    directive_id: &str,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // Confirmation prompt unless -y flag is used
    if !yes && output_format == OutputFormat::Pretty {
        let message = format!(
            "Are you sure you want to delete directive '{}'? This cannot be undone.",
            directive_id
        );

        let confirmed = ui::prompt_confirmation(&message)?;

        if !confirmed {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting directive..."))
    } else {
        None
    };

    let response = client.delete_directive(bank_id, directive_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(_) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!(
                    "Directive '{}' deleted successfully",
                    directive_id
                ));
            } else {
                println!("{{\"success\": true}}");
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// Helper function to print directive details
fn print_directive_detail(directive: &types::DirectiveResponse) {
    ui::print_section_header(&directive.name);

    println!("  {} {}", ui::dim("ID:"), ui::gradient_start(&directive.id));

    let status = if directive.is_active {
        ui::gradient_start("active")
    } else {
        ui::dim("inactive")
    };
    println!("  {} {}", ui::dim("Status:"), status);
    println!("  {} {}", ui::dim("Priority:"), directive.priority);

    if !directive.tags.is_empty() {
        println!("  {} {}", ui::dim("Tags:"), directive.tags.join(", "));
    }

    println!();
    println!("{}", ui::gradient_text("─── Content ───"));
    println!();
    println!("{}", &directive.content);
    println!();
}
