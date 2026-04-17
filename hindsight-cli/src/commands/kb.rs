//! Knowledge base commands for managing knowledge bases within a bank.

use anyhow::Result;

use crate::api::ApiClient;
use crate::api::{
    CreateKnowledgeBaseRequest, KnowledgeBase, KnowledgeBaseListResponse,
    UpdateKnowledgeBaseRequest,
};
use crate::output::{self, OutputFormat};
use crate::ui;

/// List knowledge bases for a bank
pub fn list(
    client: &ApiClient,
    bank_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching knowledge bases..."))
    } else {
        None
    };

    let response = client.list_knowledge_bases(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Knowledge Bases: {}", bank_id));

                if result.items.is_empty() {
                    println!("  {}", ui::dim("No knowledge bases found."));
                } else {
                    for kb in &result.items {
                        println!("  {} {}", ui::gradient_start(&kb.id), kb.name);

                        if !kb.mission.is_empty() {
                            let preview: String = kb.mission.chars().take(80).collect();
                            let ellipsis = if kb.mission.len() > 80 { "..." } else { "" };
                            println!("    {}{}", ui::dim(&preview), ellipsis);
                        }

                        if !kb.tags.is_empty() {
                            println!("    {} {}", ui::dim("Tags:"), kb.tags.join(", "));
                        }

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

/// Get a specific knowledge base
pub fn get(
    client: &ApiClient,
    bank_id: &str,
    kb_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching knowledge base..."))
    } else {
        None
    };

    let response = client.get_knowledge_base(bank_id, kb_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(kb) => {
            if output_format == OutputFormat::Pretty {
                print_kb_detail(&kb);
            } else {
                output::print_output(&kb, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Create a new knowledge base
#[allow(clippy::too_many_arguments)]
pub fn create(
    client: &ApiClient,
    bank_id: &str,
    kb_id: &str,
    name: &str,
    mission: &str,
    tags: Vec<String>,
    auto_create: bool,
    split_threshold: i32,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating knowledge base..."))
    } else {
        None
    };

    let request = CreateKnowledgeBaseRequest {
        id: kb_id.to_string(),
        name: name.to_string(),
        mission: mission.to_string(),
        tags,
        auto_create,
        split_threshold,
    };

    let response = client.create_knowledge_base(bank_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(kb) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Knowledge base '{}' created successfully", kb.id));
                println!();
                print_kb_detail(&kb);
            } else {
                output::print_output(&kb, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Update a knowledge base
#[allow(clippy::too_many_arguments)]
pub fn update(
    client: &ApiClient,
    bank_id: &str,
    kb_id: &str,
    name: Option<String>,
    mission: Option<String>,
    tags: Option<Vec<String>>,
    auto_create: Option<bool>,
    split_threshold: Option<i32>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if name.is_none()
        && mission.is_none()
        && tags.is_none()
        && auto_create.is_none()
        && split_threshold.is_none()
    {
        anyhow::bail!(
            "At least one of --name, --mission, --tags, --auto-create, or \
             --split-threshold must be provided"
        );
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating knowledge base..."))
    } else {
        None
    };

    let request = UpdateKnowledgeBaseRequest {
        name,
        mission,
        tags,
        auto_create,
        split_threshold,
    };

    let response = client.update_knowledge_base(bank_id, kb_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(kb) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!(
                    "Knowledge base '{}' updated successfully",
                    kb_id
                ));
                println!();
                print_kb_detail(&kb);
            } else {
                output::print_output(&kb, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Delete a knowledge base
pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    kb_id: &str,
    delete_mental_models: bool,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // Confirmation prompt unless -y flag is used
    if !yes && output_format == OutputFormat::Pretty {
        let message = format!(
            "Are you sure you want to delete knowledge base '{}'? This cannot be undone.",
            kb_id
        );

        let confirmed = ui::prompt_confirmation(&message)?;

        if !confirmed {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting knowledge base..."))
    } else {
        None
    };

    let response = client.delete_knowledge_base(bank_id, kb_id, delete_mental_models, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(_) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!(
                    "Knowledge base '{}' deleted successfully",
                    kb_id
                ));
            } else {
                println!("{{\"success\": true}}");
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

// Helper function to print knowledge base details
fn print_kb_detail(kb: &KnowledgeBase) {
    ui::print_section_header(&kb.name);

    println!("  {} {}", ui::dim("ID:"), ui::gradient_start(&kb.id));
    println!("  {} {}", ui::dim("Bank:"), kb.bank_id);

    if !kb.mission.is_empty() {
        println!();
        println!("{}", ui::gradient_text("─── Mission ───"));
        println!();
        println!("{}", kb.mission);
        println!();
    }

    if !kb.tags.is_empty() {
        println!("  {} {}", ui::dim("Tags:"), kb.tags.join(", "));
    }

    println!("  {} {}", ui::dim("Auto-create:"), kb.auto_create);
    println!("  {} {}", ui::dim("Split threshold:"), kb.split_threshold);

    if let Some(ref created_at) = kb.created_at {
        println!("  {} {}", ui::dim("Created:"), created_at);
    }
    if let Some(ref updated_at) = kb.updated_at {
        println!("  {} {}", ui::dim("Updated:"), updated_at);
    }

    if let Some(count) = kb.mental_model_count {
        println!("  {} {}", ui::dim("Mental models:"), count);
    }

    if let Some(ref models) = kb.mental_models {
        if !models.is_empty() {
            println!();
            println!("{}", ui::gradient_text("─── Mental Models ───"));
            for m in models {
                println!("  {} {}", ui::gradient_start(&m.id), m.name);
            }
        }
    }
}
