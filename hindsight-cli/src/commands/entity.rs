use anyhow::Result;
use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

pub fn list(
    client: &ApiClient,
    bank_id: &str,
    limit: i64,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching entities..."))
    } else {
        None
    };

    let response = client.list_entities(bank_id, Some(limit), verbose)?;

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    if output_format == OutputFormat::Pretty {
        ui::print_section_header(&format!("Entities for Bank: {}", bank_id));

        if response.items.is_empty() {
            ui::print_warning("No entities found");
            return Ok(());
        }

        println!("Total entities: {}\n", response.items.len());

        for entity in &response.items {
            println!("ID: {}", entity.id);
            println!("  Name: {}", entity.canonical_name);
            println!("  Mentions: {}", entity.mention_count);
            if let Some(first_seen) = &entity.first_seen {
                println!("  First seen: {}", first_seen);
            }
            if let Some(last_seen) = &entity.last_seen {
                println!("  Last seen: {}", last_seen);
            }
            println!();
        }
    } else {
        output::print_output(&response, output_format)?;
    }

    Ok(())
}

pub fn get(
    client: &ApiClient,
    bank_id: &str,
    entity_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching entity details..."))
    } else {
        None
    };

    let response = client.get_entity(bank_id, entity_id, verbose)?;

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    if output_format == OutputFormat::Pretty {
        ui::print_section_header(&format!("Entity: {}", entity_id));

        println!("ID: {}", response.id);
        println!("Name: {}", response.canonical_name);
        println!("Mentions: {}", response.mention_count);

        if let Some(first_seen) = &response.first_seen {
            println!("First seen: {}", first_seen);
        }
        if let Some(last_seen) = &response.last_seen {
            println!("Last seen: {}", last_seen);
        }

        println!();
    } else {
        output::print_output(&response, output_format)?;
    }

    Ok(())
}

pub fn regenerate(
    client: &ApiClient,
    bank_id: &str,
    entity_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Regenerating entity observations..."))
    } else {
        None
    };

    let response = client.regenerate_entity(bank_id, entity_id, verbose)?;

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    if output_format == OutputFormat::Pretty {
        ui::print_success(&format!("Successfully regenerated observations for entity: {}", entity_id));
        println!("\nUpdated entity:");
        println!("  Name: {}", response.canonical_name);
        println!("  Mentions: {}", response.mention_count);
        println!("  Observations: {}", response.observations.len());
    } else {
        output::print_output(&response, output_format)?;
    }

    Ok(())
}
