//! Webhook commands for managing event delivery hooks.

use anyhow::Result;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

use hindsight_client::types;

/// List webhooks for a bank
pub fn list(
    client: &ApiClient,
    bank_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching webhooks..."))
    } else {
        None
    };

    let response = client.list_webhooks(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_section_header(&format!("Webhooks: {}", bank_id));
        if result.items.is_empty() {
            println!("  {}", ui::dim("No webhooks configured."));
        } else {
            for wh in &result.items {
                let status = if wh.enabled {
                    ui::gradient_start("enabled")
                } else {
                    ui::dim("disabled")
                };
                println!("  {} [{}] {}", ui::gradient_start(&wh.id), status, wh.url);
                if !wh.event_types.is_empty() {
                    println!("    events: {}", wh.event_types.join(", "));
                }
                println!();
            }
            // The listing is paged server-side; say so rather than quietly
            // showing the first page as if it were everything.
            let shown = result.items.len() as i64;
            if shown < result.total {
                println!(
                    "  {}",
                    ui::dim(&format!("Showing {} of {} webhooks.", shown, result.total))
                );
            }
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}

/// Create a new webhook
#[allow(clippy::too_many_arguments)]
pub fn create(
    client: &ApiClient,
    bank_id: &str,
    url: &str,
    event_types: Vec<String>,
    enabled: bool,
    secret: Option<String>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating webhook..."))
    } else {
        None
    };

    let effective_events = if event_types.is_empty() {
        vec!["consolidation.completed".to_string()]
    } else {
        event_types
    };

    let request = types::CreateWebhookRequest {
        enabled,
        event_types: effective_events,
        http_config: None,
        secret,
        url: url.to_string(),
    };

    let response = client.create_webhook(bank_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let wh = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_success(&format!("Webhook '{}' created", wh.id));
        println!("  URL: {}", wh.url);
        println!("  Events: {}", wh.event_types.join(", "));
    } else {
        output::print_output(&wh, output_format)?;
    }
    Ok(())
}

/// Update a webhook
#[allow(clippy::too_many_arguments)]
pub fn update(
    client: &ApiClient,
    bank_id: &str,
    webhook_id: &str,
    url: Option<String>,
    event_types: Option<Vec<String>>,
    enabled: Option<bool>,
    secret: Option<String>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if url.is_none() && event_types.is_none() && enabled.is_none() && secret.is_none() {
        anyhow::bail!(
            "At least one of --url, --event-types, --enabled, or --secret must be provided"
        );
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating webhook..."))
    } else {
        None
    };

    let request = types::UpdateWebhookRequest {
        enabled,
        event_types,
        http_config: None,
        secret,
        url,
    };

    let response = client.update_webhook(bank_id, webhook_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let wh = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_success(&format!("Webhook '{}' updated", wh.id));
    } else {
        output::print_output(&wh, output_format)?;
    }
    Ok(())
}

/// Delete a webhook
pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    webhook_id: &str,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if !yes && output_format == OutputFormat::Pretty {
        let message = format!(
            "Are you sure you want to delete webhook '{}'? This cannot be undone.",
            webhook_id
        );
        if !ui::prompt_confirmation(&message)? {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting webhook..."))
    } else {
        None
    };

    let response = client.delete_webhook(bank_id, webhook_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        if result.success {
            ui::print_success(&format!("Webhook '{}' deleted", webhook_id));
        } else {
            ui::print_error("Failed to delete webhook");
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}

/// List recent delivery attempts for a webhook
pub fn deliveries(
    client: &ApiClient,
    bank_id: &str,
    webhook_id: &str,
    cursor: Option<String>,
    limit: Option<i64>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching deliveries..."))
    } else {
        None
    };

    let response =
        client.list_webhook_deliveries(bank_id, webhook_id, cursor.as_deref(), limit, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_section_header(&format!("Deliveries for {}", webhook_id));
        if result.items.is_empty() {
            println!("  {}", ui::dim("No delivery attempts recorded."));
        } else {
            for d in &result.items {
                println!(
                    "  {} [{}] {} — attempts: {}",
                    ui::gradient_start(&d.id),
                    d.event_type,
                    d.last_response_status
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    d.attempts
                );
                if let Some(err) = &d.last_error {
                    println!("    {} {}", ui::dim("error:"), err);
                }
            }
            if let Some(cursor) = &result.next_cursor {
                println!();
                println!("  {} {}", ui::dim("next cursor:"), cursor);
            }
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}
