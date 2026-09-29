//! Audit log commands.

use anyhow::Result;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

/// List audit log entries for a bank
#[allow(clippy::too_many_arguments)]
pub fn list(
    client: &ApiClient,
    bank_id: &str,
    action: Option<String>,
    transport: Option<String>,
    start_date: Option<String>,
    end_date: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching audit logs..."))
    } else {
        None
    };

    let response = client.list_audit_logs(
        bank_id,
        action.as_deref(),
        transport.as_deref(),
        start_date.as_deref(),
        end_date.as_deref(),
        limit,
        offset,
        verbose,
    );

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_section_header(&format!("Audit logs: {}", bank_id));
        println!(
            "  {} {} ({} total)",
            ui::dim("Showing:"),
            result.items.len(),
            result.total
        );
        println!();
        if result.items.is_empty() {
            println!("  {}", ui::dim("No audit log entries."));
        } else {
            for entry in &result.items {
                let started = entry.started_at.as_deref().unwrap_or("-");
                let duration = entry
                    .duration_ms
                    .map(|d| format!("{}ms", d))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "  {} {} [{}] {}",
                    ui::dim(started),
                    ui::gradient_start(&entry.action),
                    entry.transport,
                    duration
                );
            }
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}

/// Get audit log statistics for a bank
pub fn stats(
    client: &ApiClient,
    bank_id: &str,
    action: Option<String>,
    period: Option<String>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching audit log stats..."))
    } else {
        None
    };

    let response = client.audit_log_stats(bank_id, action.as_deref(), period.as_deref(), verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    let result = response?;
    if output_format == OutputFormat::Pretty {
        ui::print_section_header(&format!("Audit stats: {}", bank_id));
        println!("  {} {}", ui::dim("Period:"), result.period);
        println!("  {} {}", ui::dim("Start:"), result.start);
        println!("  {} {}", ui::dim("Bucket:"), result.trunc);
        println!();
        if result.buckets.is_empty() {
            println!("  {}", ui::dim("No activity in this period."));
        } else {
            for bucket in &result.buckets {
                let json = serde_json::to_value(bucket)?;
                println!("  {}", json);
            }
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}
