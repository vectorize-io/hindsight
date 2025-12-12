use anyhow::Result;
use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

pub fn list(client: &ApiClient, verbose: bool, output_format: OutputFormat) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching banks..."))
    } else {
        None
    };

    let response = client.list_agents(verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(banks_list) => {
            if output_format == OutputFormat::Pretty {
                if banks_list.is_empty() {
                    ui::print_warning("No banks found");
                } else {
                    ui::print_info(&format!("Found {} bank(s)", banks_list.len()));
                    for bank in &banks_list {
                        println!("  - {}", bank.bank_id);
                    }
                }
            } else {
                output::print_output(&banks_list, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn disposition(client: &ApiClient, bank_id: &str, verbose: bool, output_format: OutputFormat) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching disposition..."))
    } else {
        None
    };

    let response = client.get_profile(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(profile) => {
            if output_format == OutputFormat::Pretty {
                ui::print_disposition(&profile);
            } else {
                output::print_output(&profile, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn stats(client: &ApiClient, bank_id: &str, verbose: bool, output_format: OutputFormat) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching statistics..."))
    } else {
        None
    };

    let response = client.get_stats(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(stats) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Statistics: {}", bank_id));

                println!("  {} {}", ui::dim("memory units:"), ui::gradient_start(&stats.total_nodes.to_string()));
                println!("  {} {}", ui::dim("links:"), ui::gradient_mid(&stats.total_links.to_string()));
                println!("  {} {}", ui::dim("documents:"), ui::gradient_end(&stats.total_documents.to_string()));
                println!();

                println!("{}", ui::gradient_text("─── Memory Units by Type ───"));
                let mut fact_types: Vec<_> = stats.nodes_by_fact_type.iter().collect();
                fact_types.sort_by_key(|(k, _)| *k);
                for (i, (fact_type, count)) in fact_types.iter().enumerate() {
                    let t = i as f32 / fact_types.len().max(1) as f32;
                    println!("  {:<10} {}", fact_type, ui::gradient(&count.to_string(), t));
                }
                println!();

                println!("{}", ui::gradient_text("─── Links by Type ───"));
                let mut link_types: Vec<_> = stats.links_by_link_type.iter().collect();
                link_types.sort_by_key(|(k, _)| *k);
                for (i, (link_type, count)) in link_types.iter().enumerate() {
                    let t = i as f32 / link_types.len().max(1) as f32;
                    println!("  {:<10} {}", link_type, ui::gradient(&count.to_string(), t));
                }
                println!();

                println!("{}", ui::gradient_text("─── Links by Fact Type ───"));
                let mut fact_type_links: Vec<_> = stats.links_by_fact_type.iter().collect();
                fact_type_links.sort_by_key(|(k, _)| *k);
                for (i, (fact_type, count)) in fact_type_links.iter().enumerate() {
                    let t = i as f32 / fact_type_links.len().max(1) as f32;
                    println!("  {:<10} {}", fact_type, ui::gradient(&count.to_string(), t));
                }
                println!();

                if !stats.links_breakdown.is_empty() {
                    println!("{}", ui::gradient_text("─── Detailed Link Breakdown ───"));
                    let mut fact_types: Vec<_> = stats.links_breakdown.iter().collect();
                    fact_types.sort_by_key(|(k, _)| *k);
                    for (fact_type, link_types) in fact_types {
                        println!("  {}", fact_type);
                        let mut sorted_links: Vec<_> = link_types.iter().collect();
                        sorted_links.sort_by_key(|(k, _)| *k);
                        for (link_type, count) in sorted_links {
                            println!("    {:<10} {}", ui::dim(link_type), count);
                        }
                    }
                    println!();
                }

                if stats.pending_operations > 0 || stats.failed_operations > 0 {
                    println!("{}", ui::gradient_text("─── Operations ───"));
                    if stats.pending_operations > 0 {
                        println!("  {} {}", ui::dim("pending:"), stats.pending_operations);
                    }
                    if stats.failed_operations > 0 {
                        println!("  {} {}", ui::dim("failed:"), stats.failed_operations);
                    }
                }
            } else {
                output::print_output(&stats, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn update_name(client: &ApiClient, bank_id: &str, name: &str, verbose: bool, output_format: OutputFormat) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating bank name..."))
    } else {
        None
    };

    let response = client.update_agent_name(bank_id, name, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(profile) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Bank name updated to '{}'", profile.name));
            } else {
                output::print_output(&profile, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn update_background(
    client: &ApiClient,
    bank_id: &str,
    content: &str,
    no_update_disposition: bool,
    verbose: bool,
    output_format: OutputFormat
) -> Result<()> {
    let current_profile = if !no_update_disposition {
        client.get_profile(bank_id, verbose).ok()
    } else {
        None
    };

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Merging background..."))
    } else {
        None
    };

    let response = client.add_background(bank_id, content, !no_update_disposition, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(profile) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success("Background updated successfully");
                println!("\n{}", profile.background);

                if !no_update_disposition {
                    if let (Some(old_p), Some(new_p)) =
                        (current_profile.as_ref().map(|p| p.disposition.clone()), &profile.disposition)
                    {
                        println!("\nDisposition changes:");
                        println!("  Skepticism:  {} → {}", old_p.skepticism, new_p.skepticism);
                        println!("  Literalism:  {} → {}", old_p.literalism, new_p.literalism);
                        println!("  Empathy:     {} → {}", old_p.empathy, new_p.empathy);
                    }
                }
            } else {
                output::print_output(&profile, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}

pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat
) -> Result<()> {
    // Confirmation prompt unless -y flag is used
    if !yes && output_format == OutputFormat::Pretty {
        let message = format!(
            "Are you sure you want to delete bank '{}' and ALL its data? This cannot be undone.",
            bank_id
        );

        let confirmed = ui::prompt_confirmation(&message)?;

        if !confirmed {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting bank..."))
    } else {
        None
    };

    let response = client.delete_bank(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                if result.success {
                    ui::print_success(&format!("Bank '{}' deleted successfully", bank_id));
                    if let Some(count) = result.deleted_count {
                        println!("  Items deleted: {}", count);
                    }
                } else {
                    ui::print_error("Failed to delete bank");
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e)
    }
}
