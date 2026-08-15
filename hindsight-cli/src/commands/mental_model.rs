//! Mental model commands for managing user-curated summaries.

use anyhow::Result;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

use hindsight_client::types;

/// Parse a --tags-match string into the generated TagsMatch enum, rejecting
/// unknown values so a typo fails loudly instead of silently changing scope.
fn parse_tags_match(value: &str) -> Result<types::TagsMatch> {
    Ok(match value.to_lowercase().as_str() {
        "any" => types::TagsMatch::Any,
        "all" => types::TagsMatch::All,
        "any_strict" => types::TagsMatch::AnyStrict,
        "all_strict" => types::TagsMatch::AllStrict,
        "exact" => types::TagsMatch::Exact,
        other => anyhow::bail!(
            "invalid --tags-match '{other}': expected one of any, all, any_strict, all_strict, exact"
        ),
    })
}

/// Parse a --trigger-mode value into the generated refresh-mode enum
fn parse_trigger_mode(value: &str) -> Result<types::Mode> {
    Ok(match value.to_lowercase().as_str() {
        "full" => types::Mode::Full,
        "delta" => types::Mode::Delta,
        other => anyhow::bail!("invalid --trigger-mode '{other}': expected one of full, delta"),
    })
}

/// Build the trigger override sent on update when at least one trigger flag
/// was passed. `None` (no trigger flags) leaves the stored trigger config
/// untouched on the server. When an override IS sent, the server replaces the
/// whole trigger, so fields the user did not specify fall back to their
/// defaults.
fn build_trigger_override(
    trigger_mode: Option<&str>,
    trigger_refresh_after_consolidation: Option<bool>,
) -> Result<Option<types::MentalModelTriggerInput>> {
    if trigger_mode.is_none() && trigger_refresh_after_consolidation.is_none() {
        return Ok(None);
    }
    Ok(Some(types::MentalModelTriggerInput {
        mode: trigger_mode
            .map(parse_trigger_mode)
            .transpose()?
            .unwrap_or(types::Mode::Full),
        refresh_after_consolidation: trigger_refresh_after_consolidation.unwrap_or(false),
        refresh_cron: None,
        exclude_mental_models: false,
        exclude_mental_model_ids: None,
        fact_types: None,
        tag_groups: None,
        tags_match: None,
        include_chunks: None,
        recall_max_tokens: None,
        recall_chunks_max_tokens: None,
        response_schema: None,
        keep_trace: false,
    }))
}

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
                        if let Some(ref content) = mental_model.content {
                            let preview: String = content.chars().take(80).collect();
                            let ellipsis = if content.len() > 80 { "..." } else { "" };
                            println!("    {}{}", ui::dim(&preview), ellipsis);
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
#[allow(clippy::too_many_arguments)]
pub fn create(
    client: &ApiClient,
    bank_id: &str,
    name: &str,
    source_query: &str,
    id: Option<&str>,
    tags: Vec<String>,
    max_tokens: i64,
    tags_match: Option<&str>,
    trigger_refresh_after_consolidation: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating mental model..."))
    } else {
        None
    };

    let tags_match = tags_match.map(parse_tags_match).transpose()?;

    // Only send a trigger when the user opted into one of its fields, so the
    // server's default behaviour (all_strict for tagged models) is preserved
    // otherwise.
    let trigger = if trigger_refresh_after_consolidation || tags_match.is_some() {
        Some(types::MentalModelTriggerInput {
            mode: types::Mode::Full,
            refresh_after_consolidation: trigger_refresh_after_consolidation,
            refresh_cron: None,
            exclude_mental_models: false,
            exclude_mental_model_ids: None,
            fact_types: None,
            tag_groups: None,
            tags_match,
            include_chunks: None,
            recall_max_tokens: None,
            recall_chunks_max_tokens: None,
            response_schema: None,
            keep_trace: false,
        })
    } else {
        None
    };

    let request = types::CreateMentalModelRequest {
        id: id.map(|s| s.to_string()),
        name: name.to_string(),
        source_query: source_query.to_string(),
        max_tokens,
        tags,
        trigger,
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
#[allow(clippy::too_many_arguments)]
pub fn update(
    client: &ApiClient,
    bank_id: &str,
    mental_model_id: &str,
    name: Option<String>,
    source_query: Option<String>,
    max_tokens: Option<i64>,
    tags: Option<Vec<String>>,
    trigger_refresh_after_consolidation: Option<bool>,
    trigger_mode: Option<&str>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if name.is_none()
        && source_query.is_none()
        && max_tokens.is_none()
        && tags.is_none()
        && trigger_refresh_after_consolidation.is_none()
        && trigger_mode.is_none()
    {
        anyhow::bail!(
            "At least one of --name, --source-query, --max-tokens, --tags, \
             --trigger-mode, or --trigger-refresh-after-consolidation must be provided"
        );
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating mental model..."))
    } else {
        None
    };

    // Only build a trigger override when the user actually passed a trigger
    // flag; sending None leaves the existing trigger config untouched on the
    // server. When an override IS sent, the server replaces the whole trigger,
    // so fields the user did not specify fall back to their defaults.
    let trigger = build_trigger_override(trigger_mode, trigger_refresh_after_consolidation)?;

    let request = types::UpdateMentalModelRequest {
        name,
        source_query,
        max_tokens,
        tags,
        trigger,
    };

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

/// Preview a mental model refresh without changing anything.
///
/// Runs the real refresh pipeline and reports what it would do — which mode it
/// ended up in and why, the scope and window it read, how many facts retrieval
/// returned versus how many the agent used, and a diff of the content it would
/// write. Nothing is persisted, so repeating it reads the same window again.
pub fn dry_run_refresh(
    client: &ApiClient,
    bank_id: &str,
    mental_model_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        // The dry run makes the same LLM calls a refresh does, so it is not quick.
        Some(ui::create_spinner("Running refresh dry run..."))
    } else {
        None
    };

    let response = client.dry_run_refresh_mental_model(bank_id, mental_model_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format != OutputFormat::Pretty {
                return output::print_output(&result, output_format);
            }

            println!();
            if result.requested_mode.to_string() == result.effective_mode.to_string() {
                println!("  {} {}", ui::dim("Mode:"), result.effective_mode);
            } else {
                println!(
                    "  {} {} → {}",
                    ui::dim("Mode:"),
                    result.requested_mode,
                    result.effective_mode
                );
            }
            if let Some(reason) = &result.mode_fallback_reason {
                println!("  {} {}", ui::dim("Fell back:"), reason);
            }
            println!("  {} {}", ui::dim("Outcome:"), result.outcome);
            println!("  {} {}", ui::dim("Would save:"), result.would_persist);
            println!(
                "  {} {}",
                ui::dim("Retrieved:"),
                format_counts(&result.facts.retrieved)
            );
            println!("  {} {}", ui::dim("Used:"), format_counts(&result.facts.used));
            if let Some(ops) = &result.delta_operations {
                println!(
                    "  {} {} applied, {} skipped",
                    ui::dim("Delta ops:"),
                    ops.applied.len(),
                    ops.skipped.len()
                );
            }
            println!("  {} {} ms", ui::dim("Duration:"), result.duration_ms);

            for warning in &result.warnings {
                println!();
                ui::print_warning(warning);
            }

            println!();
            if result.diff.is_empty() {
                println!("{}", ui::dim("No content change."));
            } else {
                println!("{}", result.diff);
            }
            println!();
            println!("{}", ui::dim("Nothing was saved. Use 'mental-model refresh' to apply."));
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Render a fact-type → count map as "observation: 12, world: 3".
fn format_counts(counts: &std::collections::HashMap<String, i64>) -> String {
    if counts.is_empty() {
        return "none".to_string();
    }
    let mut entries: Vec<String> = counts.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
    entries.sort();
    entries.join(", ")
}

/// Get the change history of a mental model
pub fn history(
    client: &ApiClient,
    bank_id: &str,
    mental_model_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching mental model history..."))
    } else {
        None
    };

    let response = client.get_mental_model_history(bank_id, mental_model_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(history) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("History: {}", mental_model_id));

                if let Some(entries) = history.as_array() {
                    if entries.is_empty() {
                        println!("  {}", ui::dim("No history entries found."));
                    } else {
                        for entry in entries {
                            let changed_at = entry.get("changed_at").and_then(|v| v.as_str()).unwrap_or("unknown");
                            let previous = entry.get("previous_content").and_then(|v| v.as_str()).unwrap_or("(none)");
                            println!("  {} {}", ui::dim("Changed at:"), changed_at);
                            let preview: String = previous.chars().take(80).collect();
                            let ellipsis = if previous.len() > 80 { "..." } else { "" };
                            println!("  {} {}{}", ui::dim("Previous:"), ui::dim(&preview), ellipsis);
                            println!();
                        }
                    }
                }
            } else {
                output::print_output(&history, output_format)?;
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
    if let Some(ref source_query) = mental_model.source_query {
        println!("  {} {}", ui::dim("Source Query:"), source_query);
    }

    if let Some(ref content) = mental_model.content {
        println!();
        println!("{}", ui::gradient_text("─── Content ───"));
        println!();
        println!("{}", content);
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tags_match_accepts_all_modes() {
        assert_eq!(parse_tags_match("any").unwrap(), types::TagsMatch::Any);
        assert_eq!(parse_tags_match("all").unwrap(), types::TagsMatch::All);
        assert_eq!(parse_tags_match("any_strict").unwrap(), types::TagsMatch::AnyStrict);
        assert_eq!(parse_tags_match("all_strict").unwrap(), types::TagsMatch::AllStrict);
        assert_eq!(parse_tags_match("exact").unwrap(), types::TagsMatch::Exact);
    }

    #[test]
    fn parse_tags_match_is_case_insensitive() {
        assert_eq!(parse_tags_match("ANY").unwrap(), types::TagsMatch::Any);
    }

    #[test]
    fn parse_tags_match_rejects_unknown() {
        let err = parse_tags_match("most").unwrap_err().to_string();
        assert!(err.contains("invalid --tags-match 'most'"), "unexpected error: {err}");
    }

    #[test]
    fn parse_trigger_mode_accepts_both_modes() {
        assert_eq!(parse_trigger_mode("full").unwrap(), types::Mode::Full);
        assert_eq!(parse_trigger_mode("delta").unwrap(), types::Mode::Delta);
    }

    #[test]
    fn parse_trigger_mode_is_case_insensitive() {
        assert_eq!(parse_trigger_mode("DELTA").unwrap(), types::Mode::Delta);
    }

    #[test]
    fn parse_trigger_mode_rejects_unknown() {
        let err = parse_trigger_mode("incremental").unwrap_err().to_string();
        assert!(
            err.contains("invalid --trigger-mode 'incremental'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn build_trigger_override_none_when_no_flags() {
        assert!(build_trigger_override(None, None).unwrap().is_none());
    }

    #[test]
    fn build_trigger_override_mode_only() {
        let trigger = build_trigger_override(Some("delta"), None)
            .unwrap()
            .unwrap();
        assert_eq!(trigger.mode, types::Mode::Delta);
        assert!(!trigger.refresh_after_consolidation);
    }

    #[test]
    fn build_trigger_override_refresh_only_defaults_to_full_mode() {
        let trigger = build_trigger_override(None, Some(true)).unwrap().unwrap();
        assert_eq!(trigger.mode, types::Mode::Full);
        assert!(trigger.refresh_after_consolidation);

        // Explicit false must stay distinguishable from omission.
        let trigger = build_trigger_override(None, Some(false)).unwrap().unwrap();
        assert_eq!(trigger.mode, types::Mode::Full);
        assert!(!trigger.refresh_after_consolidation);
    }

    #[test]
    fn build_trigger_override_both_flags() {
        let trigger = build_trigger_override(Some("FULL"), Some(true))
            .unwrap()
            .unwrap();
        assert_eq!(trigger.mode, types::Mode::Full);
        assert!(trigger.refresh_after_consolidation);
    }

    #[test]
    fn build_trigger_override_propagates_parse_error() {
        let err = build_trigger_override(Some("incremental"), None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("invalid --trigger-mode 'incremental'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn update_request_serializes_delta_trigger() {
        let request = types::UpdateMentalModelRequest {
            name: None,
            source_query: None,
            max_tokens: None,
            tags: None,
            trigger: build_trigger_override(Some("delta"), Some(true)).unwrap(),
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["trigger"]["mode"], "delta");
        assert_eq!(value["trigger"]["refresh_after_consolidation"], true);
    }

    #[test]
    fn update_request_omits_trigger_when_no_flags() {
        let request = types::UpdateMentalModelRequest {
            name: Some("renamed".to_string()),
            source_query: None,
            max_tokens: None,
            tags: None,
            trigger: build_trigger_override(None, None).unwrap(),
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["name"], "renamed");
        assert!(value.get("trigger").is_none() || value["trigger"].is_null());
    }
}
