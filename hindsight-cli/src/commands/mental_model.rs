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

/// The trigger settings `mental-model update` can change, exactly as the user
/// passed them. `None` means "not passed": that setting keeps whatever the
/// server has stored.
#[derive(Debug, Default)]
pub struct TriggerUpdate {
    pub mode: Option<String>,
    pub refresh_after_consolidation: Option<bool>,
    pub refresh_cron: Option<String>,
    pub min_refresh_interval_seconds: Option<u64>,
    pub tags_match: Option<String>,
    pub keep_trace: Option<bool>,
    pub exclude_mental_models: Option<bool>,
}

impl TriggerUpdate {
    /// True when the user passed no trigger flag at all, so the update must not
    /// send a trigger and the server keeps the stored one untouched.
    fn is_empty(&self) -> bool {
        self.mode.is_none()
            && self.refresh_after_consolidation.is_none()
            && self.refresh_cron.is_none()
            && self.min_refresh_interval_seconds.is_none()
            && self.tags_match.is_none()
            && self.keep_trace.is_none()
            && self.exclude_mental_models.is_none()
    }
}

/// A trigger carrying nothing but the API's own defaults.
///
/// Used as the base when a mental model has no stored trigger yet, so the
/// user's flags land on the same values the server would have applied.
fn default_trigger_input() -> types::MentalModelTriggerInput {
    types::MentalModelTriggerInput {
        mode: types::Mode::Full,
        refresh_after_consolidation: false,
        refresh_cron: None,
        min_refresh_interval_seconds: None,
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
    }
}

/// Re-shape the trigger the server reports into the input type an update sends.
///
/// The two are the same field set; they are distinct generated types only
/// because the OpenAPI schema splits request and response models.
fn stored_trigger_as_input(
    stored: &types::MentalModelTriggerOutput,
) -> types::MentalModelTriggerInput {
    types::MentalModelTriggerInput {
        mode: stored.mode,
        refresh_after_consolidation: stored.refresh_after_consolidation,
        refresh_cron: stored.refresh_cron.clone(),
        min_refresh_interval_seconds: stored.min_refresh_interval_seconds,
        exclude_mental_models: stored.exclude_mental_models,
        exclude_mental_model_ids: stored.exclude_mental_model_ids.clone(),
        fact_types: stored.fact_types.clone(),
        tag_groups: stored.tag_groups.clone(),
        tags_match: stored.tags_match,
        include_chunks: stored.include_chunks,
        recall_max_tokens: stored.recall_max_tokens,
        recall_chunks_max_tokens: stored.recall_chunks_max_tokens,
        response_schema: stored.response_schema.clone(),
        keep_trace: stored.keep_trace,
    }
}

/// Apply the flags the user passed on top of the mental model's current trigger.
///
/// The server patches a supplied trigger over the stored one, but it can only
/// see which fields were *sent*, and the generated input type serializes its
/// non-`Option` fields (`mode`, `refresh_after_consolidation`,
/// `exclude_mental_models`, `keep_trace`) unconditionally. Sending a
/// freshly-built trigger therefore carried this type's own defaults into every
/// update and silently reset those four — switching a model to delta mode, say,
/// also turned off its trace capture. Starting from the stored trigger keeps
/// every field the user did not name.
///
/// An empty string clears `--trigger-refresh-cron` / `--trigger-tags-match`
/// back to "unset", which is how a cron schedule is removed.
fn apply_trigger_update(
    base: types::MentalModelTriggerInput,
    update: &TriggerUpdate,
) -> Result<types::MentalModelTriggerInput> {
    let mut trigger = base;

    if let Some(mode) = update.mode.as_deref() {
        trigger.mode = parse_trigger_mode(mode)?;
    }
    if let Some(refresh) = update.refresh_after_consolidation {
        trigger.refresh_after_consolidation = refresh;
    }
    if let Some(cron) = update.refresh_cron.as_deref() {
        trigger.refresh_cron = if cron.trim().is_empty() {
            None
        } else {
            Some(cron.to_string())
        };
    }
    if let Some(interval) = update.min_refresh_interval_seconds {
        trigger.min_refresh_interval_seconds = Some(interval);
    }
    if let Some(tags_match) = update.tags_match.as_deref() {
        trigger.tags_match = if tags_match.trim().is_empty() {
            None
        } else {
            Some(parse_tags_match(tags_match)?)
        };
    }
    if let Some(keep_trace) = update.keep_trace {
        trigger.keep_trace = keep_trace;
    }
    if let Some(exclude) = update.exclude_mental_models {
        trigger.exclude_mental_models = exclude;
    }

    // The two refresh triggers are mutually exclusive, and the merged trigger
    // can carry both even though neither the stored one nor the flags did: a
    // model that refreshes after consolidation, moved onto a schedule. Setting
    // one drops an unstated other, mirroring what the server's _merge_trigger
    // does for a partial trigger — a request that named only a cron would
    // otherwise be rejected by the API's own validator with a bare 422.
    let cron_set = update.refresh_cron.is_some();
    let consolidation_set = update.refresh_after_consolidation.is_some();
    if trigger.refresh_after_consolidation && trigger.refresh_cron.is_some() {
        if cron_set && !consolidation_set {
            trigger.refresh_after_consolidation = false;
        } else if consolidation_set && !cron_set {
            trigger.refresh_cron = None;
        } else {
            // Both named in one command: only the user can say which they meant.
            anyhow::bail!(
                "--trigger-refresh-cron and --trigger-refresh-after-consolidation true \
                 are mutually exclusive: a mental model refreshes either after \
                 consolidation or on a cron schedule, not both"
            );
        }
    }

    Ok(trigger)
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
    trigger_mode: Option<&str>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating mental model..."))
    } else {
        None
    };

    let tags_match = tags_match.map(parse_tags_match).transpose()?;

    let mode = trigger_mode.map(parse_trigger_mode).transpose()?;

    // Only send a trigger when the user opted into one of its fields, so the
    // server's default behaviour (all_strict for tagged models) is preserved
    // otherwise.
    let trigger = if trigger_refresh_after_consolidation || tags_match.is_some() || mode.is_some() {
        Some(types::MentalModelTriggerInput {
            mode: mode.unwrap_or(types::Mode::Full),
            refresh_after_consolidation: trigger_refresh_after_consolidation,
            tags_match,
            ..default_trigger_input()
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
                ui::print_success(&format!(
                    "Mental model created, operation_id: {}",
                    result.operation_id
                ));
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
    trigger_update: &TriggerUpdate,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if name.is_none()
        && source_query.is_none()
        && max_tokens.is_none()
        && tags.is_none()
        && trigger_update.is_empty()
    {
        anyhow::bail!(
            "At least one of --name, --source-query, --max-tokens, --tags, or a \
             --trigger-* flag must be provided"
        );
    }

    // A trigger change is applied on top of the model's current trigger, so the
    // settings the user did not name survive: see apply_trigger_update for why
    // the stored trigger has to be read back rather than defaulted. No trigger
    // flag means no read and no trigger in the request at all.
    let trigger = if trigger_update.is_empty() {
        None
    } else {
        let current = client.get_mental_model(bank_id, mental_model_id, verbose)?;
        let base = current
            .trigger
            .as_ref()
            .map(stored_trigger_as_input)
            .unwrap_or_else(default_trigger_input);
        Some(apply_trigger_update(base, trigger_update)?)
    };

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating mental model..."))
    } else {
        None
    };

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
                ui::print_success(&format!(
                    "Mental model '{}' updated successfully",
                    mental_model_id
                ));
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
                ui::print_success(&format!(
                    "Mental model '{}' deleted successfully",
                    mental_model_id
                ));
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
                println!(
                    "{}",
                    ui::dim("Use 'hindsight operations get' to check the operation status.")
                );
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
            println!(
                "  {} {}",
                ui::dim("Used:"),
                format_counts(&result.facts.used)
            );
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
            println!(
                "{}",
                ui::dim("Nothing was saved. Use 'mental-model refresh' to apply.")
            );
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
    let mut entries: Vec<String> = counts
        .iter()
        .map(|(k, v)| format!("{}: {}", k, v))
        .collect();
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
                            let changed_at = entry
                                .get("changed_at")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let previous = entry
                                .get("previous_content")
                                .and_then(|v| v.as_str())
                                .unwrap_or("(none)");
                            println!("  {} {}", ui::dim("Changed at:"), changed_at);
                            let preview: String = previous.chars().take(80).collect();
                            let ellipsis = if previous.len() > 80 { "..." } else { "" };
                            println!(
                                "  {} {}{}",
                                ui::dim("Previous:"),
                                ui::dim(&preview),
                                ellipsis
                            );
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

    println!(
        "  {} {}",
        ui::dim("ID:"),
        ui::gradient_start(&mental_model.id)
    );
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
        assert_eq!(
            parse_tags_match("any_strict").unwrap(),
            types::TagsMatch::AnyStrict
        );
        assert_eq!(
            parse_tags_match("all_strict").unwrap(),
            types::TagsMatch::AllStrict
        );
        assert_eq!(parse_tags_match("exact").unwrap(), types::TagsMatch::Exact);
    }

    #[test]
    fn parse_tags_match_is_case_insensitive() {
        assert_eq!(parse_tags_match("ANY").unwrap(), types::TagsMatch::Any);
    }

    #[test]
    fn parse_tags_match_rejects_unknown() {
        let err = parse_tags_match("most").unwrap_err().to_string();
        assert!(
            err.contains("invalid --tags-match 'most'"),
            "unexpected error: {err}"
        );
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

    /// A trigger with a non-default value in every field the CLI does not
    /// expose, standing in for whatever the server has stored.
    fn stored_trigger() -> types::MentalModelTriggerOutput {
        types::MentalModelTriggerOutput {
            mode: types::Mode::Delta,
            refresh_after_consolidation: true,
            refresh_cron: None,
            min_refresh_interval_seconds: Some(900),
            exclude_mental_models: true,
            exclude_mental_model_ids: Some(vec!["mm-other".to_string()]),
            fact_types: Some(vec![types::FactTypesItem::Observation]),
            tag_groups: None,
            tags_match: Some(types::TagsMatch::AnyStrict),
            include_chunks: Some(true),
            recall_max_tokens: Some(4096),
            recall_chunks_max_tokens: Some(2048),
            response_schema: None,
            keep_trace: true,
        }
    }

    #[test]
    fn trigger_update_is_empty_only_without_flags() {
        assert!(TriggerUpdate::default().is_empty());
        assert!(!TriggerUpdate {
            keep_trace: Some(false),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn stored_trigger_round_trips_into_the_input_type() {
        let input = stored_trigger_as_input(&stored_trigger());
        let value = serde_json::to_value(&input).unwrap();
        assert_eq!(value["mode"], "delta");
        assert_eq!(value["refresh_after_consolidation"], true);
        assert_eq!(value["min_refresh_interval_seconds"], 900);
        assert_eq!(value["exclude_mental_models"], true);
        assert_eq!(value["exclude_mental_model_ids"][0], "mm-other");
        assert_eq!(value["fact_types"][0], "observation");
        assert_eq!(value["tags_match"], "any_strict");
        assert_eq!(value["include_chunks"], true);
        assert_eq!(value["recall_max_tokens"], 4096);
        assert_eq!(value["recall_chunks_max_tokens"], 2048);
        assert_eq!(value["keep_trace"], true);
    }

    /// The regression this command had: changing one trigger setting must not
    /// reset the four fields the generated type always serializes.
    #[test]
    fn apply_trigger_update_keeps_unnamed_fields() {
        let update = TriggerUpdate {
            mode: Some("full".to_string()),
            ..Default::default()
        };
        let trigger = apply_trigger_update(stored_trigger_as_input(&stored_trigger()), &update)
            .expect("mode-only update applies");

        assert_eq!(trigger.mode, types::Mode::Full);
        assert!(trigger.refresh_after_consolidation);
        assert!(trigger.exclude_mental_models);
        assert!(trigger.keep_trace);
        assert_eq!(trigger.min_refresh_interval_seconds, Some(900));
        assert_eq!(trigger.tags_match, Some(types::TagsMatch::AnyStrict));
        assert_eq!(trigger.recall_max_tokens, Some(4096));
    }

    #[test]
    fn apply_trigger_update_sets_every_exposed_field() {
        let update = TriggerUpdate {
            mode: Some("DELTA".to_string()),
            refresh_after_consolidation: Some(false),
            refresh_cron: Some("0 3 * * *".to_string()),
            min_refresh_interval_seconds: Some(0),
            tags_match: Some("exact".to_string()),
            keep_trace: Some(false),
            exclude_mental_models: Some(false),
        };
        let trigger =
            apply_trigger_update(default_trigger_input(), &update).expect("update applies");

        assert_eq!(trigger.mode, types::Mode::Delta);
        assert!(!trigger.refresh_after_consolidation);
        assert_eq!(trigger.refresh_cron.as_deref(), Some("0 3 * * *"));
        assert_eq!(trigger.min_refresh_interval_seconds, Some(0));
        assert_eq!(trigger.tags_match, Some(types::TagsMatch::Exact));
        assert!(!trigger.keep_trace);
        assert!(!trigger.exclude_mental_models);
    }

    #[test]
    fn apply_trigger_update_clears_cron_and_tags_match_on_empty_string() {
        let base = types::MentalModelTriggerInput {
            refresh_cron: Some("0 3 * * *".to_string()),
            tags_match: Some(types::TagsMatch::All),
            ..default_trigger_input()
        };
        let update = TriggerUpdate {
            refresh_cron: Some(String::new()),
            tags_match: Some("  ".to_string()),
            ..Default::default()
        };
        let trigger = apply_trigger_update(base, &update).expect("clearing applies");

        assert!(trigger.refresh_cron.is_none());
        assert!(trigger.tags_match.is_none());
    }

    /// The stored trigger already refreshes after consolidation, so a cron
    /// lands on a combination the API rejects. Setting one drops an unstated
    /// other, as the server does when it merges a partial trigger.
    #[test]
    fn apply_trigger_update_cron_clears_stored_consolidation() {
        let trigger = apply_trigger_update(
            stored_trigger_as_input(&stored_trigger()),
            &TriggerUpdate {
                refresh_cron: Some("0 3 * * *".to_string()),
                ..Default::default()
            },
        )
        .expect("cron-only update applies");

        assert_eq!(trigger.refresh_cron.as_deref(), Some("0 3 * * *"));
        assert!(!trigger.refresh_after_consolidation);
    }

    #[test]
    fn apply_trigger_update_consolidation_clears_stored_cron() {
        let base = types::MentalModelTriggerInput {
            refresh_cron: Some("0 3 * * *".to_string()),
            ..default_trigger_input()
        };
        let trigger = apply_trigger_update(
            base,
            &TriggerUpdate {
                refresh_after_consolidation: Some(true),
                ..Default::default()
            },
        )
        .expect("consolidation-only update applies");

        assert!(trigger.refresh_after_consolidation);
        assert!(trigger.refresh_cron.is_none());
    }

    #[test]
    fn apply_trigger_update_rejects_both_refresh_flags_at_once() {
        let err = apply_trigger_update(
            default_trigger_input(),
            &TriggerUpdate {
                refresh_cron: Some("0 3 * * *".to_string()),
                refresh_after_consolidation: Some(true),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("mutually exclusive"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn apply_trigger_update_propagates_parse_errors() {
        let err = apply_trigger_update(
            default_trigger_input(),
            &TriggerUpdate {
                mode: Some("incremental".to_string()),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("invalid --trigger-mode 'incremental'"),
            "unexpected error: {err}"
        );

        let err = apply_trigger_update(
            default_trigger_input(),
            &TriggerUpdate {
                tags_match: Some("most".to_string()),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("invalid --tags-match 'most'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn update_request_omits_trigger_when_no_flags() {
        let request = types::UpdateMentalModelRequest {
            name: Some("renamed".to_string()),
            source_query: None,
            max_tokens: None,
            tags: None,
            trigger: None,
        };
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["name"], "renamed");
        assert!(value.get("trigger").is_none() || value["trigger"].is_null());
    }
}
