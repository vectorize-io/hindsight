//! Knowledge base commands for managing a bank's folder/page tree.
//!
//! A page is a mental model with document-shaped defaults plus a place in the
//! tree, so these commands mirror `hindsight mental-model` where the operations
//! overlap. `hindsight fs` mirrors the same knowledge base read-only onto disk;
//! these commands are the read/write side.

use anyhow::Result;

use crate::api::{ApiClient, KnowledgeTagFilter};
use crate::output::{self, OutputFormat};
use crate::ui;

use hindsight_client::types;

/// Parse a --fact-types value into the generated enum, rejecting unknown values
/// so a typo fails loudly instead of silently widening what a page reads.
fn parse_fact_type(value: &str) -> Result<types::FactTypesItem> {
    Ok(match value.to_lowercase().as_str() {
        "world" => types::FactTypesItem::World,
        "experience" => types::FactTypesItem::Experience,
        "observation" => types::FactTypesItem::Observation,
        other => anyhow::bail!(
            "invalid --fact-types value '{other}': expected one of world, experience, observation"
        ),
    })
}

/// Parse a --mode value into the generated refresh-mode enum.
fn parse_mode(value: &str) -> Result<types::Mode> {
    Ok(match value.to_lowercase().as_str() {
        "full" => types::Mode::Full,
        "delta" => types::Mode::Delta,
        other => anyhow::bail!("invalid --mode '{other}': expected one of full, delta"),
    })
}

/// Build the knowledge-base tag filter from the `--tags` / `--tags-match` / `--tag-groups` flags.
pub fn tag_filter(
    tags: Vec<String>,
    tags_match: Option<String>,
    tag_groups: Option<String>,
) -> Result<KnowledgeTagFilter> {
    let tags_match = tags_match
        .map(|value| {
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
        })
        .transpose()?;
    if let Some(raw) = tag_groups.as_deref() {
        validate_tag_groups_json(raw)?;
        if !tags.is_empty() {
            anyhow::bail!("--tags and --tag-groups are mutually exclusive — use --tag-groups for compound filtering");
        }
    }
    Ok(KnowledgeTagFilter {
        tags,
        tags_match,
        tag_groups,
    })
}

/// Validate `--tag-groups` locally so a malformed filter fails with a CLI error
/// instead of a server 400 — same shape as recall's `tag_groups`: a top-level
/// array whose members are leaves `{"tags": [...], "match"?, "resolve"?}` or
/// compound `{"and"|"or": [...]}` / `{"not": {...}}` nodes. No depth cap here —
/// serde_json's own parse recursion limit already bounds nesting, and the API
/// schema imposes none, so a deep `not` chain the server accepts must pass.
fn validate_tag_groups_json(raw: &str) -> Result<()> {
    fn check(node: &serde_json::Value) -> bool {
        let Some(obj) = node.as_object() else {
            return false;
        };
        if let Some(tags) = obj.get("tags") {
            tags.is_array()
                && tags.as_array().unwrap().iter().all(|t| t.is_string())
                && obj.get("match").map_or(true, |m| {
                    m.as_str().map_or(false, |s| {
                        ["any", "all", "any_strict", "all_strict", "exact"]
                            .contains(&s)
                    })
                })
                && obj.get("resolve").map_or(true, |r| {
                    r.as_str().map_or(false, |s| ["exact", "fuzzy"].contains(&s))
                })
        } else if let Some(list) = obj.get("and").or_else(|| obj.get("or")) {
            list.as_array()
                .map_or(false, |items| items.iter().all(|g| check(g)))
        } else if let Some(g) = obj.get("not") {
            check(g)
        } else {
            false
        }
    }

    let parsed: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| anyhow::anyhow!("invalid --tag-groups JSON: {e}"))?;
    match parsed.as_array() {
        Some(items) if items.iter().all(check) => Ok(()),
        _ => anyhow::bail!(
            "invalid --tag-groups: expected a JSON array of tag groups \
             (leaves {{\"tags\": [...]}} or {{\"and\"/\"or\"/\"not\": ...}})"
        ),
    }
}

/// Show the folder/page tree for a bank
pub fn tree(
    client: &ApiClient,
    bank_id: &str,
    filter: &KnowledgeTagFilter,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching knowledge base..."))
    } else {
        None
    };

    let response = client.get_knowledge_base_tree(bank_id, filter, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Knowledge Base: {}", bank_id));

                if result.roots.is_empty() {
                    println!("  {}", ui::dim("No folders or pages found."));
                } else {
                    for root in &result.roots {
                        print_node(root, 1);
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

/// Create a folder
pub fn create_folder(
    client: &ApiClient,
    bank_id: &str,
    name: &str,
    parent_id: Option<&str>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating folder..."))
    } else {
        None
    };

    let request = types::CreateFolderRequest {
        name: name.to_string(),
        parent_id: parent_id.map(|s| s.to_string()),
    };

    let response = client.create_knowledge_folder(bank_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(node) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Folder created: {}", node.id));
            } else {
                output::print_output(&node, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Create a page
#[allow(clippy::too_many_arguments)]
pub fn create_page(
    client: &ApiClient,
    bank_id: &str,
    name: &str,
    source_query: &str,
    parent_id: Option<&str>,
    tags: Vec<String>,
    max_tokens: Option<i64>,
    mode: Option<&str>,
    fact_types: Vec<String>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // Only send a trigger when the user overrode part of it. Omitting it keeps
    // the server's page defaults (observation-only, delta, refresh after
    // consolidation) — and because a supplied trigger REPLACES those defaults
    // rather than merging, a partial override has to restate them.
    let trigger = if mode.is_some() || !fact_types.is_empty() {
        let parsed_fact_types = if fact_types.is_empty() {
            vec![types::FactTypesItem::Observation]
        } else {
            fact_types
                .iter()
                .map(|ft| parse_fact_type(ft))
                .collect::<Result<Vec<_>>>()?
        };
        Some(types::MentalModelTriggerInput {
            mode: mode
                .map(parse_mode)
                .transpose()?
                .unwrap_or(types::Mode::Delta),
            refresh_after_consolidation: true,
            refresh_cron: None,
            min_refresh_interval_seconds: None,
            exclude_mental_models: true,
            exclude_mental_model_ids: None,
            fact_types: Some(parsed_fact_types),
            tag_groups: None,
            tags_match: None,
            include_chunks: None,
            recall_max_tokens: None,
            recall_chunks_max_tokens: None,
            reflect_search_observations_max_tokens: None,
            reflect_search_observations_include_entities: None,
            budget: None,
            response_schema: None,
            keep_trace: false,
        })
    } else {
        None
    };

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Creating page..."))
    } else {
        None
    };

    let request = types::CreatePageRequest {
        name: name.to_string(),
        source_query: source_query.to_string(),
        parent_id: parent_id.map(|s| s.to_string()),
        tags: if tags.is_empty() { None } else { Some(tags) },
        max_tokens,
        trigger,
    };

    let response = client.create_knowledge_page(bank_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Page created: {}", result.page_id));
                if let Some(ref operation_id) = result.operation_id {
                    println!("  {} {}", ui::dim("Operation:"), operation_id);
                    println!();
                    println!(
                        "{}",
                        ui::dim("Content is generated in the background. Use 'hindsight operation get' to track it.")
                    );
                }
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Read a page as a markdown document
pub fn get_page(
    client: &ApiClient,
    bank_id: &str,
    page_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching page..."))
    } else {
        None
    };

    let response = client.get_knowledge_page(bank_id, page_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(page) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&page.name);
                println!("  {} {}", ui::dim("ID:"), ui::gradient_start(&page.id));
                println!("  {} {}", ui::dim("Type:"), page.type_);
                if let Some(ref description) = page.description {
                    println!("  {} {}", ui::dim("Question:"), description);
                }
                if let Some(ref timestamp) = page.timestamp {
                    println!("  {} {}", ui::dim("Last refresh:"), timestamp);
                }
                if !page.tags.is_empty() {
                    println!("  {} {}", ui::dim("Tags:"), page.tags.join(", "));
                }
                if let Some(ref body) = page.body {
                    println!();
                    println!("{}", body);
                    println!();
                }
            } else {
                output::print_output(&page, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Search pages (hybrid full-text + vector)
pub fn search(
    client: &ApiClient,
    bank_id: &str,
    query: &str,
    limit: Option<u64>,
    filter: &KnowledgeTagFilter,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // The generated query type rejects an empty string, which is also what the
    // server enforces (min_length 1); surface it as a normal CLI error.
    let query: types::Q = query
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid search query: {e}"))?;
    let limit = limit
        .map(|l| {
            std::num::NonZeroU64::new(l).ok_or_else(|| {
                anyhow::anyhow!("invalid --limit 0: expected a value between 1 and 50")
            })
        })
        .transpose()?;

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Searching knowledge base..."))
    } else {
        None
    };

    let response = client.search_knowledge_base(bank_id, &query, limit, filter, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Search Results: {}", bank_id));

                if result.results.is_empty() {
                    println!("  {}", ui::dim("No pages matched."));
                } else {
                    for hit in &result.results {
                        println!(
                            "  {} {} {}",
                            ui::gradient_start(&hit.id),
                            hit.name,
                            ui::dim(&format!("({:.3})", hit.score))
                        );
                        let preview: String = hit.snippet.chars().take(100).collect();
                        let ellipsis = if hit.snippet.len() > 100 { "..." } else { "" };
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

/// Rename/move a node and/or update a page's options
#[allow(clippy::too_many_arguments)]
pub fn update(
    client: &ApiClient,
    bank_id: &str,
    node_id: &str,
    name: Option<String>,
    parent_id: Option<String>,
    source_query: Option<String>,
    tags: Option<Vec<String>>,
    max_tokens: Option<i64>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if name.is_none()
        && parent_id.is_none()
        && source_query.is_none()
        && tags.is_none()
        && max_tokens.is_none()
    {
        anyhow::bail!(
            "At least one of --name, --parent-id, --source-query, --tags, or --max-tokens must be provided"
        );
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Updating node..."))
    } else {
        None
    };

    let request = types::UpdateNodeRequest {
        name,
        parent_id,
        source_query,
        tags,
        max_tokens,
        // A page's refresh policy is not exposed as a CLI flag (see
        // .openapi-coverage.toml); omitted, it leaves the page's current trigger alone.
        trigger: None,
    };

    let response = client.update_knowledge_node(bank_id, node_id, &request, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(node) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Node '{}' updated successfully", node_id));
                print_node(&node, 1);
            } else {
                output::print_output(&node, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Delete a folder or page (and its whole subtree)
pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    node_id: &str,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    // Deleting a folder takes its whole subtree (and the pages' mental models)
    // with it, so confirm unless the caller opted out.
    if !yes && output_format == OutputFormat::Pretty {
        let message = format!(
            "Are you sure you want to delete '{}' and everything under it? This cannot be undone.",
            node_id
        );

        if !ui::prompt_confirmation(&message)? {
            ui::print_info("Operation cancelled");
            return Ok(());
        }
    }

    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Deleting node..."))
    } else {
        None
    };

    let response = client.delete_knowledge_node(bank_id, node_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(_) => {
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Node '{}' deleted successfully", node_id));
            } else {
                println!("{{\"success\": true}}");
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Export the knowledge base as a markdown bundle
pub fn export(
    client: &ApiClient,
    bank_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Exporting knowledge base..."))
    } else {
        None
    };

    let response = client.export_knowledge_base(bank_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(bundle) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Knowledge Base Bundle: {}", bank_id));
                for file in &bundle.files {
                    println!("  {}", ui::gradient_start(&file.path));
                }
                println!();
                println!(
                    "{}",
                    ui::dim("Use --output json to get the file contents, or 'hindsight fs mount' to mirror them to disk.")
                );
            } else {
                output::print_output(&bundle, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Print one tree node and its children, indented by depth.
fn print_node(node: &types::KnowledgeNode, depth: usize) {
    let indent = "  ".repeat(depth);
    let marker = if node.kind == types::Kind::Folder {
        "/"
    } else {
        ""
    };
    // Staleness only means something for pages; folders always report None.
    let stale = if node.is_stale == Some(true) {
        format!(" {}", ui::dim("(stale)"))
    } else {
        String::new()
    };
    println!(
        "{}{} {}{}{}",
        indent,
        ui::gradient_start(&node.id),
        node.name,
        marker,
        stale
    );
    for child in &node.children {
        print_node(child, depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fact_type_accepts_every_type() {
        assert_eq!(
            parse_fact_type("world").unwrap(),
            types::FactTypesItem::World
        );
        assert_eq!(
            parse_fact_type("experience").unwrap(),
            types::FactTypesItem::Experience
        );
        assert_eq!(
            parse_fact_type("OBSERVATION").unwrap(),
            types::FactTypesItem::Observation
        );
    }

    #[test]
    fn parse_fact_type_rejects_unknown() {
        let err = parse_fact_type("belief").unwrap_err().to_string();
        assert!(
            err.contains("invalid --fact-types value 'belief'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parse_mode_accepts_both_modes() {
        assert_eq!(parse_mode("full").unwrap(), types::Mode::Full);
        assert_eq!(parse_mode("DELTA").unwrap(), types::Mode::Delta);
    }

    #[test]
    fn parse_mode_rejects_unknown() {
        let err = parse_mode("incremental").unwrap_err().to_string();
        assert!(
            err.contains("invalid --mode 'incremental'"),
            "unexpected error: {err}"
        );
    }
}

#[cfg(test)]
mod tag_filter_tests {
    use super::*;

    #[test]
    fn parses_every_mode_and_rejects_unknown() {
        for (raw, want) in [
            ("any", types::TagsMatch::Any),
            ("ALL", types::TagsMatch::All),
            ("any_strict", types::TagsMatch::AnyStrict),
            ("all_strict", types::TagsMatch::AllStrict),
            ("exact", types::TagsMatch::Exact),
        ] {
            let f = tag_filter(vec!["user:kate".into()], Some(raw.into()), None).unwrap();
            assert_eq!(f.tags_match, Some(want));
        }
        assert!(tag_filter(vec![], Some("most".into()), None).is_err());
        let f = tag_filter(vec![], None, Some("[]".into())).unwrap();
        assert_eq!((f.tags_match, f.tag_groups.as_deref()), (None, Some("[]")));
    }

    #[test]
    fn tags_and_tag_groups_are_mutually_exclusive() {
        let groups = Some(r#"[{"tags":["a","b"]}]"#.to_string());
        let err = tag_filter(vec!["user:kate".into()], None, groups.clone()).unwrap_err();
        assert!(err.to_string().contains("mutually exclusive"));
        let f = tag_filter(vec!["user:kate".into()], None, None).unwrap();
        assert_eq!(f.tags, vec!["user:kate".to_string()]);
        let f = tag_filter(vec![], None, groups).unwrap();
        assert_eq!(f.tag_groups.as_deref(), Some(r#"[{"tags":["a","b"]}]"#));
    }
}
