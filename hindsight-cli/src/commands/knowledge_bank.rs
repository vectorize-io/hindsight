//! Knowledge bank commands.
//!
//! A knowledge bank stores files and texts as they are: each becomes a document split into
//! passages that search finds by meaning. Nothing is extracted into memories, so these commands
//! stand apart from `bank` / `memory` / `document`, which act on memory banks.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use base64::Engine;
use serde_json::{json, Map, Value};
use walkdir::WalkDir;

use crate::api::{ApiClient, KnowledgeFileOptions};
use crate::commands::memory::document_id_for_path;
use crate::output::{self, OutputFormat};
use crate::ui;
use crate::KnowledgeBankCommands as Cmd;

use hindsight_client::types;

pub fn run(client: &ApiClient, cmd: Cmd, verbose: bool, format: OutputFormat) -> Result<()> {
    match cmd {
        Cmd::List {
            query,
            limit,
            offset,
        } => {
            let page = spin(format, "Fetching knowledge banks...", || {
                client.list_knowledge_banks(query.as_deref(), limit, offset)
            })?;
            pretty_or(format, &page, || {
                ui::print_section_header(&format!("Knowledge banks ({})", page.total));
                if page.banks.is_empty() {
                    println!("  {}", ui::dim("No knowledge banks."));
                }
                for bank in &page.banks {
                    println!(
                        "  {}  {}",
                        ui::gradient_start(&bank.bank_id),
                        ui::dim(&format!(
                            "{} documents, {} passages",
                            bank.document_count, bank.passage_count
                        ))
                    );
                }
            })
        }
        Cmd::Create { bank_id } => {
            let bank = spin(format, "Creating knowledge bank...", || {
                client.create_knowledge_bank(&bank_id)
            })?;
            pretty_or(format, &bank, || {
                ui::print_success(&format!("Created knowledge bank '{}'", bank.bank_id))
            })
        }
        Cmd::Get { bank_id } => {
            let bank = spin(format, "Fetching knowledge bank...", || {
                client.get_knowledge_bank(&bank_id)
            })?;
            pretty_or(format, &bank, || print_bank(&bank))
        }
        Cmd::Delete { bank_id, yes } => {
            if !confirm(
                yes,
                format,
                &format!("Delete knowledge bank '{bank_id}' with all its documents?"),
            )? {
                return Ok(());
            }
            let result = spin(format, "Deleting knowledge bank...", || {
                client.delete_knowledge_bank(&bank_id)
            })?;
            pretty_or(format, &result, || {
                ui::print_success(&format!("Deleted knowledge bank '{bank_id}'"))
            })
        }
        Cmd::Config {
            bank_id,
            overrides_only,
        } => {
            let config = spin(format, "Fetching configuration...", || {
                client.get_knowledge_bank_config(&bank_id)
            })?;
            if overrides_only {
                return pretty_or(format, &config.overrides, || {
                    print_settings("Overrides", &config.overrides)
                });
            }
            pretty_or(format, &config, || {
                print_settings(&format!("Configuration: {bank_id}"), &config.config);
                print_settings("Overrides", &config.overrides);
            })
        }
        Cmd::SetConfig {
            bank_id,
            set,
            unset,
        } => {
            let updates = config_updates(&set, &unset)?;
            let config = spin(format, "Updating configuration...", || {
                client.update_knowledge_bank_config(&bank_id, updates)
            })?;
            pretty_or(format, &config, || {
                ui::print_success(&format!("Configuration updated for '{bank_id}'"));
                print_settings("Overrides", &config.overrides);
            })
        }
        Cmd::ResetConfig { bank_id, yes } => {
            if !confirm(
                yes,
                format,
                &format!("Remove every configuration override of '{bank_id}'?"),
            )? {
                return Ok(());
            }
            let config = spin(format, "Resetting configuration...", || {
                client.reset_knowledge_bank_config(&bank_id)
            })?;
            pretty_or(format, &config, || {
                ui::print_success(&format!("Configuration reset for '{bank_id}'"))
            })
        }
        Cmd::Ingest {
            bank_id,
            paths,
            document_id,
            tags,
            strategy,
            wait,
        } => ingest(
            client,
            &bank_id,
            &paths,
            document_id,
            tags,
            strategy,
            wait,
            verbose,
            format,
        ),
        Cmd::IngestText {
            bank_id,
            text,
            document_id,
            filename,
            tags,
            strategy,
            r#async,
        } => {
            let text = if text == "-" {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                text
            };
            let body: types::KnowledgeIngestRequest = serde_json::from_value(json!({
                "items": [{
                    "text": text,
                    "document_id": document_id,
                    "filename": filename,
                    "tags": tags,
                    "strategy": strategy,
                }],
                "async": r#async,
            }))
            .context("invalid text ingest")?;
            let result = spin(format, "Ingesting text...", || {
                client.ingest_knowledge_texts(&bank_id, &body)
            })?;
            pretty_or(format, &result, || {
                let id = result
                    .document_ids
                    .first()
                    .map(String::as_str)
                    .unwrap_or("");
                if result.unchanged_document_ids.iter().any(|u| u == id) {
                    ui::print_info(&format!("Document '{id}' is unchanged"));
                } else if r#async {
                    ui::print_success(&format!("Document '{id}' queued"));
                } else {
                    ui::print_success(&format!("Document '{id}' is searchable"));
                }
            })
        }
        Cmd::Search {
            bank_id,
            query,
            query_image,
            query_video,
            budget,
            max_tokens,
            tags,
            tags_match,
            tag_groups,
            document_ids,
            media_types,
            no_rerank,
            trace,
        } => {
            if query.is_none() && query_image.is_none() && query_video.is_none() {
                anyhow::bail!("Give a text query, --image or --video (or several)");
            }
            let tag_groups: Option<Value> = tag_groups
                .map(|raw| serde_json::from_str(&raw).context("--tag-groups is not valid JSON"))
                .transpose()?;
            let mut body = json!({
                "query": query,
                "query_image": query_image.as_deref().map(base64_file).transpose()?,
                "query_video": query_video.as_deref().map(base64_file).transpose()?,
                "tags": (!tags.is_empty()).then_some(tags),
                "tag_groups": tag_groups,
                "document_ids": (!document_ids.is_empty()).then_some(document_ids),
                "media_types": (!media_types.is_empty()).then_some(media_types),
                "enable_reranking": no_rerank.then_some(false),
                "trace": trace,
            });
            // Unset flags keep the server's defaults.
            for (key, value) in [
                ("budget", budget.map(Value::from)),
                ("max_tokens", max_tokens.map(Value::from)),
                ("tags_match", tags_match.map(Value::from)),
            ] {
                if let Some(value) = value {
                    body[key] = value;
                }
            }
            let body: types::KnowledgeSearchRequest =
                serde_json::from_value(body).context("invalid search options")?;
            let response = spin(format, "Searching...", || {
                client.search_knowledge_bank(&bank_id, &body, verbose)
            })?;
            pretty_or(format, &response, || print_search(&response))
        }
        Cmd::ListDocuments {
            bank_id,
            query,
            status,
            media_type,
            tags,
            limit,
            offset,
        } => {
            let page = spin(format, "Fetching documents...", || {
                client.list_knowledge_documents(
                    &bank_id,
                    query.as_deref(),
                    status.as_deref(),
                    media_type.as_deref(),
                    &tags,
                    limit,
                    offset,
                )
            })?;
            pretty_or(format, &page, || {
                ui::print_section_header(&format!("Documents: {bank_id} ({})", page.total));
                if page.documents.is_empty() {
                    println!("  {}", ui::dim("No documents."));
                }
                for doc in &page.documents {
                    println!(
                        "  {}  {}",
                        ui::gradient_start(&doc.document_id),
                        ui::dim(&format!(
                            "{} · {} · {} passages{}",
                            doc.media_type,
                            doc.status,
                            doc.passage_count,
                            tags_suffix(&doc.tags)
                        ))
                    );
                    if let Some(error) = &doc.error {
                        println!("    {}", error);
                    }
                }
            })
        }
        Cmd::GetDocument {
            bank_id,
            document_id,
        } => {
            let doc = spin(format, "Fetching document...", || {
                client.get_knowledge_document(&bank_id, &document_id)
            })?;
            pretty_or(format, &doc, || print_document(&doc))
        }
        Cmd::DeleteDocument {
            bank_id,
            document_id,
        } => {
            let result = spin(format, "Deleting document...", || {
                client.delete_knowledge_document(&bank_id, &document_id)
            })?;
            pretty_or(format, &result, || {
                ui::print_success(&format!("Deleted document '{document_id}'"))
            })
        }
        Cmd::Download {
            bank_id,
            document_id,
            out,
        } => {
            let output = match out {
                Some(path) => path,
                None => {
                    let doc = client.get_knowledge_document(&bank_id, &document_id)?;
                    let name = default_download_name(doc.filename.as_deref(), &document_id)?;
                    // Nobody named this file, so don't clobber one that is already there.
                    if name.exists() {
                        anyhow::bail!(
                            "{} already exists; pass --out to choose where to write",
                            name.display()
                        );
                    }
                    name
                }
            };
            write_to(&output, format, |out| {
                client.get_knowledge_document_file(&bank_id, &document_id, out)
            })
        }
        Cmd::PassageMedia {
            bank_id,
            document_id,
            index,
            out,
        } => write_to(&out, format, |out| {
            client.get_knowledge_passage_media(&bank_id, &document_id, index, out)
        }),
        Cmd::ListTags {
            bank_id,
            query,
            limit,
            offset,
        } => {
            let page = spin(format, "Fetching tags...", || {
                client.list_knowledge_bank_tags(&bank_id, query.as_deref(), limit, offset)
            })?;
            pretty_or(format, &page, || {
                ui::print_section_header(&format!("Tags: {bank_id}"));
                if page.items.is_empty() {
                    println!("  {}", ui::dim("No tags found."));
                }
                for tag in &page.items {
                    println!(
                        "  {} {}",
                        ui::gradient_start(&tag.tag),
                        ui::dim(&format!("({})", tag.count))
                    );
                }
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ingest(
    client: &ApiClient,
    bank_id: &str,
    paths: &[PathBuf],
    document_id: Option<String>,
    tags: Vec<String>,
    strategy: Option<String>,
    wait: bool,
    verbose: bool,
    format: OutputFormat,
) -> Result<()> {
    let files = collect_files(paths)?;
    if files.is_empty() {
        anyhow::bail!("No files found");
    }
    if document_id.is_some() && files.len() > 1 {
        anyhow::bail!("--document-id needs a single file; got {}", files.len());
    }

    // One file per request: a file the server refuses fails alone, and each upload streams.
    let mut document_ids = Vec::new();
    let mut operation_ids = Vec::new();
    let mut unchanged = Vec::new();
    let mut failed = Vec::new();
    let progress = (format == OutputFormat::Pretty)
        .then(|| ui::create_progress_bar(files.len() as u64, "Uploading"));
    for path in &files {
        let options = KnowledgeFileOptions {
            document_id: Some(
                document_id
                    .clone()
                    .unwrap_or_else(|| document_id_for_path(path)),
            ),
            tags: tags.clone(),
            strategy: strategy.clone(),
        };
        match client.ingest_knowledge_files(bank_id, path, &options, verbose) {
            Ok(result) => {
                document_ids.extend(result.document_ids);
                unchanged.extend(result.unchanged_document_ids);
                for op in result.operation_ids {
                    if !operation_ids.contains(&op) {
                        operation_ids.push(op);
                    }
                }
            }
            Err(e) => {
                failed.push(json!({"path": path.display().to_string(), "error": e.to_string()}))
            }
        }
        if let Some(pb) = &progress {
            pb.inc(1);
        }
    }
    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    let mut statuses = Map::new();
    if wait {
        let pending: Vec<&String> = document_ids
            .iter()
            .filter(|id| !unchanged.contains(id))
            .collect();
        let spinner =
            (format == OutputFormat::Pretty).then(|| ui::create_spinner("Processing documents..."));
        for id in pending {
            let doc = wait_for_document(client, bank_id, id)?;
            if doc.status == types::KnowledgeDocumentStatus::Failed {
                failed.push(json!({"document_id": id, "error": doc.error}));
            }
            statuses.insert(id.clone(), json!(doc.status));
        }
        if let Some(mut sp) = spinner {
            sp.finish();
        }
    }

    let summary = json!({
        "document_ids": document_ids,
        "operation_ids": operation_ids,
        "unchanged_document_ids": unchanged,
        "statuses": statuses,
        "failed": failed,
    });
    if format == OutputFormat::Pretty {
        let queued = document_ids.len() - unchanged.len();
        if wait {
            ui::print_success(&format!(
                "{} document(s) processed, {} unchanged",
                queued,
                unchanged.len()
            ));
        } else {
            ui::print_success(&format!(
                "{} document(s) queued, {} unchanged",
                queued,
                unchanged.len()
            ));
            if queued > 0 {
                println!(
                    "  {}",
                    ui::dim("Each is searchable once its status is ready (see list-documents), or pass --wait.")
                );
            }
        }
        for f in &failed {
            let what = f["path"]
                .as_str()
                .or(f["document_id"].as_str())
                .unwrap_or("");
            ui::print_error(&format!(
                "{}: {}",
                what,
                f["error"].as_str().unwrap_or("failed")
            ));
        }
    } else {
        output::print_output(&summary, format)?;
    }
    if !failed.is_empty() {
        anyhow::bail!("{} of {} file(s) failed", failed.len(), files.len());
    }
    Ok(())
}

/// The file name a download defaults to: the document's filename, else its id. Only the last
/// path segment is kept, so a stored name like `../x` cannot write outside the current dir.
fn default_download_name(filename: Option<&str>, document_id: &str) -> Result<PathBuf> {
    filename
        .and_then(|f| Path::new(f).file_name())
        .or_else(|| Path::new(document_id).file_name())
        .map(PathBuf::from)
        .with_context(|| format!("no file name for '{document_id}'; pass --out"))
}

/// Every file under `paths`: files as given, folders walked recursively without hidden entries.
/// Unlike `memory retain-files`, folders are not filtered by extension here: which types a
/// knowledge bank reads (and which an ingestion strategy adds) is the server's call, so a file it
/// refuses fails alone and is reported, instead of a client-side list drifting from the server's.
fn collect_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_file() {
            files.push(path.clone());
        } else if path.is_dir() {
            let walker = WalkDir::new(path).sort_by_file_name().into_iter();
            for entry in walker.filter_entry(|e| e.depth() == 0 || !is_hidden(e.file_name())) {
                let entry = entry?;
                if entry.file_type().is_file() {
                    files.push(entry.into_path());
                }
            }
        } else {
            anyhow::bail!("Path does not exist: {}", path.display());
        }
    }
    Ok(files)
}

fn is_hidden(name: &std::ffi::OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}

fn wait_for_document(
    client: &ApiClient,
    bank_id: &str,
    document_id: &str,
) -> Result<types::KnowledgeDocumentDetailResponse> {
    loop {
        let doc = client.get_knowledge_document(bank_id, document_id)?;
        match doc.status {
            types::KnowledgeDocumentStatus::Ready | types::KnowledgeDocumentStatus::Failed => {
                return Ok(doc)
            }
            _ => std::thread::sleep(Duration::from_secs(1)),
        }
    }
}

/// `--set KEY=VALUE` (VALUE as JSON when it parses, else text) and `--unset KEY` (null clears).
fn config_updates(set: &[String], unset: &[String]) -> Result<Map<String, Value>> {
    let mut updates = Map::new();
    for pair in set {
        let (key, raw) = pair
            .split_once('=')
            .with_context(|| format!("--set '{pair}' is not KEY=VALUE"))?;
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        updates.insert(key.trim().to_string(), value);
    }
    for key in unset {
        updates.insert(key.clone(), Value::Null);
    }
    if updates.is_empty() {
        anyhow::bail!("Nothing to change: pass --set KEY=VALUE or --unset KEY");
    }
    Ok(updates)
}

fn base64_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// Write a download to `output` (`-` is stdout).
fn write_to(
    output: &Path,
    format: OutputFormat,
    download: impl FnOnce(&mut dyn std::io::Write) -> Result<u64>,
) -> Result<()> {
    if output == Path::new("-") {
        download(&mut std::io::stdout().lock())?;
        return Ok(());
    }
    let mut file = std::io::BufWriter::new(
        fs::File::create(output)
            .with_context(|| format!("Failed to create {}", output.display()))?,
    );
    let written = download(&mut file).inspect_err(|_| {
        let _ = fs::remove_file(output);
    })?;
    let summary = json!({"path": output.display().to_string(), "bytes": written});
    pretty_or(format, &summary, || {
        ui::print_success(&format!("Wrote {} ({} bytes)", output.display(), written))
    })
}

fn spin<T>(format: OutputFormat, message: &str, call: impl FnOnce() -> Result<T>) -> Result<T> {
    let spinner = (format == OutputFormat::Pretty).then(|| ui::create_spinner(message));
    let result = call();
    if let Some(mut sp) = spinner {
        sp.finish();
    }
    result
}

fn pretty_or<T: serde::Serialize>(
    format: OutputFormat,
    data: &T,
    pretty: impl FnOnce(),
) -> Result<()> {
    if format == OutputFormat::Pretty {
        pretty();
        Ok(())
    } else {
        output::print_output(data, format)
    }
}

fn confirm(yes: bool, format: OutputFormat, question: &str) -> Result<bool> {
    if yes || format != OutputFormat::Pretty {
        return Ok(true);
    }
    let confirmed = ui::prompt_confirmation(question)?;
    if !confirmed {
        ui::print_info("Operation cancelled");
    }
    Ok(confirmed)
}

fn tags_suffix(tags: &[String]) -> String {
    if tags.is_empty() {
        String::new()
    } else {
        format!(" · {}", tags.join(", "))
    }
}

fn print_settings<T: serde::Serialize>(title: &str, settings: &T) {
    ui::print_section_header(title);
    let value = serde_json::to_value(settings).unwrap_or_default();
    let entries: Vec<_> = value
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, v)| !v.is_null())
        .collect();
    if entries.is_empty() {
        println!("  {}", ui::dim("(none)"));
    }
    for (key, v) in entries {
        println!("  {}: {}", key, v);
    }
}

fn print_bank(bank: &types::KnowledgeBankResponse) {
    ui::print_section_header(&format!("Knowledge bank: {}", bank.bank_id));
    println!("  Documents: {}", bank.document_count);
    println!(
        "    ready {}, pending {}, failed {}",
        bank.ready_count, bank.pending_count, bank.failed_count
    );
    let mut by_type: Vec<_> = bank.documents_by_media_type.iter().collect();
    by_type.sort();
    for (media_type, count) in by_type {
        println!("    {media_type}: {count}");
    }
    println!("  Passages: {}", bank.passage_count);
    if let Some(created) = &bank.created_at {
        println!("  Created: {created}");
    }
    if let Some(last) = &bank.last_document_at {
        println!("  Last document: {last}");
    }
}

fn print_document(doc: &types::KnowledgeDocumentDetailResponse) {
    ui::print_section_header(&format!("Document: {}", doc.document_id));
    if let Some(filename) = &doc.filename {
        println!("  Filename: {filename}");
    }
    println!("  Type: {} ({})", doc.media_type, doc.content_type);
    println!("  Size: {} bytes", doc.size_bytes);
    println!("  Status: {}", doc.status);
    if let Some(error) = &doc.error {
        println!("  Error: {error}");
    }
    if !doc.tags.is_empty() {
        println!("  Tags: {}", doc.tags.join(", "));
    }
    if let Some(source) = &doc.source_document_id {
        println!("  Split from: {source}");
    }
    println!("  Passages: {}", doc.passage_count);
    for p in &doc.passages {
        println!();
        println!(
            "  {} {}",
            ui::gradient_start(&format!("[{}]", p.index)),
            ui::dim(&locator(p.page, p.start_ms, p.end_ms, &p.media_type))
        );
        if let Some(text) = &p.text {
            println!("    {}", text.trim().replace('\n', "\n    "));
        }
    }
}

fn print_search(response: &types::KnowledgeSearchResponse) {
    ui::print_section_header(&format!("Results ({})", response.results.len()));
    if response.results.is_empty() {
        println!("  {}", ui::dim("No passages found."));
    }
    for (i, hit) in response.results.iter().enumerate() {
        let name = hit.filename.as_deref().unwrap_or(&hit.document_id);
        println!(
            "  {} {}  {}",
            ui::gradient_start(&format!("{}.", i + 1)),
            name,
            ui::dim(&format!(
                "{} [{}] {}",
                hit.document_id,
                hit.index,
                locator(hit.page, hit.start_ms, hit.end_ms, &hit.media_type)
            ))
        );
        if let Some(text) = &hit.text {
            println!("     {}", text.trim().replace('\n', "\n     "));
        }
        println!();
    }
    if let Some(trace) = &response.trace {
        ui::print_section_header("Trace");
        println!(
            "{}",
            serde_json::to_string_pretty(trace).unwrap_or_default()
        );
    }
}

/// Where a passage sits in its document: its page, or its time window.
fn locator(
    page: Option<i64>,
    start_ms: Option<i64>,
    end_ms: Option<i64>,
    media_type: &types::KnowledgeMediaType,
) -> String {
    match (page, start_ms, end_ms) {
        (Some(page), _, _) => format!("{media_type} · page {page}"),
        (None, Some(start), Some(end)) => format!("{media_type} · {}–{}", clock(start), clock(end)),
        _ => media_type.to_string(),
    }
}

fn clock(ms: i64) -> String {
    let s = ms / 1000;
    format!("{}:{:02}", s / 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_updates_reads_json_text_and_unset() {
        let updates = config_updates(
            &[
                "text_passage_tokens=256".into(),
                "disabled_media_types=[\"video\"]".into(),
                "pdf_bands=false".into(),
                "label=plain text".into(),
            ],
            &["max_pdf_pages".into()],
        )
        .unwrap();
        assert_eq!(updates["text_passage_tokens"], json!(256));
        assert_eq!(updates["disabled_media_types"], json!(["video"]));
        assert_eq!(updates["pdf_bands"], json!(false));
        assert_eq!(updates["label"], json!("plain text"));
        assert_eq!(updates["max_pdf_pages"], Value::Null);
    }

    #[test]
    fn config_updates_rejects_bad_input() {
        assert!(config_updates(&["no-equals".into()], &[]).is_err());
        assert!(config_updates(&[], &[]).is_err());
    }

    #[test]
    fn collect_files_walks_folders_and_skips_hidden() {
        let dir = std::env::temp_dir().join(format!("hs-kb-collect-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join("a.md"), "a").unwrap();
        fs::write(dir.join("sub/b.pdf"), "b").unwrap();
        fs::write(dir.join(".DS_Store"), "x").unwrap();
        fs::write(dir.join(".git/config"), "x").unwrap();

        let files = collect_files(std::slice::from_ref(&dir)).unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|p| p.strip_prefix(&dir).unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a.md", "sub/b.pdf"]);
        assert!(collect_files(&[dir.join("missing")]).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn default_download_name_keeps_only_the_last_segment() {
        let name = |f: Option<&str>, id: &str| default_download_name(f, id).ok();
        assert_eq!(
            name(Some("manual.pdf"), "x"),
            Some(PathBuf::from("manual.pdf"))
        );
        assert_eq!(
            name(Some("../../etc/x.pdf"), "x"),
            Some(PathBuf::from("x.pdf"))
        );
        assert_eq!(name(None, "docs/a/b.md"), Some(PathBuf::from("b.md")));
        assert_eq!(name(None, ".."), None);
    }

    #[test]
    fn locator_names_page_or_window() {
        let doc = types::KnowledgeMediaType::Document;
        let video = types::KnowledgeMediaType::Video;
        assert_eq!(locator(Some(3), None, None, &doc), "document · page 3");
        assert_eq!(
            locator(None, Some(65_000), Some(95_000), &video),
            "video · 1:05–1:35"
        );
        assert_eq!(locator(None, None, None, &doc), "document");
    }
}
