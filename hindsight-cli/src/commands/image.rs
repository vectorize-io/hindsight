use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;
use anyhow::{Context, Result};
use hindsight_client::ImageRetainFile;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[allow(clippy::too_many_arguments)]
pub fn retain(
    client: &ApiClient,
    bank_id: &str,
    paths: Vec<PathBuf>,
    content: Option<String>,
    context: Option<String>,
    document_id: Option<String>,
    timestamp: Option<String>,
    tags: Vec<String>,
    strategy: Option<String>,
    update_mode: String,
    asset_ids: Vec<String>,
    image_context: Option<String>,
    idempotency_key: Option<String>,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if !asset_ids.is_empty() && asset_ids.len() != paths.len() {
        anyhow::bail!("--asset-id must be omitted or supplied once per image");
    }

    let files = paths
        .iter()
        .map(|path| load_image(path))
        .collect::<Result<Vec<ImageRetainFile>>>()?;
    let images = paths
        .iter()
        .enumerate()
        .map(|(index, _)| {
            json!({
                "asset_id": asset_ids.get(index),
                "context": image_context.as_ref(),
            })
        })
        .collect::<Vec<Value>>();
    let request = json!({
        "images": images,
        "content": content,
        "context": context,
        "document_id": document_id,
        "timestamp": timestamp,
        "tags": if tags.is_empty() { None } else { Some(tags) },
        "strategy": strategy,
        "update_mode": update_mode,
    });

    let result =
        client.image_retain(bank_id, files, request, idempotency_key.as_deref(), verbose)?;
    if output_format == OutputFormat::Pretty {
        ui::print_success("Images queued for semantic analysis");
        println!("  Document ID: {}", result.document_id);
        println!("  Operation ID: {}", result.operation_id);
        for asset in result.image_assets {
            println!("  Asset ID: {}", asset.asset_id);
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}

pub fn list(
    client: &ApiClient,
    bank_id: &str,
    document_id: Option<String>,
    status: String,
    limit: u64,
    offset: u64,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let result = client.list_image_assets(
        bank_id,
        document_id.as_deref(),
        &status,
        limit,
        offset,
        verbose,
    )?;
    if output_format == OutputFormat::Pretty {
        ui::print_info(&format!(
            "Image assets for bank '{}' (total: {})",
            bank_id, result.total
        ));
        for asset in result.items {
            println!("\n  Asset ID: {}", asset.asset_id);
            println!("    Type: {}", asset.mime_type);
            println!("    Size: {} bytes", asset.size_bytes);
            println!("    Dimensions: {}x{}", asset.width, asset.height);
            println!("    Status: {}", asset.status);
        }
    } else {
        output::print_output(&result, output_format)?;
    }
    Ok(())
}

pub fn get(
    client: &ApiClient,
    bank_id: &str,
    asset_id: &str,
    out: PathBuf,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let result = client.get_image_asset(bank_id, asset_id, verbose)?;
    fs::write(&out, &result.content)
        .with_context(|| format!("Failed to write image: {}", out.display()))?;
    if output_format == OutputFormat::Pretty {
        ui::print_success(&format!("Image saved to {}", out.display()));
        println!("  Asset ID: {}", result.descriptor.asset_id);
        println!("  Type: {}", result.descriptor.mime_type);
        println!("  Size: {} bytes", result.descriptor.size_bytes);
    } else {
        output::print_output(&result.descriptor, output_format)?;
    }
    Ok(())
}

pub fn delete(
    client: &ApiClient,
    bank_id: &str,
    asset_id: &str,
    yes: bool,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    if !yes
        && !ui::prompt_confirmation(&format!(
            "Are you sure you want to delete image asset '{}'? This cannot be undone.",
            asset_id
        ))?
    {
        ui::print_info("Delete cancelled");
        return Ok(());
    }
    client.delete_image_asset(bank_id, asset_id, verbose)?;
    if output_format == OutputFormat::Pretty {
        ui::print_success("Image asset deleted");
    } else {
        output::print_output(
            &json!({ "deleted": true, "asset_id": asset_id }),
            output_format,
        )?;
    }
    Ok(())
}

fn load_image(path: &Path) -> Result<ImageRetainFile> {
    let mime_type = match path.extension().and_then(|value| value.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("jpg") => "image/jpeg",
        Some(extension) if extension.eq_ignore_ascii_case("jpeg") => "image/jpeg",
        Some(extension) if extension.eq_ignore_ascii_case("png") => "image/png",
        Some(extension) if extension.eq_ignore_ascii_case("webp") => "image/webp",
        _ => anyhow::bail!("Unsupported image format: {}", path.display()),
    };
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .with_context(|| format!("Invalid image filename: {}", path.display()))?;
    Ok(ImageRetainFile {
        filename: filename.to_string(),
        mime_type: mime_type.to_string(),
        content: fs::read(path)
            .with_context(|| format!("Failed to read image: {}", path.display()))?,
    })
}
