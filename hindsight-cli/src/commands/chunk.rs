//! Chunk commands for retrieving document chunks.

use anyhow::Result;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

/// Get a specific chunk by ID
pub fn get(
    client: &ApiClient,
    chunk_id: &str,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching chunk..."))
    } else {
        None
    };

    let response = client.get_chunk(chunk_id, verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Chunk: {}", chunk_id));

                println!("  {} {}", ui::dim("ID:"), result.chunk_id);
                println!("  {} {}", ui::dim("Index:"), result.chunk_index);
                println!("  {} {}", ui::dim("Document:"), result.document_id);
                println!("  {} {}", ui::dim("Bank:"), result.bank_id);
                println!("  {} {}", ui::dim("Created:"), result.created_at);

                println!();
                println!("{}", ui::gradient_text("─── Content ───"));
                println!();
                println!("{}", result.chunk_text);

                println!();
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use hindsight_client::types::ChunkResponse;

    #[test]
    fn test_chunk_response_deserialization() {
        let json = r#"{
            "chunk_id": "chunk-123",
            "bank_id": "test-bank",
            "document_id": "doc-456",
            "chunk_index": 0,
            "chunk_text": "This is the chunk content.",
            "created_at": "2024-01-15T10:00:00Z"
        }"#;

        let result: ChunkResponse = serde_json::from_str(json).unwrap();

        assert_eq!(result.chunk_id, "chunk-123");
        assert_eq!(result.bank_id, "test-bank");
        assert_eq!(result.document_id, "doc-456");
        assert_eq!(result.chunk_index, 0);
        assert_eq!(result.chunk_text, "This is the chunk content.");
        assert_eq!(result.created_at, "2024-01-15T10:00:00Z");
    }

    #[test]
    fn test_chunk_response_multiline_content() {
        let json = r#"{
            "chunk_id": "chunk-456",
            "bank_id": "test-bank",
            "document_id": "doc-789",
            "chunk_index": 5,
            "chunk_text": "Line 1\nLine 2\nLine 3",
            "created_at": "2024-01-15T11:00:00Z"
        }"#;

        let result: ChunkResponse = serde_json::from_str(json).unwrap();

        assert_eq!(result.chunk_index, 5);
        assert!(result.chunk_text.contains('\n'));
        assert_eq!(result.chunk_text.lines().count(), 3);
    }
}
