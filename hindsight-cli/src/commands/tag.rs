//! Tag commands for listing tags in a memory bank.

use anyhow::Result;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

/// List tags in a bank
pub fn list(
    client: &ApiClient,
    bank_id: &str,
    query: Option<String>,
    limit: i64,
    offset: i64,
    verbose: bool,
    output_format: OutputFormat,
) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching tags..."))
    } else {
        None
    };

    let response = client.list_tags(
        bank_id,
        query.as_deref(),
        Some(limit),
        Some(offset),
        verbose,
    );

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header(&format!("Tags: {}", bank_id));

                if result.items.is_empty() {
                    println!("  {}", ui::dim("No tags found."));
                } else {
                    for (i, tag) in result.items.iter().enumerate() {
                        let t = i as f32 / result.items.len().max(1) as f32;
                        println!(
                            "  {} {}",
                            ui::gradient(&tag.tag, t),
                            ui::dim(&format!("({})", tag.count))
                        );
                    }
                    println!();
                    println!("  {} {} total", ui::dim("Total:"), result.total);
                }
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
    use hindsight_client::types::{ListTagsResponse, TagItem};

    #[test]
    fn test_tag_item_fields() {
        // Verify TagItem has the expected fields
        let tag = TagItem {
            tag: "test-tag".to_string(),
            count: 5,
        };

        assert_eq!(tag.tag, "test-tag");
        assert_eq!(tag.count, 5);
    }

    #[test]
    fn test_list_tags_response_deserialization() {
        let json = r#"{
            "items": [
                {"tag": "user", "count": 10},
                {"tag": "system", "count": 5}
            ],
            "limit": 100,
            "offset": 0,
            "total": 2
        }"#;

        let result: ListTagsResponse = serde_json::from_str(json).unwrap();

        assert_eq!(result.items.len(), 2);
        assert_eq!(result.items[0].tag, "user");
        assert_eq!(result.items[0].count, 10);
        assert_eq!(result.items[1].tag, "system");
        assert_eq!(result.items[1].count, 5);
        assert_eq!(result.total, 2);
        assert_eq!(result.limit, 100);
        assert_eq!(result.offset, 0);
    }

    #[test]
    fn test_empty_tags_response() {
        let json = r#"{
            "items": [],
            "limit": 100,
            "offset": 0,
            "total": 0
        }"#;

        let result: ListTagsResponse = serde_json::from_str(json).unwrap();

        assert!(result.items.is_empty());
        assert_eq!(result.total, 0);
    }
}
