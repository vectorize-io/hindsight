//! Hindsight API Client
//!
//! A Rust client library for the Hindsight semantic memory system API.
//!
//! # Example
//!
//! ```rust,no_run
//! use hindsight_client::Client;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::new("http://localhost:8888");
//!
//!     // List memory banks
//!     let banks = client.list_banks().await?;
//!     println!("Found {} banks", banks.into_inner().len());
//!
//!     Ok(())
//! }
//! ```

// Include the generated client code (which already exports Error and ResponseValue)
include!(concat!(env!("OUT_DIR"), "/hindsight_client_generated.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let _client = Client::new("http://localhost:8888");
        // Just verify we can create a client
        assert!(true);
    }

    #[tokio::test]
    async fn test_memory_lifecycle() {
        let api_url = std::env::var("HINDSIGHT_API_URL")
            .unwrap_or_else(|_| "http://localhost:8888".to_string());
        let client = Client::new(&api_url);

        // Generate unique bank ID for this test
        let bank_id = format!("rust-test-{}", uuid::Uuid::new_v4());

        // 1. Create a bank
        let create_request = types::CreateBankRequest {
            name: Some(format!("Rust Test Bank")),
            ..Default::default()
        };
        let create_response = client
            .create_or_update_bank(&bank_id, &create_request)
            .await
            .expect("Failed to create bank");
        assert_eq!(create_response.into_inner().bank_id, bank_id);

        // 2. Retain some memories
        let retain_request = types::RetainRequest {
            async_: false,
            items: vec![
                types::MemoryItem {
                    content: "Alice is a software engineer at Google".to_string(),
                    context: None,
                    document_id: None,
                    metadata: None,
                    timestamp: None,
                },
                types::MemoryItem {
                    content: "Bob works with Alice on the search team".to_string(),
                    context: None,
                    document_id: None,
                    metadata: None,
                    timestamp: None,
                },
            ],
        };
        let retain_response = client
            .retain_memories(&bank_id, &retain_request)
            .await
            .expect("Failed to retain memories");
        assert!(retain_response.into_inner().success);

        // 3. Recall memories
        let recall_request = types::RecallRequest {
            query: "Who is Alice?".to_string(),
            max_tokens: 4096,
            trace: false,
            budget: None,
            include: None,
            query_timestamp: None,
            types: None,
        };
        let recall_response = client
            .recall_memories(&bank_id, &recall_request)
            .await
            .expect("Failed to recall memories");
        let recall_result = recall_response.into_inner();
        assert!(!recall_result.results.is_empty(), "Should recall at least one memory");

        // 4. Reflect on a question
        let reflect_request = types::ReflectRequest {
            query: "What do you know about Alice?".to_string(),
            budget: None,
            context: None,
            include: None,
        };
        let reflect_response = client
            .reflect(&bank_id, &reflect_request)
            .await
            .expect("Failed to reflect");
        let reflect_result = reflect_response.into_inner();
        assert!(!reflect_result.text.is_empty(), "Reflect should return some text");

        // Cleanup: delete the test bank's memories
        let _ = client.clear_bank_memories(&bank_id, None).await;
    }
}
