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
//!     let banks = client.list_banks(None).await?;
//!     println!("Found {} banks", banks.into_inner().banks.len());
//!
//!     Ok(())
//! }
//! ```

// Include the generated client code (which already exports Error and ResponseValue)
include!(concat!(env!("OUT_DIR"), "/hindsight_client_generated.rs"));

/// Semantic version of this Rust client, kept in sync with the other language
/// wrappers when a coordinated release is cut.
pub const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default `User-Agent` header sent on every request unless overridden.
pub const DEFAULT_USER_AGENT: &str = concat!("hindsight-client-rust/", env!("CARGO_PKG_VERSION"));

/// Build a [`reqwest::Client`] with the given `User-Agent` header.
///
/// Integrations should use this to identify themselves (e.g.
/// `"hindsight-cli/0.6.2"`) so self-hosted deployments behind Cloudflare or
/// other UA-based filters accept the traffic. Pass the resulting client to
/// [`Client::new_with_client`].
pub fn reqwest_client_with_user_agent(
    user_agent: impl Into<String>,
) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder().user_agent(user_agent.into()).build()
}

/// Construct a [`Client`] with a custom `User-Agent` header.
///
/// Equivalent to [`Client::new`] but sets the UA string. Use this instead of
/// the bare `Client::new` when pointing at a hosted Hindsight deployment.
pub fn client_with_user_agent(
    base_url: &str,
    user_agent: impl Into<String>,
) -> Result<Client, reqwest::Error> {
    let http = reqwest_client_with_user_agent(user_agent)?;
    Ok(Client::new_with_client(base_url, http))
}

/// Construct a [`Client`] with the default Hindsight `User-Agent`.
///
/// Prefer this over `Client::new` — the bare `Client::new` uses reqwest's
/// default UA which is blocked by some reverse proxies (e.g. Cloudflare).
pub fn default_client(base_url: &str) -> Result<Client, reqwest::Error> {
    client_with_user_agent(base_url, DEFAULT_USER_AGENT)
}

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

        // Use a custom reqwest client with longer timeout for LLM operations
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        let client = Client::new_with_client(&api_url, http_client);

        // Generate unique bank ID for this test
        let bank_id = format!("rust-test-{}", uuid::Uuid::new_v4());

        // 1. Create a bank
        let create_request = types::CreateBankRequest {
            name: Some(format!("Rust Test Bank")),
            ..Default::default()
        };
        let create_response = client
            .create_or_update_bank(&bank_id, None, &create_request)
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
                    entities: None,
                    tags: None,
                    observation_scopes: None,
                    strategy: None,
                    update_mode: None,
                },
                types::MemoryItem {
                    content: "Bob works with Alice on the search team".to_string(),
                    context: None,
                    document_id: None,
                    metadata: None,
                    timestamp: None,
                    entities: None,
                    tags: None,
                    observation_scopes: None,
                    strategy: None,
                    update_mode: None,
                },
            ],
            document_tags: None,
        };
        let retain_response = client
            .retain_memories(&bank_id, None, &retain_request)
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
            tags: None,
            tags_match: types::TagsMatch::Any,
            tag_groups: None,
        };
        let recall_response = client
            .recall_memories(&bank_id, None, &recall_request)
            .await
            .expect("Failed to recall memories");
        let recall_result = recall_response.into_inner();
        assert!(!recall_result.results.is_empty(), "Should recall at least one memory");

        // Cleanup: delete the test bank's memories
        let _ = client.clear_bank_memories(&bank_id, None, None).await;
    }
}
