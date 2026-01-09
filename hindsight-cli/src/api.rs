//! API client wrapper
//!
//! This module provides a thin wrapper around the auto-generated hindsight-client
//! to bridge from the CLI's synchronous code to the async API client.

use anyhow::Result;
use hindsight_client::Client as AsyncClient;
pub use hindsight_client::types;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;

// Types not defined in OpenAPI spec (TODO: add to openapi.json)
#[derive(Debug, Serialize, Deserialize)]
pub struct AgentStats {
    pub bank_id: String,
    pub total_nodes: i32,
    pub total_links: i32,
    pub total_documents: i32,
    pub nodes_by_fact_type: HashMap<String, i32>,
    pub links_by_link_type: HashMap<String, i32>,
    pub links_by_fact_type: HashMap<String, i32>,
    pub links_breakdown: HashMap<String, HashMap<String, i32>>,
    pub pending_operations: i32,
    pub failed_operations: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Operation {
    pub id: String,
    pub task_type: String,
    pub items_count: i32,
    pub document_id: Option<String>,
    pub created_at: String,
    pub status: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OperationsResponse {
    pub bank_id: String,
    pub operations: Vec<Operation>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TraceInfo {
    pub total_time: Option<f64>,
    pub activation_count: Option<i32>,
}

// Unified result for put_memories that handles both sync and async responses
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryPutResult {
    pub success: bool,
    pub items_count: i64,
    pub message: String,
    pub is_async: bool,
}

#[derive(Clone)]
pub struct ApiClient {
    client: AsyncClient,
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
}

impl ApiClient {
    pub fn new(base_url: String, api_key: Option<String>) -> Result<Self> {
        let runtime = std::sync::Arc::new(tokio::runtime::Runtime::new()?);

        // Create HTTP client with 2-minute timeout and optional auth header
        let mut client_builder = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120));

        if let Some(key) = api_key {
            let mut headers = reqwest::header::HeaderMap::new();
            let auth_value = format!("Bearer {}", key);
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&auth_value)?,
            );
            client_builder = client_builder.default_headers(headers);
        }

        let http_client = client_builder.build()?;

        let client = AsyncClient::new_with_client(&base_url, http_client);
        Ok(ApiClient { client, runtime })
    }

    pub fn list_agents(&self, _verbose: bool) -> Result<Vec<types::BankListItem>> {
        self.runtime.block_on(async {
            let response = self.client.list_banks(None).await?;
            Ok(response.into_inner().banks)
        })
    }

    pub fn get_profile(&self, agent_id: &str, _verbose: bool) -> Result<types::BankProfileResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_bank_profile(agent_id, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn get_stats(&self, agent_id: &str, _verbose: bool) -> Result<AgentStats> {
        self.runtime.block_on(async {
            let response = self.client.get_agent_stats(agent_id, None).await?;
            let value = response.into_inner();
            // Convert to JSON Value first, then parse into our type
            let json_value = serde_json::to_value(&value)?;
            let stats: AgentStats = serde_json::from_value(json_value)?;
            Ok(stats)
        })
    }

    pub fn update_agent_name(&self, agent_id: &str, name: &str, _verbose: bool) -> Result<types::BankProfileResponse> {
        self.runtime.block_on(async {
            let request = types::CreateBankRequest {
                name: Some(name.to_string()),
                background: None,
                disposition: None,
            };
            let response = self.client.create_or_update_bank(agent_id, None, &request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn add_background(&self, agent_id: &str, content: &str, update_disposition: bool, _verbose: bool) -> Result<types::BackgroundResponse> {
        self.runtime.block_on(async {
            let request = types::AddBackgroundRequest {
                content: content.to_string(),
                update_disposition,
            };
            let response = self.client.add_bank_background(agent_id, None, &request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn recall(&self, agent_id: &str, request: &types::RecallRequest, verbose: bool) -> Result<types::RecallResponse> {
        if verbose {
            eprintln!("Request body: {}", serde_json::to_string_pretty(request).unwrap_or_default());
        }
        self.runtime.block_on(async {
            let response = self.client.recall_memories(agent_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn reflect(&self, agent_id: &str, request: &types::ReflectRequest, _verbose: bool) -> Result<types::ReflectResponse> {
        self.runtime.block_on(async {
            let response = self.client.reflect(agent_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn retain(&self, agent_id: &str, request: &types::RetainRequest, _async_mode: bool, _verbose: bool) -> Result<MemoryPutResult> {
        self.runtime.block_on(async {
            let response = self.client.retain_memories(agent_id, None, request).await?;
            let result = response.into_inner();
            Ok(MemoryPutResult {
                success: result.success,
                items_count: result.items_count,
                message: format!("Stored {} memory units", result.items_count),
                is_async: result.async_,
            })
        })
    }

    pub fn delete_memory(&self, _agent_id: &str, _unit_id: &str, _verbose: bool) -> Result<types::DeleteResponse> {
        // Note: Individual memory deletion is no longer supported in the API
        anyhow::bail!("Individual memory deletion is no longer supported. Use 'memory clear' to clear all memories.")
    }

    pub fn clear_memories(&self, agent_id: &str, fact_type: Option<&str>, _verbose: bool) -> Result<types::DeleteResponse> {
        self.runtime.block_on(async {
            let response = self.client.clear_bank_memories(agent_id, None, Some(fact_type)).await?;
            Ok(response.into_inner())
        })
    }

    pub fn list_documents(&self, agent_id: &str, q: Option<&str>, limit: Option<i32>, offset: Option<i32>, _verbose: bool) -> Result<types::ListDocumentsResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_documents(
                agent_id,
                limit.map(|l| l as i64),
                offset.map(|o| o as i64),
                q,
                None,
            ).await?;
            Ok(response.into_inner())
        })
    }

    pub fn get_document(&self, agent_id: &str, document_id: &str, _verbose: bool) -> Result<types::DocumentResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_document(agent_id, document_id, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn delete_document(&self, agent_id: &str, document_id: &str, _verbose: bool) -> Result<types::DeleteResponse> {
        self.runtime.block_on(async {
            let response = self.client.delete_document(agent_id, document_id, None).await?;
            let value = response.into_inner();
            // Convert typed response to DeleteResponse
            Ok(types::DeleteResponse {
                deleted_count: Some(value.memory_units_deleted),
                message: Some(value.message),
                success: value.success,
            })
        })
    }

    pub fn list_operations(&self, agent_id: &str, _verbose: bool) -> Result<OperationsResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_operations(agent_id, None).await?;
            let value = response.into_inner();
            // Convert to JSON Value first, then parse into our type
            let json_value = serde_json::to_value(&value)?;
            let ops: OperationsResponse = serde_json::from_value(json_value)?;
            Ok(ops)
        })
    }

    pub fn cancel_operation(&self, agent_id: &str, operation_id: &str, _verbose: bool) -> Result<types::DeleteResponse> {
        self.runtime.block_on(async {
            let response = self.client.cancel_operation(agent_id, operation_id, None).await?;
            let value = response.into_inner();
            // Convert typed response to DeleteResponse
            Ok(types::DeleteResponse {
                deleted_count: None,
                message: Some(value.message),
                success: value.success,
            })
        })
    }

    pub fn list_memories(&self, bank_id: &str, type_filter: Option<&str>, q: Option<&str>, limit: Option<i64>, offset: Option<i64>, _verbose: bool) -> Result<types::ListMemoryUnitsResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_memories(bank_id, limit, offset, q, type_filter, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn list_entities(&self, bank_id: &str, limit: Option<i64>, _verbose: bool) -> Result<types::EntityListResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_entities(bank_id, limit, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn get_entity(&self, bank_id: &str, entity_id: &str, _verbose: bool) -> Result<types::EntityDetailResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_entity(bank_id, entity_id, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn regenerate_entity(&self, bank_id: &str, entity_id: &str, _verbose: bool) -> Result<types::EntityDetailResponse> {
        self.runtime.block_on(async {
            let response = self.client.regenerate_entity_observations(bank_id, entity_id, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn delete_bank(&self, bank_id: &str, _verbose: bool) -> Result<types::DeleteResponse> {
        self.runtime.block_on(async {
            let response = self.client.delete_bank(bank_id, None).await?;
            Ok(response.into_inner())
        })
    }
}

// Re-export types from the generated client for use in commands
pub use types::{
    BankProfileResponse,
    MemoryItem,
    RecallRequest,
    RecallResponse,
    RecallResult,
    ReflectRequest,
    ReflectResponse,
    RetainRequest,
};
