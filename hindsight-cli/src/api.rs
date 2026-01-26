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
    pub operation_id: Option<String>,
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
                mission: None,
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
                operation_id: result.operation_id,
            })
        })
    }

    /// Poll an operation until it completes or fails.
    /// Returns Ok(true) if completed successfully, Ok(false) if failed, Err if polling error.
    pub fn poll_operation(&self, agent_id: &str, operation_id: &str, verbose: bool) -> Result<(bool, Option<String>)> {
        self.runtime.block_on(async {
            loop {
                let response = self.client.list_operations(agent_id, None, None, None, None).await?;
                let ops = response.into_inner();

                // Find our operation
                let op = ops.operations.iter().find(|o| o.id == operation_id);

                match op {
                    Some(operation) => {
                        if verbose {
                            eprintln!("Operation {} status: {}", operation_id, operation.status);
                        }
                        match operation.status.as_str() {
                            "pending" => {
                                // Still running, wait and poll again
                                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                            }
                            "completed" => {
                                // Operation completed successfully
                                return Ok((true, None));
                            }
                            "failed" => {
                                return Ok((false, operation.error_message.clone()));
                            }
                            _ => {
                                // Unknown status, treat as failed
                                return Ok((false, Some(format!("Unknown status: {}", operation.status))));
                            }
                        }
                    }
                    None => {
                        // Operation not in list means it completed successfully (removed from pending/failed)
                        return Ok((true, None));
                    }
                }
            }
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
            let response = self.client.list_operations(agent_id, None, None, None, None).await?;
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

    pub fn list_entities(&self, bank_id: &str, limit: Option<i64>, offset: Option<i64>, _verbose: bool) -> Result<types::EntityListResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_entities(bank_id, limit, offset, None).await?;
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

// ============================================================================
// Additional API methods for complete CLI coverage
// ============================================================================

impl ApiClient {
    // --- Memory Methods ---

    pub fn get_memory(&self, bank_id: &str, memory_id: &str, _verbose: bool) -> Result<serde_json::Value> {
        self.runtime.block_on(async {
            let response = self.client.get_memory(bank_id, memory_id, None).await?;
            Ok(response.into_inner())
        })
    }

    // --- Bank Methods ---

    pub fn create_bank(
        &self,
        bank_id: &str,
        request: &types::CreateBankRequest,
        _verbose: bool,
    ) -> Result<types::BankProfileResponse> {
        self.runtime.block_on(async {
            let response = self.client.create_or_update_bank(bank_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn update_bank(
        &self,
        bank_id: &str,
        request: &types::CreateBankRequest,
        _verbose: bool,
    ) -> Result<types::BankProfileResponse> {
        self.runtime.block_on(async {
            let response = self.client.update_bank(bank_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn set_mission(
        &self,
        bank_id: &str,
        mission: &str,
        _verbose: bool,
    ) -> Result<types::BankProfileResponse> {
        self.runtime.block_on(async {
            let request = types::CreateBankRequest {
                name: None,
                mission: Some(mission.to_string()),
                background: None,
                disposition: None,
            };
            let response = self.client.update_bank(bank_id, None, &request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn get_graph(
        &self,
        bank_id: &str,
        type_filter: Option<&str>,
        limit: Option<i64>,
        _verbose: bool,
    ) -> Result<types::GraphDataResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_graph(bank_id, limit, type_filter, None).await?;
            Ok(response.into_inner())
        })
    }

    // --- Tag Methods ---

    pub fn list_tags(
        &self,
        bank_id: &str,
        q: Option<&str>,
        limit: Option<i64>,
        offset: Option<i64>,
        _verbose: bool,
    ) -> Result<types::ListTagsResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_tags(bank_id, limit, offset, q, None).await?;
            Ok(response.into_inner())
        })
    }

    // --- Chunk Methods ---

    pub fn get_chunk(&self, chunk_id: &str, _verbose: bool) -> Result<types::ChunkResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_chunk(chunk_id, None).await?;
            Ok(response.into_inner())
        })
    }

    // --- Operation Methods ---

    pub fn get_operation(&self, bank_id: &str, operation_id: &str, _verbose: bool) -> Result<types::OperationStatusResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_operation_status(bank_id, operation_id, None).await?;
            Ok(response.into_inner())
        })
    }

    // --- Health Methods ---

    pub fn health(&self, _verbose: bool) -> Result<serde_json::Value> {
        self.runtime.block_on(async {
            let response = self.client.health_endpoint_health_get().await?;
            Ok(response.into_inner())
        })
    }

    pub fn metrics(&self, _verbose: bool) -> Result<serde_json::Value> {
        self.runtime.block_on(async {
            let response = self.client.metrics_endpoint_metrics_get().await?;
            Ok(response.into_inner())
        })
    }

    // --- Mental Model Methods ---

    pub fn list_mental_models(&self, bank_id: &str, _verbose: bool) -> Result<types::MentalModelListResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_mental_models(bank_id, None, None, None, None, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn get_mental_model(&self, bank_id: &str, mental_model_id: &str, _verbose: bool) -> Result<types::MentalModelResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_mental_model(bank_id, mental_model_id, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn create_mental_model(
        &self,
        bank_id: &str,
        request: &types::CreateMentalModelRequest,
        _verbose: bool,
    ) -> Result<types::CreateMentalModelResponse> {
        self.runtime.block_on(async {
            let response = self.client.create_mental_model(bank_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn update_mental_model(
        &self,
        bank_id: &str,
        mental_model_id: &str,
        request: &types::UpdateMentalModelRequest,
        _verbose: bool,
    ) -> Result<types::MentalModelResponse> {
        self.runtime.block_on(async {
            let response = self.client.update_mental_model(bank_id, mental_model_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn delete_mental_model(&self, bank_id: &str, mental_model_id: &str, _verbose: bool) -> Result<serde_json::Value> {
        self.runtime.block_on(async {
            let response = self.client.delete_mental_model(bank_id, mental_model_id, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn refresh_mental_model(&self, bank_id: &str, mental_model_id: &str, _verbose: bool) -> Result<types::AsyncOperationSubmitResponse> {
        self.runtime.block_on(async {
            let response = self.client.refresh_mental_model(bank_id, mental_model_id, None).await?;
            Ok(response.into_inner())
        })
    }

    // --- Directive Methods ---

    pub fn list_directives(&self, bank_id: &str, _verbose: bool) -> Result<types::DirectiveListResponse> {
        self.runtime.block_on(async {
            let response = self.client.list_directives(bank_id, None, None, None, None, None, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn get_directive(&self, bank_id: &str, directive_id: &str, _verbose: bool) -> Result<types::DirectiveResponse> {
        self.runtime.block_on(async {
            let response = self.client.get_directive(bank_id, directive_id, None).await?;
            Ok(response.into_inner())
        })
    }

    pub fn create_directive(
        &self,
        bank_id: &str,
        request: &types::CreateDirectiveRequest,
        _verbose: bool,
    ) -> Result<types::DirectiveResponse> {
        self.runtime.block_on(async {
            let response = self.client.create_directive(bank_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn update_directive(
        &self,
        bank_id: &str,
        directive_id: &str,
        request: &types::UpdateDirectiveRequest,
        _verbose: bool,
    ) -> Result<types::DirectiveResponse> {
        self.runtime.block_on(async {
            let response = self.client.update_directive(bank_id, directive_id, None, request).await?;
            Ok(response.into_inner())
        })
    }

    pub fn delete_directive(&self, bank_id: &str, directive_id: &str, _verbose: bool) -> Result<serde_json::Value> {
        self.runtime.block_on(async {
            let response = self.client.delete_directive(bank_id, directive_id, None).await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_deserialize() {
        let json = r#"{
            "id": "test-op-123",
            "task_type": "retain",
            "items_count": 5,
            "document_id": "doc-456",
            "created_at": "2024-01-15T10:00:00Z",
            "status": "pending",
            "error_message": null
        }"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        assert_eq!(op.id, "test-op-123");
        assert_eq!(op.task_type, "retain");
        assert_eq!(op.items_count, 5);
        assert_eq!(op.document_id, Some("doc-456".to_string()));
        assert_eq!(op.status, "pending");
        assert!(op.error_message.is_none());
    }

    #[test]
    fn test_operation_deserialize_with_error() {
        let json = r#"{
            "id": "test-op-456",
            "task_type": "retain",
            "items_count": 3,
            "document_id": null,
            "created_at": "2024-01-15T10:00:00Z",
            "status": "failed",
            "error_message": "Something went wrong"
        }"#;
        let op: Operation = serde_json::from_str(json).unwrap();
        assert_eq!(op.status, "failed");
        assert_eq!(op.error_message, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_memory_put_result_serialize() {
        let result = MemoryPutResult {
            success: true,
            items_count: 10,
            message: "Stored 10 memory units".to_string(),
            is_async: true,
            operation_id: Some("op-789".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"items_count\":10"));
        assert!(json.contains("\"is_async\":true"));
        assert!(json.contains("\"operation_id\":\"op-789\""));
    }

    #[test]
    fn test_memory_put_result_without_operation_id() {
        let result = MemoryPutResult {
            success: true,
            items_count: 5,
            message: "Stored 5 memory units".to_string(),
            is_async: false,
            operation_id: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"operation_id\":null"));
    }

    #[test]
    fn test_operations_response_deserialize() {
        let json = r#"{
            "bank_id": "test-bank",
            "operations": [
                {
                    "id": "op-1",
                    "task_type": "retain",
                    "items_count": 2,
                    "document_id": null,
                    "created_at": "2024-01-15T10:00:00Z",
                    "status": "pending",
                    "error_message": null
                },
                {
                    "id": "op-2",
                    "task_type": "retain",
                    "items_count": 3,
                    "document_id": "doc-123",
                    "created_at": "2024-01-15T11:00:00Z",
                    "status": "completed",
                    "error_message": null
                }
            ]
        }"#;
        let ops: OperationsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(ops.bank_id, "test-bank");
        assert_eq!(ops.operations.len(), 2);
        assert_eq!(ops.operations[0].status, "pending");
        assert_eq!(ops.operations[1].status, "completed");
    }
}
