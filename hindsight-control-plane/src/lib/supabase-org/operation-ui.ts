type ApiKeyOperation = string;

export interface OperationUiSection {
  id: string;
  labelKey: string;
  defaultLabel: string;
  operations: readonly ApiKeyOperation[];
}

export interface OperationUiGroup {
  id: string;
  labelKey: string;
  defaultLabel: string;
  bankScoped: boolean;
  operations: readonly ApiKeyOperation[];
  sections?: readonly OperationUiSection[];
}

export const OPERATION_UI_GROUPS: readonly OperationUiGroup[] = [
  {
    id: "key_level",
    labelKey: "authProfiles.supabaseOrg.operations.groups.createBank",
    defaultLabel: "Create bank",
    bankScoped: false,
    operations: ["create_bank"],
  },
  {
    id: "bank_management",
    labelKey: "authProfiles.supabaseOrg.operations.groups.bankManagement",
    defaultLabel: "Bank management",
    bankScoped: true,
    operations: [
      "export_bank_template",
      "import_bank_template",
      "get_bank_config",
      "get_bank_profile",
      "get_bank_stats",
      "update_bank",
      "update_bank_config",
      "update_bank_disposition",
      "merge_bank_mission",
      "reset_bank_config",
      "delete_bank",
      "list_audit_logs",
      "audit_log_stats",
      "list_llm_requests",
      "llm_request_stats",
      "list_directives",
      "get_directive",
      "create_directive",
      "update_directive",
      "delete_directive",
    ],
  },
  {
    id: "memory_and_mental_models",
    labelKey: "authProfiles.supabaseOrg.operations.groups.memoryAndMentalModels",
    defaultLabel: "Memory & mental models",
    bankScoped: true,
    operations: [
      "retain",
      "recall",
      "reflect",
      "export_documents",
      "import_documents",
      "import_bank",
      "get_memories_timeseries",
      "get_memory_unit",
      "list_memory_units",
      "update_memory_unit",
      "get_document",
      "list_documents",
      "update_document",
      "delete_document",
      "list_document_chunks",
      "get_chunk",
      "reprocess_document",
      "list_tags",
      "list_entities",
      "get_entity",
      "get_entity_graph",
      "get_graph_data",
      "list_observation_scopes",
      "get_observation_history",
      "clear_observations",
      "clear_observations_for_memory",
      "submit_async_graph_maintenance",
      "submit_async_consolidation",
      "retry_failed_consolidation",
      "list_mental_models",
      "list_mental_model_tags",
      "mental_model_get",
      "mental_model_refresh",
      "create_mental_model",
      "update_mental_model",
      "delete_mental_model",
      "clear_mental_model",
    ],
  },
  {
    id: "operations_and_automation",
    labelKey: "authProfiles.supabaseOrg.operations.groups.operationsAndAutomation",
    defaultLabel: "Operations & automation",
    bankScoped: true,
    operations: [
      "list_operations",
      "get_operation_status",
      "cancel_operation",
      "retry_operation",
      "list_webhooks",
      "list_webhook_deliveries",
      "create_webhook",
      "update_webhook",
      "delete_webhook",
    ],
  },
];
