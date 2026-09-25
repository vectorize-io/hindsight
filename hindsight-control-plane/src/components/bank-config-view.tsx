"use client";

import { useState, useEffect, useRef, useMemo, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { client, type BankAliasEntry } from "@/lib/api";
import { PreviewPromptButton } from "@/components/prompt-preview-dialog";
import { TagFilterInput } from "@/components/tag-filter-input";
import type {
  ConsolidationStrategiesPreview,
  StrategyRulePreview,
  StrategyScopePreview,
} from "@/lib/api";
import {
  MentalModelTriggerFields,
  TriggerSummary,
  triggerFormFromTrigger,
  triggerFromForm,
  type TriggerForm,
} from "@/components/mental-model-trigger-fields";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  EntityLabelsEditor,
  type LabelGroup,
  type LabelValue,
  type MapField,
} from "@/components/entity-labels-editor";
import {
  deserializeRetainStrategies,
  serializeRetainStrategies,
  type RetainStrategy,
  type RetainStrategyValues,
} from "@/lib/retain-strategy-config";
import {
  mergeObservationsOverrides,
  mergeResolvedObservations,
  observationsSlice,
  reconcileObservationsEdits,
  type ObservationsEdits,
  type ConsolidationSettings,
  type ConsolidationStrategy,
  type ScopePattern,
  type StrategyTagsMatch,
  compactStrategy,
  scopesLabel,
  strategyOverridesSomething,
  suggestedTags,
} from "@/lib/observations-config";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { AlertCircle, Plus, Trash2, ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";
import { IdChip } from "@/components/ui/facet-chip";
import { Spinner } from "@/components/ui/spinner";
import { Card } from "@/components/ui/card";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ProfileData {
  reflect_mission: string;
  disposition_skepticism: number;
  disposition_literalism: number;
  disposition_empathy: number;
}

type RetainEdits = {
  retain_context_chars: number | null;
  retain_chunk_size: number | null;
  retain_structured_chunk_size: number | null;
  retain_extraction_mode: string | null;
  retain_mission: string | null;
  retain_custom_instructions: string | null;
  entities_allow_free_form: boolean | null;
  entity_labels: LabelGroup[] | null;
};

type StrategiesEdits = {
  retain_default_strategy: string | null;
  retain_strategies: Record<string, Record<string, any>> | null;
};

type MCPEdits = {
  mcp_enabled_tools: string[] | null;
};

type GeminiSafetySetting = {
  category: string;
  threshold: string;
};

type GeminiEdits = {
  llm_gemini_safety_settings: GeminiSafetySetting[] | null;
};

type AuditEdits = {
  // null = no bank override, inherit the server default. Distinct from an
  // explicit false, which overrides a server default of true.
  audit_log_enabled: boolean | null;
};

type DocStorageEdits = {
  // null = inherit the server default. Explicit false keeps only derived
  // facts (documents.original_text NULL, chunks.chunk_text empty).
  store_document_text: boolean | null;
};

// Mental models and the knowledge pages backed by them. null = inherit the
// server default.
type MentalModelsEdits = {
  mental_model_min_refresh_interval_seconds: number | null;
};

// The bank's default reflect options (reflect_default_options). Stored as one
// object, edited as two fields; null means "not set", so reflect falls back to
// the shipped default.
type ReflectOptionsEdits = {
  reflect_search_observations_max_tokens: number | null;
  reflect_search_observations_include_entities: boolean | null;
};

// The server's built-in knowledge-page trigger (MemoryEngine.KNOWLEDGE_PAGE_DEFAULT_TRIGGER).
// The configured default merges over it, so the form starts from the pair to show
// what a new page actually gets.
const KNOWLEDGE_PAGE_BUILTIN_TRIGGER = {
  mode: "delta",
  fact_types: ["observation"],
  exclude_mental_models: true,
  refresh_after_consolidation: true,
} as const;

function effectivePageTrigger(configured: Record<string, any> | null | undefined): TriggerForm {
  const merged: Record<string, any> = { ...KNOWLEDGE_PAGE_BUILTIN_TRIGGER, ...configured };
  // Same exclusivity rule as the server's merge: a configured cron replaces the
  // built-in refresh-after-consolidation.
  if (configured?.refresh_cron && configured.refresh_after_consolidation === undefined)
    merged.refresh_after_consolidation = false;
  return triggerFormFromTrigger(merged);
}

// Recall pipeline stages. null = inherit the server default (all four ship
// enabled); explicit false switches that stage off for this bank, trading
// recall breadth for latency. Semantic always runs — it is the baseline arm.
type RecallEdits = {
  enable_text_search: boolean | null;
  enable_temporal_retrieval: boolean | null;
  enable_graph_retrieval: boolean | null;
  enable_reranking: boolean | null;
};

// ─── Gemini safety settings catalogue ────────────────────────────────────────

const GEMINI_HARM_CATEGORY_VALUES = [
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
] as const;

function getGeminiHarmCategories(t: (key: string) => string): { value: string; label: string }[] {
  return [
    { value: "HARM_CATEGORY_HARASSMENT", label: t("geminiCategoryHarassment") },
    { value: "HARM_CATEGORY_HATE_SPEECH", label: t("geminiCategoryHateSpeech") },
    { value: "HARM_CATEGORY_SEXUALLY_EXPLICIT", label: t("geminiCategorySexuallyExplicit") },
    { value: "HARM_CATEGORY_DANGEROUS_CONTENT", label: t("geminiCategoryDangerousContent") },
  ];
}

function getGeminiThresholds(t: (key: string) => string): { value: string; label: string }[] {
  return [
    { value: "HARM_BLOCK_THRESHOLD_UNSPECIFIED", label: t("geminiThresholdUnspecified") },
    { value: "OFF", label: t("geminiThresholdOff") },
    { value: "BLOCK_NONE", label: t("geminiThresholdBlockNone") },
    { value: "BLOCK_LOW_AND_ABOVE", label: t("geminiThresholdBlockLowAndAbove") },
    { value: "BLOCK_MEDIUM_AND_ABOVE", label: t("geminiThresholdBlockMediumAndAbove") },
    { value: "BLOCK_ONLY_HIGH", label: t("geminiThresholdBlockOnlyHigh") },
  ];
}

const DEFAULT_GEMINI_SAFETY_SETTINGS: GeminiSafetySetting[] = GEMINI_HARM_CATEGORY_VALUES.map(
  (value) => ({
    category: value,
    threshold: "BLOCK_NONE",
  })
);

// ─── MCP tool catalogue ───────────────────────────────────────────────────────

type McpToolGroup = { key: string; label: string; tools: string[] };

function getMcpToolGroups(t: (key: string) => string): McpToolGroup[] {
  return [
    {
      key: "core",
      label: t("mcpGroupCore"),
      tools: ["retain", "sync_retain", "recall", "reflect"],
    },
    {
      key: "bankManagement",
      label: t("mcpGroupBankManagement"),
      tools: [
        "list_banks",
        "create_bank",
        "get_bank",
        "get_bank_stats",
        "update_bank",
        "delete_bank",
        "clear_memories",
      ],
    },
    {
      key: "mentalModels",
      label: t("mcpGroupMentalModels"),
      tools: [
        "list_mental_models",
        "get_mental_model",
        "create_mental_model",
        "update_mental_model",
        "delete_mental_model",
        "refresh_mental_model",
        "clear_mental_model",
      ],
    },
    {
      key: "directives",
      label: t("mcpGroupDirectives"),
      tools: ["list_directives", "create_directive", "delete_directive"],
    },
    {
      key: "memories",
      label: t("mcpGroupMemories"),
      tools: ["list_memories", "get_memory", "update_memory", "invalidate_memory"],
    },
    {
      key: "documents",
      label: t("mcpGroupDocuments"),
      tools: ["list_documents", "get_document", "delete_document"],
    },
    {
      key: "operations",
      label: t("mcpGroupOperations"),
      tools: ["list_operations", "get_operation", "cancel_operation"],
    },
    { key: "tags", label: t("mcpGroupTags"), tools: ["list_tags"] },
    {
      key: "knowledgeBase",
      label: t("mcpGroupKnowledgeBase"),
      tools: [
        "get_knowledge_base_tree",
        "search_knowledge_base",
        "get_knowledge_page",
        "create_knowledge_folder",
        "create_knowledge_page",
        "update_knowledge_node",
        "delete_knowledge_node",
      ],
    },
  ];
}

const MCP_ALL_TOOLS: string[] = [
  "retain",
  "sync_retain",
  "recall",
  "reflect",
  "list_banks",
  "create_bank",
  "get_bank",
  "get_bank_stats",
  "update_bank",
  "delete_bank",
  "clear_memories",
  "list_mental_models",
  "get_mental_model",
  "create_mental_model",
  "update_mental_model",
  "delete_mental_model",
  "refresh_mental_model",
  "clear_mental_model",
  "list_directives",
  "create_directive",
  "delete_directive",
  "list_memories",
  "get_memory",
  "update_memory",
  "invalidate_memory",
  "list_documents",
  "get_document",
  "delete_document",
  "list_operations",
  "get_operation",
  "cancel_operation",
  "list_tags",
  "get_knowledge_base_tree",
  "search_knowledge_base",
  "get_knowledge_page",
  "create_knowledge_folder",
  "create_knowledge_page",
  "update_knowledge_node",
  "delete_knowledge_node",
];
const ALL_TOOLS: string[] = MCP_ALL_TOOLS;

// ─── Slice helpers ────────────────────────────────────────────────────────────

function parseEntityLabels(raw: unknown): LabelGroup[] | null {
  if (Array.isArray(raw)) return raw as LabelGroup[];
  if (raw && typeof raw === "object" && Array.isArray((raw as any).attributes))
    return (raw as any).attributes as LabelGroup[];
  return null;
}

function retainSlice(config: Record<string, any>): RetainEdits {
  return {
    retain_context_chars: config.retain_context_chars ?? null,
    retain_chunk_size: config.retain_chunk_size ?? null,
    retain_structured_chunk_size: config.retain_structured_chunk_size ?? null,
    retain_extraction_mode: config.retain_extraction_mode ?? null,
    retain_mission: config.retain_mission ?? null,
    retain_custom_instructions: config.retain_custom_instructions ?? null,
    entities_allow_free_form: config.entities_allow_free_form ?? null,
    entity_labels: parseEntityLabels(config.entity_labels),
  };
}

function strategiesSlice(config: Record<string, any>): StrategiesEdits {
  return {
    retain_default_strategy: config.retain_default_strategy ?? null,
    retain_strategies: config.retain_strategies ?? null,
  };
}

function mcpSlice(config: Record<string, any>): MCPEdits {
  return {
    mcp_enabled_tools: config.mcp_enabled_tools ?? null,
  };
}

function geminiSlice(config: Record<string, any>): GeminiEdits {
  return {
    llm_gemini_safety_settings: config.llm_gemini_safety_settings ?? null,
  };
}

// Reads the bank's OVERRIDES, not the resolved config: the resolved value can
// not distinguish "inherited true" from "explicitly set to true", and the UI
// needs that distinction to offer "Server Default".
function auditSlice(overrides: Record<string, any>): AuditEdits {
  return {
    audit_log_enabled: overrides.audit_log_enabled ?? null,
  };
}

function docStorageSlice(overrides: Record<string, any>): DocStorageEdits {
  return {
    store_document_text: overrides.store_document_text ?? null,
  };
}

function mentalModelsSlice(overrides: Record<string, any>): MentalModelsEdits {
  return {
    mental_model_min_refresh_interval_seconds:
      overrides.mental_model_min_refresh_interval_seconds ?? null,
  };
}

/** The bank's reflect defaults as chips, so the row reads without opening the dialog. */
function ReflectOptionsSummary({ options }: { options: ReflectOptionsEdits }) {
  const t = useTranslations("bankConfig");
  const chips = [
    options.reflect_search_observations_max_tokens != null
      ? t("reflectObservationsMaxTokensChip", {
          tokens: options.reflect_search_observations_max_tokens,
        })
      : t("reflectObservationsMaxTokensChipDefault"),
    options.reflect_search_observations_include_entities === false
      ? t("reflectObservationsEntitiesOff")
      : t("reflectObservationsEntitiesOn"),
  ];
  return (
    <div className="flex flex-wrap gap-1.5">
      {chips.map((chip) => (
        <span
          key={chip}
          className="rounded-full border border-border/60 bg-background px-2.5 py-0.5 text-xs text-foreground"
        >
          {chip}
        </span>
      ))}
    </div>
  );
}

function reflectOptionsSlice(overrides: Record<string, any>): ReflectOptionsEdits {
  const opts = overrides.reflect_default_options ?? {};
  return {
    reflect_search_observations_max_tokens: opts.reflect_search_observations_max_tokens ?? null,
    reflect_search_observations_include_entities:
      opts.reflect_search_observations_include_entities ?? null,
  };
}

function recallSlice(overrides: Record<string, any>): RecallEdits {
  return {
    enable_text_search: overrides.enable_text_search ?? null,
    enable_temporal_retrieval: overrides.enable_temporal_retrieval ?? null,
    enable_graph_retrieval: overrides.enable_graph_retrieval ?? null,
    enable_reranking: overrides.enable_reranking ?? null,
  };
}

const DEFAULT_PROFILE: ProfileData = {
  reflect_mission: "",
  disposition_skepticism: 3,
  disposition_literalism: 3,
  disposition_empathy: 3,
};

// ─── BankConfigView ───────────────────────────────────────────────────────────

export function BankConfigView() {
  const t = useTranslations("bankConfig");
  const tMentalModels = useTranslations("mentalModels");
  const { currentBank: bankId } = useBank();
  const { features } = useFeatures();
  const bankConfigEnabled = features?.bank_config_api ?? true; // optimistic default while loading
  const [loading, setLoading] = useState(true);

  // Source of truth
  const [baseConfig, setBaseConfig] = useState<Record<string, any>>({});
  // Explicit per-bank overrides only (server defaults excluded), so sections
  // can tell "inherited" apart from "explicitly set to the same value".
  const [baseOverrides, setBaseOverrides] = useState<Record<string, any>>({});
  const [baseProfile, setBaseProfile] = useState<ProfileData>(DEFAULT_PROFILE);

  // Per-section local edits
  const [retainEdits, setRetainEdits] = useState<RetainEdits>(retainSlice({}));
  const [strategiesEdits, setStrategiesEdits] = useState<StrategiesEdits>(strategiesSlice({}));
  const [observationsEdits, setObservationsEdits] = useState<ObservationsEdits>(
    observationsSlice({}, {})
  );
  const [reflectEdits, setReflectEdits] = useState<ProfileData>(DEFAULT_PROFILE);
  // Reflect's default options are edited in their own dialog and saved from
  // there, apart from the section's Save — same as the knowledge-page trigger.
  const [reflectOptionsOpen, setReflectOptionsOpen] = useState(false);
  const [reflectOptionsForm, setReflectOptionsForm] = useState<ReflectOptionsEdits>(
    reflectOptionsSlice({})
  );
  const [reflectOptionsSaving, setReflectOptionsSaving] = useState(false);
  const [reflectOptionsError, setReflectOptionsError] = useState<string | null>(null);
  const [mcpEdits, setMcpEdits] = useState<MCPEdits>(mcpSlice({}));
  const [geminiEdits, setGeminiEdits] = useState<GeminiEdits>(geminiSlice({}));
  const [auditEdits, setAuditEdits] = useState<AuditEdits>(auditSlice({}));
  const [docStorageEdits, setDocStorageEdits] = useState<DocStorageEdits>(docStorageSlice({}));
  const [recallEdits, setRecallEdits] = useState<RecallEdits>(recallSlice({}));
  const [mentalModelsEdits, setMentalModelsEdits] = useState<MentalModelsEdits>(
    mentalModelsSlice({})
  );
  // The knowledge-page default trigger is edited in its own dialog and saved
  // from there, apart from the section's Save.
  const [pageTriggerOpen, setPageTriggerOpen] = useState(false);
  const [pageTriggerForm, setPageTriggerForm] = useState<TriggerForm>(() =>
    effectivePageTrigger(null)
  );
  const [pageTriggerSaving, setPageTriggerSaving] = useState(false);
  const [pageTriggerError, setPageTriggerError] = useState<string | null>(null);

  // Per-section saving/error state
  const [retainSaving, setRetainSaving] = useState(false);
  const [observationsSaving, setObservationsSaving] = useState(false);
  const [reflectSaving, setReflectSaving] = useState(false);
  const [mcpSaving, setMcpSaving] = useState(false);
  const [geminiSaving, setGeminiSaving] = useState(false);
  const [securityPrivacySaving, setSecurityPrivacySaving] = useState(false);
  const [recallSaving, setRecallSaving] = useState(false);
  const [mentalModelsSaving, setMentalModelsSaving] = useState(false);
  const [retainError, setRetainError] = useState<string | null>(null);
  const [observationsError, setObservationsError] = useState<string | null>(null);
  const [reflectError, setReflectError] = useState<string | null>(null);
  const [mcpError, setMcpError] = useState<string | null>(null);
  const [geminiError, setGeminiError] = useState<string | null>(null);
  const [securityPrivacyError, setSecurityPrivacyError] = useState<string | null>(null);
  const [recallError, setRecallError] = useState<string | null>(null);
  const [mentalModelsError, setMentalModelsError] = useState<string | null>(null);

  // Dirty tracking
  const retainDirty = useMemo(
    () =>
      JSON.stringify(retainEdits) !== JSON.stringify(retainSlice(baseConfig)) ||
      JSON.stringify(strategiesEdits) !== JSON.stringify(strategiesSlice(baseConfig)),
    [retainEdits, strategiesEdits, baseConfig]
  );
  const observationsDirty = useMemo(
    () =>
      JSON.stringify(observationsEdits) !==
      JSON.stringify(observationsSlice(baseConfig, baseOverrides)),
    [observationsEdits, baseConfig, baseOverrides]
  );
  const reflectDirty = useMemo(
    () => JSON.stringify(reflectEdits) !== JSON.stringify(baseProfile),
    [reflectEdits, baseProfile]
  );
  const mcpDirty = useMemo(
    () => JSON.stringify(mcpEdits) !== JSON.stringify(mcpSlice(baseConfig)),
    [mcpEdits, baseConfig]
  );
  const geminiDirty = useMemo(
    () => JSON.stringify(geminiEdits) !== JSON.stringify(geminiSlice(baseConfig)),
    [geminiEdits, baseConfig]
  );
  const auditDirty = useMemo(
    () => JSON.stringify(auditEdits) !== JSON.stringify(auditSlice(baseOverrides)),
    [auditEdits, baseOverrides]
  );
  const docStorageDirty = useMemo(
    () => JSON.stringify(docStorageEdits) !== JSON.stringify(docStorageSlice(baseOverrides)),
    [docStorageEdits, baseOverrides]
  );
  const recallDirty = useMemo(
    () => JSON.stringify(recallEdits) !== JSON.stringify(recallSlice(baseOverrides)),
    [recallEdits, baseOverrides]
  );
  const mentalModelsDirty = useMemo(
    () => JSON.stringify(mentalModelsEdits) !== JSON.stringify(mentalModelsSlice(baseOverrides)),
    [mentalModelsEdits, baseOverrides]
  );
  useEffect(() => {
    if (bankId) loadAll();
  }, [bankId]);

  const loadAll = async () => {
    if (!bankId) return;
    setLoading(true);
    try {
      const configResp = await client.getBankConfig(bankId);
      const cfg = configResp.config;
      const overrides = configResp.overrides ?? {};
      // Disposition and the reflect mission are ordinary config keys — the separate
      // profile read they used to be merged with no longer exists.
      const prof: ProfileData = {
        reflect_mission: cfg.reflect_mission ?? "",
        disposition_skepticism: cfg.disposition_skepticism ?? 3,
        disposition_literalism: cfg.disposition_literalism ?? 3,
        disposition_empathy: cfg.disposition_empathy ?? 3,
      };
      setBaseConfig(cfg);
      setBaseOverrides(overrides);
      setBaseProfile(prof);
      setRetainEdits(retainSlice(cfg));
      setStrategiesEdits(strategiesSlice(cfg));
      setObservationsEdits(observationsSlice(cfg, overrides));
      setReflectEdits(prof);
      setReflectOptionsForm(reflectOptionsSlice(cfg));
      setMcpEdits(mcpSlice(cfg));
      setGeminiEdits(geminiSlice(cfg));
      setAuditEdits(auditSlice(overrides));
      setDocStorageEdits(docStorageSlice(overrides));
      setRecallEdits(recallSlice(overrides));
      setMentalModelsEdits(mentalModelsSlice(overrides));
    } catch (err) {
      console.error("Failed to load bank data:", err);
    } finally {
      setLoading(false);
    }
  };

  const saveRetain = async () => {
    if (!bankId) return;
    setRetainSaving(true);
    setRetainError(null);
    try {
      const payload = { ...retainEdits, ...strategiesEdits };
      await client.updateBankConfig(bankId, payload);
      setBaseConfig((prev) => ({ ...prev, ...payload }));
    } catch (err: any) {
      setRetainError(err.message || t("retainFailedToSave"));
    } finally {
      setRetainSaving(false);
    }
  };

  const saveObservations = async () => {
    if (!bankId) return;
    setObservationsSaving(true);
    setObservationsError(null);
    const submittedEdits = observationsEdits;
    try {
      const response = await client.updateBankConfig(bankId, submittedEdits);
      // Overrides are a complete bank-only snapshot. Resolved config may omit
      // permission-filtered fields, so merge it against the accepted payload.
      const overrides = response.overrides ?? {};
      setBaseConfig((prev) => mergeResolvedObservations(prev, submittedEdits, response.config));
      setBaseOverrides((prev) => mergeObservationsOverrides(prev, overrides));
      setObservationsEdits((current) =>
        reconcileObservationsEdits(current, submittedEdits, response.config, overrides)
      );
    } catch (err: any) {
      setObservationsError(err.message || t("observationsFailedToSave"));
    } finally {
      setObservationsSaving(false);
    }
  };

  const saveReflect = async () => {
    if (!bankId) return;
    setReflectSaving(true);
    setReflectError(null);
    try {
      await client.updateBankConfig(bankId, {
        reflect_mission: reflectEdits.reflect_mission || null,
        disposition_skepticism: reflectEdits.disposition_skepticism,
        disposition_literalism: reflectEdits.disposition_literalism,
        disposition_empathy: reflectEdits.disposition_empathy,
      });
      setBaseProfile(reflectEdits);
    } catch (err: any) {
      setReflectError(err.message || t("reflectFailedToSave"));
    } finally {
      setReflectSaving(false);
    }
  };

  const openReflectOptions = () => {
    setReflectOptionsError(null);
    setReflectOptionsForm(reflectOptionsSlice(baseConfig));
    setReflectOptionsOpen(true);
  };

  const saveReflectOptions = async (reset: boolean) => {
    if (!bankId) return;
    setReflectOptionsSaving(true);
    setReflectOptionsError(null);
    try {
      // Every field left unset means "no bank default at all": send null so the
      // override is cleared rather than stored as an empty object.
      const options = Object.fromEntries(
        Object.entries(reflectOptionsForm).filter(([, v]) => v !== null)
      );
      const reflect_default_options = reset || Object.keys(options).length === 0 ? null : options;
      await client.updateBankConfig(bankId, { reflect_default_options });
      setBaseConfig((prev) => ({ ...prev, reflect_default_options }));
      setBaseOverrides((prev) => ({ ...prev, reflect_default_options }));
      setReflectOptionsForm(reflectOptionsSlice({ reflect_default_options }));
      setReflectOptionsOpen(false);
    } catch (err: any) {
      setReflectOptionsError(err.message || t("reflectFailedToSave"));
    } finally {
      setReflectOptionsSaving(false);
    }
  };

  const saveMCP = async () => {
    if (!bankId) return;
    setMcpSaving(true);
    setMcpError(null);
    try {
      await client.updateBankConfig(bankId, mcpEdits);
      setBaseConfig((prev) => ({ ...prev, ...mcpEdits }));
    } catch (err: any) {
      setMcpError(err.message || t("mcpFailedToSave"));
    } finally {
      setMcpSaving(false);
    }
  };

  const saveGemini = async () => {
    if (!bankId) return;
    setGeminiSaving(true);
    setGeminiError(null);
    try {
      await client.updateBankConfig(bankId, geminiEdits);
      setBaseConfig((prev) => ({ ...prev, ...geminiEdits }));
    } catch (err: any) {
      setGeminiError(err.message || t("geminiFailedToSave"));
    } finally {
      setGeminiSaving(false);
    }
  };

  const saveSecurityPrivacy = async () => {
    if (!bankId) return;
    setSecurityPrivacySaving(true);
    setSecurityPrivacyError(null);
    try {
      // A null on either key clears that override server-side (JSON null is the
      // "Server Default" tombstone); mirror both into local override state.
      await client.updateBankConfig(bankId, { ...auditEdits, ...docStorageEdits });
      setBaseOverrides((prev) => {
        const next = { ...prev };
        if (auditEdits.audit_log_enabled === null) delete next.audit_log_enabled;
        else next.audit_log_enabled = auditEdits.audit_log_enabled;
        if (docStorageEdits.store_document_text === null) delete next.store_document_text;
        else next.store_document_text = docStorageEdits.store_document_text;
        return next;
      });
    } catch (err: any) {
      setSecurityPrivacyError(err.message || t("securityPrivacyFailedToSave"));
    } finally {
      setSecurityPrivacySaving(false);
    }
  };

  const saveRecall = async () => {
    if (!bankId) return;
    setRecallSaving(true);
    setRecallError(null);
    try {
      // Same tombstone convention as the sections above: a null clears the bank
      // override server-side, so the stage falls back to the server default.
      await client.updateBankConfig(bankId, { ...recallEdits });
      setBaseOverrides((prev) => {
        const next = { ...prev };
        for (const key of [
          "enable_text_search",
          "enable_temporal_retrieval",
          "enable_graph_retrieval",
          "enable_reranking",
        ] as const) {
          if (recallEdits[key] === null) delete next[key];
          else next[key] = recallEdits[key];
        }
        return next;
      });
    } catch (err: any) {
      setRecallError(err.message || t("recallFailedToSave"));
    } finally {
      setRecallSaving(false);
    }
  };

  const saveMentalModels = async () => {
    if (!bankId) return;
    setMentalModelsSaving(true);
    setMentalModelsError(null);
    try {
      // Same tombstone convention as the sections above: a null clears the bank
      // override server-side, so the setting falls back to the server default.
      await client.updateBankConfig(bankId, { ...mentalModelsEdits });
      setBaseOverrides((prev) => {
        const next = { ...prev };
        const value = mentalModelsEdits.mental_model_min_refresh_interval_seconds;
        if (value === null) delete next.mental_model_min_refresh_interval_seconds;
        else next.mental_model_min_refresh_interval_seconds = value;
        return next;
      });
    } catch (err: any) {
      setMentalModelsError(err.message || t("mentalModelsFailedToSave"));
    } finally {
      setMentalModelsSaving(false);
    }
  };

  const openPageTrigger = () => {
    setPageTriggerForm(effectivePageTrigger(baseConfig.knowledge_page_default_trigger));
    setPageTriggerError(null);
    setPageTriggerOpen(true);
  };

  // null clears the bank override; the value inherited from the tenant/server
  // isn't known client-side, so a reset reloads the resolved config.
  const savePageTrigger = async (reset: boolean) => {
    if (!bankId) return;
    const trigger = reset ? null : triggerFromForm(pageTriggerForm);
    if (!reset && !trigger) {
      setPageTriggerError(tMentalModels("invalidTagGroupsJson"));
      return;
    }
    setPageTriggerSaving(true);
    setPageTriggerError(null);
    try {
      await client.updateBankConfig(bankId, { knowledge_page_default_trigger: trigger });
      if (reset) {
        await loadAll();
      } else {
        setBaseConfig((prev) => ({ ...prev, knowledge_page_default_trigger: trigger }));
        setBaseOverrides((prev) => ({ ...prev, knowledge_page_default_trigger: trigger }));
      }
      setPageTriggerOpen(false);
    } catch (err: any) {
      setPageTriggerError(err.message || t("mentalModelsFailedToSave"));
    } finally {
      setPageTriggerSaving(false);
    }
  };

  if (!bankId) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-muted-foreground">{t("noBankSelected")}</p>
      </div>
    );
  }

  if (!bankConfigEnabled) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
        <p className="text-base font-medium text-foreground">{t("apiDisabledTitle")}</p>
        <p className="text-sm text-muted-foreground max-w-sm">
          {t.rich("apiDisabledDescription", {
            code: (chunks) => (
              <code className="font-mono text-xs bg-muted px-1 py-0.5 rounded">{chunks}</code>
            ),
          })}
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spinner size="lg" variant="jump" />
      </div>
    );
  }

  return (
    <>
      <div className="space-y-8">
        {/* Retain + Strategies Section */}
        <ConfigSection
          title={t("retainTitle")}
          description={t("retainSectionDescription")}
          error={retainError}
          dirty={retainDirty}
          saving={retainSaving}
          onSave={saveRetain}
          action={
            // At section level, not inside the strategy form: the tester renders the
            // bank's resolved retain config and picks its own strategy, so it is not
            // a property of whichever strategy tab happens to be open.
            <PreviewPromptButton
              operation="retain"
              onSaved={(field, value) =>
                setRetainEdits((prev) => ({ ...prev, [field]: value }) as RetainEdits)
              }
            />
          }
        >
          <FieldRow label={t("defaultStrategyLabel")} description={t("defaultStrategyDescription")}>
            <Select
              value={strategiesEdits.retain_default_strategy ?? "__none__"}
              onValueChange={(v) =>
                setStrategiesEdits((prev) => ({
                  ...prev,
                  retain_default_strategy: v === "__none__" ? null : v,
                }))
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">
                  <span className="text-muted-foreground italic">{t("default")}</span>
                </SelectItem>
                {Object.keys(strategiesEdits.retain_strategies ?? {}).map((name) => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldRow>
          <RetainStrategiesPanel
            defaultValues={retainEdits}
            onDefaultChange={(patch) => setRetainEdits((prev) => ({ ...prev, ...patch }))}
            strategies={strategiesEdits.retain_strategies}
            onStrategiesChange={(v) =>
              setStrategiesEdits((prev) => ({ ...prev, retain_strategies: v }))
            }
          />
        </ConfigSection>

        {/* Observations Section */}
        <ConfigSection
          title={t("observationsTitle")}
          description={t("observationsDescription")}
          error={observationsError}
          dirty={observationsDirty}
          saving={observationsSaving}
          onSave={saveObservations}
        >
          <FieldRow
            label={t("enableObservationsLabel")}
            description={t("enableObservationsDescription")}
          >
            <Select
              value={
                observationsEdits.enable_observations === null
                  ? INHERIT_SENTINEL
                  : String(observationsEdits.enable_observations)
              }
              onValueChange={(v) =>
                setObservationsEdits((prev) => ({
                  ...prev,
                  enable_observations: v === INHERIT_SENTINEL ? null : v === "true",
                }))
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={INHERIT_SENTINEL}>
                  {/* The resolved value reveals the parent default only when
                      there is no bank override. */}
                  {(baseOverrides.enable_observations === undefined ||
                    baseOverrides.enable_observations === null) &&
                  typeof baseConfig.enable_observations === "boolean"
                    ? t("auditServerDefault", {
                        state: baseConfig.enable_observations ? t("enabled") : t("disabled"),
                      })
                    : t("serverDefault")}
                </SelectItem>
                <SelectItem value="true">{t("enabled")}</SelectItem>
                <SelectItem value="false">{t("disabled")}</SelectItem>
              </SelectContent>
            </Select>
          </FieldRow>
          <FieldRow label={t("llmBatchSizeLabel")} description={t("llmBatchSizeDescription")}>
            <Input
              type="number"
              min={1}
              max={64}
              value={observationsEdits.consolidation_llm_batch_size ?? ""}
              onChange={(e) =>
                setObservationsEdits((prev) => ({
                  ...prev,
                  consolidation_llm_batch_size: e.target.value
                    ? parseInt(e.target.value, 10)
                    : null,
                }))
              }
              placeholder={t("serverDefault")}
            />
          </FieldRow>
          <ConsolidationStrategiesPanel
            defaults={observationsEdits}
            onDefaultChange={(patch) => setObservationsEdits((prev) => ({ ...prev, ...patch }))}
            strategies={observationsEdits.consolidation_strategies}
            onStrategiesChange={(next) =>
              setObservationsEdits((prev) => ({ ...prev, consolidation_strategies: next }))
            }
            defaultAction={
              <PreviewPromptButton
                operation="consolidation"
                onSaved={(field, value) =>
                  setObservationsEdits((prev) => ({ ...prev, [field]: value }) as ObservationsEdits)
                }
              />
            }
          />
        </ConfigSection>

        {/* Reflect Section */}
        <ConfigSection
          title={t("reflectTitle")}
          description={t("reflectDescription")}
          error={reflectError}
          dirty={reflectDirty}
          saving={reflectSaving}
          onSave={saveReflect}
        >
          <TextareaRow
            label={t("missionLabel")}
            description={t("reflectMissionDescription")}
            value={reflectEdits.reflect_mission}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, reflect_mission: v }))}
            placeholder={t("reflectMissionPlaceholder")}
            rows={3}
            action={
              <PreviewPromptButton
                operation="reflect"
                onSaved={(field, value) =>
                  setReflectEdits((prev) => ({ ...prev, [field]: value ?? "" }) as ProfileData)
                }
              />
            }
          />
          <div className="px-6 py-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
            <div className="min-w-0 space-y-2">
              <div>
                <p className="text-sm font-medium">{t("reflectDefaultOptionsLabel")}</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {t("reflectDefaultOptionsDescription")}
                </p>
              </div>
              <ReflectOptionsSummary options={reflectOptionsSlice(baseConfig)} />
            </div>
            <Button variant="outline" size="sm" className="shrink-0" onClick={openReflectOptions}>
              {t("knowledgePageDefaultTriggerEdit")}
            </Button>
          </div>
          <Dialog
            open={reflectOptionsOpen}
            onOpenChange={(o) => !o && setReflectOptionsOpen(false)}
          >
            <DialogContent className="sm:max-w-2xl">
              <DialogHeader>
                <DialogTitle>{t("reflectDefaultOptionsDialogTitle")}</DialogTitle>
                <DialogDescription>{t("reflectDefaultOptionsDialogHint")}</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-2">
                <div className="space-y-2">
                  <div>
                    <p className="text-sm font-medium">{t("reflectObservationsMaxTokensLabel")}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {t("reflectObservationsMaxTokensDescription")}
                    </p>
                  </div>
                  <Input
                    type="number"
                    min={1}
                    value={reflectOptionsForm.reflect_search_observations_max_tokens ?? ""}
                    onChange={(e) =>
                      setReflectOptionsForm((prev) => ({
                        ...prev,
                        reflect_search_observations_max_tokens: e.target.value
                          ? parseInt(e.target.value, 10)
                          : null,
                      }))
                    }
                    placeholder={t("serverDefault")}
                  />
                </div>
                <div className="space-y-2">
                  <div>
                    <p className="text-sm font-medium">{t("reflectObservationsEntitiesLabel")}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {t("reflectObservationsEntitiesDescription")}
                    </p>
                  </div>
                  <Select
                    value={
                      reflectOptionsForm.reflect_search_observations_include_entities === null
                        ? "default"
                        : String(reflectOptionsForm.reflect_search_observations_include_entities)
                    }
                    onValueChange={(v) =>
                      setReflectOptionsForm((prev) => ({
                        ...prev,
                        reflect_search_observations_include_entities:
                          v === "default" ? null : v === "true",
                      }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">{t("serverDefault")}</SelectItem>
                      <SelectItem value="true">{t("reflectObservationsEntitiesOn")}</SelectItem>
                      <SelectItem value="false">{t("reflectObservationsEntitiesOff")}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              {reflectOptionsError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{reflectOptionsError}</AlertDescription>
                </Alert>
              )}
              <DialogFooter className="sm:justify-between">
                <div>
                  {baseOverrides.reflect_default_options && (
                    <Button
                      variant="ghost"
                      onClick={() => saveReflectOptions(true)}
                      disabled={reflectOptionsSaving}
                    >
                      {t("resetToInherited")}
                    </Button>
                  )}
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setReflectOptionsOpen(false)}
                    disabled={reflectOptionsSaving}
                  >
                    {tMentalModels("cancelButton")}
                  </Button>
                  <Button onClick={() => saveReflectOptions(false)} disabled={reflectOptionsSaving}>
                    {reflectOptionsSaving ? (
                      <>
                        <Spinner size="sm" className="mr-2" />
                        {t("saving")}
                      </>
                    ) : (
                      t("saveChanges")
                    )}
                  </Button>
                </div>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          <TraitRow
            label={t("skepticismLabel")}
            description={t("skepticismDescription")}
            lowLabel={t("skepticismLowLabel")}
            highLabel={t("skepticismHighLabel")}
            value={reflectEdits.disposition_skepticism}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, disposition_skepticism: v }))}
          />
          <TraitRow
            label={t("literalismLabel")}
            description={t("literalismDescription")}
            lowLabel={t("literalismLowLabel")}
            highLabel={t("literalismHighLabel")}
            value={reflectEdits.disposition_literalism}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, disposition_literalism: v }))}
          />
          <TraitRow
            label={t("empathyLabel")}
            description={t("empathyDescription")}
            lowLabel={t("empathyLowLabel")}
            highLabel={t("empathyHighLabel")}
            value={reflectEdits.disposition_empathy}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, disposition_empathy: v }))}
          />
        </ConfigSection>

        {/* Mental Models & Knowledge Pages Section */}
        <ConfigSection
          title={t("mentalModelsTitle")}
          description={t("mentalModelsDescription")}
          error={mentalModelsError}
          dirty={mentalModelsDirty}
          saving={mentalModelsSaving}
          onSave={saveMentalModels}
        >
          <FieldRow
            label={t("mentalModelMinRefreshIntervalLabel")}
            description={t("mentalModelMinRefreshIntervalDescription")}
          >
            <Input
              type="number"
              min={0}
              value={mentalModelsEdits.mental_model_min_refresh_interval_seconds ?? ""}
              onChange={(e) =>
                setMentalModelsEdits((prev) => ({
                  ...prev,
                  mental_model_min_refresh_interval_seconds: e.target.value
                    ? parseInt(e.target.value, 10)
                    : null,
                }))
              }
              placeholder={t("serverDefault")}
            />
          </FieldRow>
          <div className="px-6 py-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
            <div className="min-w-0 space-y-2">
              <div>
                <p className="text-sm font-medium">{t("knowledgePageDefaultTriggerLabel")}</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {t("knowledgePageDefaultTriggerDescription")}
                </p>
              </div>
              <TriggerSummary
                form={effectivePageTrigger(baseConfig.knowledge_page_default_trigger)}
              />
            </div>
            <Button variant="outline" size="sm" className="shrink-0" onClick={openPageTrigger}>
              {t("knowledgePageDefaultTriggerEdit")}
            </Button>
          </div>
          <Dialog open={pageTriggerOpen} onOpenChange={(o) => !o && setPageTriggerOpen(false)}>
            <DialogContent className="sm:max-w-4xl max-h-[90vh] flex flex-col">
              <DialogHeader>
                <DialogTitle>{t("knowledgePageDefaultTriggerDialogTitle")}</DialogTitle>
                <DialogDescription>{t("knowledgePageDefaultTriggerDialogHint")}</DialogDescription>
              </DialogHeader>
              <div className="flex-1 overflow-y-auto px-1.5 py-2">
                <MentalModelTriggerFields value={pageTriggerForm} onChange={setPageTriggerForm} />
              </div>
              {pageTriggerError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{pageTriggerError}</AlertDescription>
                </Alert>
              )}
              <DialogFooter className="sm:justify-between">
                <div>
                  {baseOverrides.knowledge_page_default_trigger && (
                    <Button
                      variant="ghost"
                      onClick={() => savePageTrigger(true)}
                      disabled={pageTriggerSaving}
                    >
                      {t("resetToInherited")}
                    </Button>
                  )}
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setPageTriggerOpen(false)}
                    disabled={pageTriggerSaving}
                  >
                    {tMentalModels("cancelButton")}
                  </Button>
                  <Button onClick={() => savePageTrigger(false)} disabled={pageTriggerSaving}>
                    {pageTriggerSaving ? (
                      <>
                        <Spinner size="sm" className="mr-2" />
                        {t("saving")}
                      </>
                    ) : (
                      t("saveChanges")
                    )}
                  </Button>
                </div>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </ConfigSection>

        {/* MCP Tools Section */}
        <ConfigSection
          title={t("clientsTitle")}
          description={t("clientsDescription")}
          error={mcpError}
          dirty={mcpDirty}
          saving={mcpSaving}
          onSave={saveMCP}
        >
          <BankAliasRows bankId={bankId} />
          <FieldRow label={t("restrictToolsLabel")} description={t("restrictToolsDescription")}>
            <div className="flex items-center gap-2 justify-end">
              <Switch
                checked={mcpEdits.mcp_enabled_tools !== null}
                onCheckedChange={(restricted) =>
                  setMcpEdits({
                    mcp_enabled_tools: restricted ? [...ALL_TOOLS] : null,
                  })
                }
              />
              <Label className="text-xs text-muted-foreground">
                {mcpEdits.mcp_enabled_tools !== null ? t("enabled") : t("disabled")}
              </Label>
            </div>
          </FieldRow>
          {mcpEdits.mcp_enabled_tools !== null && (
            <ToolSelector
              selected={mcpEdits.mcp_enabled_tools}
              onChange={(tools) => setMcpEdits({ mcp_enabled_tools: tools })}
            />
          )}
        </ConfigSection>

        {/* Security & Privacy Section — audit logging + document-text storage */}
        <ConfigSection
          title={t("securityPrivacyTitle")}
          description={t("securityPrivacyDescription")}
          error={securityPrivacyError}
          dirty={auditDirty || docStorageDirty}
          saving={securityPrivacySaving}
          onSave={saveSecurityPrivacy}
        >
          <FieldRow label={t("auditEnabledLabel")} description={t("auditEnabledDescription")}>
            {/* Tri-state rather than a Switch: the bank may inherit the server
                default, or override it in either direction. A Switch cannot
                express "inherit", and would silently write an explicit value. */}
            <Select
              value={
                auditEdits.audit_log_enabled === null
                  ? INHERIT_SENTINEL
                  : String(auditEdits.audit_log_enabled)
              }
              onValueChange={(v) =>
                setAuditEdits({
                  audit_log_enabled: v === INHERIT_SENTINEL ? null : v === "true",
                })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {/* Sentinel, not "": Radix rejects an empty SelectItem value. */}
                <SelectItem value={INHERIT_SENTINEL}>
                  {t("auditServerDefault", {
                    state: features?.audit_log ? t("enabled") : t("disabled"),
                  })}
                </SelectItem>
                <SelectItem value="true">{t("enabled")}</SelectItem>
                <SelectItem value="false">{t("disabled")}</SelectItem>
              </SelectContent>
            </Select>
          </FieldRow>
          <FieldRow
            label={t("docStorageEnabledLabel")}
            description={t("docStorageEnabledDescription")}
          >
            {/* Tri-state: inherit the server default, or override per bank. */}
            <Select
              value={
                docStorageEdits.store_document_text === null
                  ? INHERIT_SENTINEL
                  : String(docStorageEdits.store_document_text)
              }
              onValueChange={(v) =>
                setDocStorageEdits({
                  store_document_text: v === INHERIT_SENTINEL ? null : v === "true",
                })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={INHERIT_SENTINEL}>
                  {t("docStorageServerDefault", {
                    state: features?.store_document_text ? t("enabled") : t("disabled"),
                  })}
                </SelectItem>
                <SelectItem value="true">{t("enabled")}</SelectItem>
                <SelectItem value="false">{t("disabled")}</SelectItem>
              </SelectContent>
            </Select>
          </FieldRow>
        </ConfigSection>

        {/* Recall Section — per-bank retrieval pipeline stages */}
        <ConfigSection
          title={t("recallTitle")}
          description={t("recallDescription")}
          error={recallError}
          dirty={recallDirty}
          saving={recallSaving}
          onSave={saveRecall}
        >
          {(
            [
              ["enable_text_search", "recallTextSearch"],
              ["enable_temporal_retrieval", "recallTemporalRetrieval"],
              ["enable_graph_retrieval", "recallGraphRetrieval"],
              ["enable_reranking", "recallReranking"],
            ] as const
          ).map(([field, key]) => (
            <FieldRow key={field} label={t(`${key}Label`)} description={t(`${key}Description`)}>
              {/* Tri-state: inherit the server default, or override per bank. */}
              <Select
                value={recallEdits[field] === null ? INHERIT_SENTINEL : String(recallEdits[field])}
                onValueChange={(v) =>
                  setRecallEdits({
                    ...recallEdits,
                    [field]: v === INHERIT_SENTINEL ? null : v === "true",
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={INHERIT_SENTINEL}>{t("recallServerDefault")}</SelectItem>
                  <SelectItem value="true">{t("enabled")}</SelectItem>
                  <SelectItem value="false">{t("disabled")}</SelectItem>
                </SelectContent>
              </Select>
            </FieldRow>
          ))}
        </ConfigSection>

        {/* Models Section */}
        <ConfigSection
          title={t("modelsTitle")}
          description={t("modelsDescription")}
          error={geminiError}
          dirty={geminiDirty}
          saving={geminiSaving}
          onSave={saveGemini}
        >
          {/* Gemini subsection */}
          <div className="px-6 py-4 space-y-4">
            <p className="text-sm font-semibold">{t("geminiSubsectionTitle")}</p>
            <div className="pl-4 border-l-2 border-border/40 space-y-4">
              <FieldRow
                label={t("safetySettingsLabel")}
                description={
                  <>
                    {t("safetySettingsDescriptionPart1")}{" "}
                    <a
                      href="https://ai.google.dev/gemini-api/docs/safety-settings"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline hover:text-foreground transition-colors"
                    >
                      {t("safetySettingsLearnMore")}
                    </a>
                  </>
                }
              >
                <div className="flex items-center gap-2 justify-end">
                  <Switch
                    checked={geminiEdits.llm_gemini_safety_settings !== null}
                    onCheckedChange={(enabled) =>
                      setGeminiEdits({
                        llm_gemini_safety_settings: enabled
                          ? [...DEFAULT_GEMINI_SAFETY_SETTINGS]
                          : null,
                      })
                    }
                  />
                  <Label className="text-xs text-muted-foreground">
                    {geminiEdits.llm_gemini_safety_settings !== null ? t("custom") : t("default")}
                  </Label>
                </div>
              </FieldRow>
              {geminiEdits.llm_gemini_safety_settings !== null && (
                <GeminiSafetyEditor
                  value={geminiEdits.llm_gemini_safety_settings}
                  onChange={(settings) => setGeminiEdits({ llm_gemini_safety_settings: settings })}
                />
              )}
            </div>
          </div>
        </ConfigSection>
      </div>
    </>
  );
}

// ─── Retain strategies panel ──────────────────────────────────────────────────

type RetainFormValues = RetainStrategyValues<LabelGroup[]>;

// Value/label pairs rather than bare values: the option list is user-facing, so
// the labels are translated while the values stay the API's mode strings.
function getExtractionModes(t: (key: string) => string): { value: string; label: string }[] {
  return [
    { value: "concise", label: t("extractionModeConcise") },
    { value: "verbose", label: t("extractionModeVerbose") },
    { value: "verbatim", label: t("extractionModeVerbatim") },
    { value: "chunks", label: t("extractionModeChunks") },
    { value: "custom", label: t("extractionModeCustom") },
  ];
}
const INHERIT_SENTINEL = "__inherit__";

function RetainStrategyForm({
  values,
  onChange,
  isOverride = false,
}: {
  values: RetainFormValues;
  onChange: (patch: Partial<RetainFormValues>) => void;
  isOverride?: boolean;
}) {
  const t = useTranslations("bankConfig");
  const modeValue = values.retain_extraction_mode ?? (isOverride ? INHERIT_SENTINEL : "");
  const showCustomField = values.retain_extraction_mode === "custom";

  return (
    <div className="divide-y divide-border/40">
      <FieldRow label={t("extractionModeLabel")} description={t("extractionModeDescription")}>
        <Select
          value={modeValue}
          onValueChange={(val) =>
            onChange({ retain_extraction_mode: val === INHERIT_SENTINEL ? null : val || null })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={isOverride ? t("inherited") : undefined} />
          </SelectTrigger>
          <SelectContent>
            {isOverride && (
              <SelectItem value={INHERIT_SENTINEL}>
                <span className="text-muted-foreground italic">{t("inherited")}</span>
              </SelectItem>
            )}
            {getExtractionModes(t).map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </FieldRow>
      <FieldRow label={t("sourceContextLabel")} description={t("sourceContextDescription")}>
        <Input
          aria-label={t("sourceContextLabel")}
          type="number"
          min={0}
          max={32000}
          step={1}
          value={values.retain_context_chars ?? ""}
          onChange={(e) =>
            onChange({ retain_context_chars: e.target.value ? parseInt(e.target.value, 10) : null })
          }
          placeholder={isOverride ? t("inherited") : undefined}
        />
      </FieldRow>
      <FieldRow label={t("chunkSizeLabel")} description={t("chunkSizeDescription")}>
        <Input
          type="number"
          min={500}
          max={8000}
          step={1}
          value={values.retain_chunk_size ?? ""}
          onChange={(e) =>
            onChange({ retain_chunk_size: e.target.value ? parseInt(e.target.value, 10) : null })
          }
          placeholder={isOverride ? t("inherited") : undefined}
        />
      </FieldRow>
      <FieldRow
        label={t("structuredChunkSizeLabel")}
        description={t("structuredChunkSizeDescription")}
      >
        <Input
          type="number"
          min={1}
          step={1}
          value={values.retain_structured_chunk_size ?? ""}
          onChange={(e) => {
            onChange({
              retain_structured_chunk_size: e.target.value ? parseInt(e.target.value, 10) : null,
            });
          }}
          placeholder={isOverride ? t("inherited") : undefined}
        />
      </FieldRow>
      <TextareaRow
        label={t("missionLabel")}
        description={t("retainMissionDescription")}
        value={values.retain_mission ?? ""}
        onChange={(v) => onChange({ retain_mission: v || null })}
        placeholder={isOverride ? t("inherited") : t("retainMissionPlaceholder")}
        rows={3}
      />
      {showCustomField && (
        <TextareaRow
          label={t("customExtractionPromptLabel")}
          description={t("customExtractionPromptDescription")}
          value={values.retain_custom_instructions ?? ""}
          onChange={(v) => onChange({ retain_custom_instructions: v || null })}
          rows={5}
        />
      )}
      <FieldRow label={t("freeFormEntitiesLabel")} description={t("freeFormEntitiesDescription")}>
        <div className="flex justify-end items-center gap-2">
          <Label className="text-sm text-muted-foreground cursor-pointer select-none">
            {(values.entities_allow_free_form ?? true) ? t("enabled") : t("disabled")}
          </Label>
          <Switch
            checked={values.entities_allow_free_form ?? true}
            onCheckedChange={(v) => onChange({ entities_allow_free_form: v })}
          />
        </div>
      </FieldRow>
      <EntityLabelsEditor
        value={values.entity_labels ?? []}
        onChange={(attrs) => onChange({ entity_labels: attrs.length > 0 ? attrs : null })}
      />
    </div>
  );
}

type LocalStrategy = RetainStrategy<LabelGroup[]>;

function fromStrategiesDict(dict: Record<string, Record<string, any>> | null): LocalStrategy[] {
  return deserializeRetainStrategies(dict, parseEntityLabels);
}

function toStrategiesDict(local: LocalStrategy[]): Record<string, Record<string, any>> | null {
  return serializeRetainStrategies(local);
}

function RetainStrategiesPanel({
  defaultValues,
  onDefaultChange,
  strategies,
  onStrategiesChange,
}: {
  defaultValues: RetainFormValues;
  onDefaultChange: (patch: Partial<RetainFormValues>) => void;
  strategies: Record<string, Record<string, any>> | null;
  onStrategiesChange: (v: Record<string, Record<string, any>> | null) => void;
}) {
  const t = useTranslations("bankConfig");
  const tCommon = useTranslations("common");
  const [local, setLocal] = useState<LocalStrategy[]>(() => fromStrategiesDict(strategies));
  const [selectedTab, setSelectedTab] = useState<number | "default">("default");
  const [pendingDelete, setPendingDelete] = useState<LocalStrategy | null>(null);
  const skipSyncRef = useRef(false);

  const strategiesKey = JSON.stringify(strategies);
  useEffect(() => {
    if (skipSyncRef.current) {
      skipSyncRef.current = false;
      return;
    }
    setLocal(fromStrategiesDict(strategies));
  }, [strategiesKey]);

  const updateLocal = (next: LocalStrategy[]) => {
    skipSyncRef.current = true;
    setLocal(next);
    onStrategiesChange(toStrategiesDict(next));
  };

  const addStrategy = () => {
    const id = Date.now();
    const next = [
      ...local,
      {
        id,
        name: "",
        values: {
          retain_extraction_mode: null,
          retain_context_chars: null,
          retain_chunk_size: null,
          retain_structured_chunk_size: null,
          retain_mission: null,
          retain_custom_instructions: null,
          entities_allow_free_form: null,
          entity_labels: null,
        },
      },
    ];
    updateLocal(next);
    setSelectedTab(id);
  };

  const removeStrategy = (id: number) => {
    const next = local.filter((s) => s.id !== id);
    updateLocal(next);
    if (selectedTab === id) setSelectedTab("default");
  };

  const updateStrategy = (id: number, patch: Partial<LocalStrategy>) => {
    updateLocal(local.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  };

  const activeStrategy = selectedTab !== "default" ? local.find((s) => s.id === selectedTab) : null;

  return (
    <div>
      {/* Tab bar */}
      <div className="border-b border-border px-6 flex items-stretch gap-1 flex-wrap">
        {/* Default tab */}
        {/* Active tab is marked with the brand gradient, which can't be
            expressed as a border colour — so the active underline is an
            absolutely-positioned bar and the border-b-2 is kept transparent
            purely to carry the subtle hover underline on inactive tabs. */}
        <button
          type="button"
          onClick={() => setSelectedTab("default")}
          className={`relative py-3 px-4 text-sm font-semibold transition-colors border-b-2 border-transparent -mb-px ${
            selectedTab === "default"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground hover:border-border"
          }`}
        >
          {t("default")}
          {selectedTab === "default" && (
            <div className="absolute bottom-[-2px] left-0 right-0 h-0.5 bg-primary-gradient" />
          )}
        </button>

        {/* Named strategy tabs */}
        {local.map((s) => (
          <div
            key={s.id}
            className={`relative flex items-center gap-2 py-3 px-4 text-sm font-semibold transition-colors border-b-2 border-transparent -mb-px cursor-pointer ${
              selectedTab === s.id
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground hover:border-border"
            }`}
            onClick={() => setSelectedTab(s.id)}
          >
            {selectedTab === s.id && (
              <div className="absolute bottom-[-2px] left-0 right-0 h-0.5 bg-primary-gradient" />
            )}
            <span className="font-mono">
              {s.name || <span className="italic font-normal opacity-50">{t("unnamed")}</span>}
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setPendingDelete(s);
              }}
              className="opacity-40 hover:opacity-100 hover:text-destructive transition-opacity text-base leading-none"
            >
              ×
            </button>
          </div>
        ))}

        <button
          type="button"
          onClick={addStrategy}
          className="py-3 px-3 text-sm text-muted-foreground hover:text-primary transition-colors flex items-center gap-1.5"
        >
          <Plus className="h-3.5 w-3.5" />
          {t("addStrategy")}
        </button>
      </div>

      {/* Form */}
      <div>
        {selectedTab === "default" ? (
          <RetainStrategyForm values={defaultValues} onChange={onDefaultChange} />
        ) : activeStrategy ? (
          <div>
            <div className="px-6 py-3 flex items-center gap-3 border-b border-border/40">
              <label className="text-xs text-muted-foreground shrink-0">{t("nameLabel")}</label>
              <div className="flex flex-col gap-1">
                <Input
                  value={activeStrategy.name}
                  onChange={(e) => updateStrategy(activeStrategy.id, { name: e.target.value })}
                  placeholder={t("strategyNamePlaceholder")}
                  className={`h-7 text-xs font-mono max-w-[200px] ${!activeStrategy.name.trim() ? "border-destructive focus-visible:ring-destructive" : ""}`}
                />
                {!activeStrategy.name.trim() && (
                  <p className="text-xs text-destructive">{t("nameIsRequired")}</p>
                )}
              </div>
            </div>
            <RetainStrategyForm
              values={activeStrategy.values}
              onChange={(patch) =>
                updateStrategy(activeStrategy.id, {
                  values: { ...activeStrategy.values, ...patch },
                })
              }
              isOverride
            />
          </div>
        ) : null}
      </div>

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("deleteStrategyTitle", { name: pendingDelete?.name || t("unnamed") })}
            </AlertDialogTitle>
            <AlertDialogDescription>{t("deleteStrategyDescription")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{tCommon("cancel")}</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                if (pendingDelete) {
                  removeStrategy(pendingDelete.id);
                  setPendingDelete(null);
                }
              }}
            >
              {tCommon("delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// ─── ToolSelector ─────────────────────────────────────────────────────────────

function ToolSelector({
  selected,
  onChange,
}: {
  selected: string[];
  onChange: (tools: string[]) => void;
}) {
  const t = useTranslations("bankConfig");
  const mcpToolGroups = getMcpToolGroups(t);
  const selectedSet = new Set(selected);

  const toggleTool = (tool: string) => {
    const next = new Set(selectedSet);
    if (next.has(tool)) {
      next.delete(tool);
    } else {
      next.add(tool);
    }
    onChange(ALL_TOOLS.filter((tool) => next.has(tool)));
  };

  const allSelected = ALL_TOOLS.every((tool) => selectedSet.has(tool));
  const noneSelected = selected.length === 0;

  const toggleAll = () => {
    onChange(allSelected ? [] : [...ALL_TOOLS]);
  };

  return (
    <div className="px-6 py-4 space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {t("toolsEnabled", { selected: selected.length, total: ALL_TOOLS.length })}
        </p>
        <button type="button" onClick={toggleAll} className="text-xs text-primary hover:underline">
          {allSelected ? t("deselectAll") : t("selectAll")}
        </button>
      </div>
      <div className="space-y-4">
        {mcpToolGroups.map((group) => {
          const groupSelected = group.tools.filter((tool) => selectedSet.has(tool)).length;
          const groupAll = groupSelected === group.tools.length;
          return (
            <div key={group.key}>
              <div className="flex items-center justify-between mb-1.5">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  {group.label}
                </p>
                <button
                  type="button"
                  onClick={() => {
                    const next = new Set(selectedSet);
                    if (groupAll) {
                      group.tools.forEach((tool) => next.delete(tool));
                    } else {
                      group.tools.forEach((tool) => next.add(tool));
                    }
                    onChange(ALL_TOOLS.filter((tool) => next.has(tool)));
                  }}
                  className="text-xs text-primary hover:underline"
                >
                  {groupAll ? t("deselect") : t("selectAll")}
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {group.tools.map((tool) => {
                  const active = selectedSet.has(tool);
                  return (
                    <button
                      key={tool}
                      type="button"
                      onClick={() => toggleTool(tool)}
                      className={`px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
                        active
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-muted/30 text-muted-foreground border-border/40 hover:border-primary/40"
                      }`}
                    >
                      {tool}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
      {noneSelected && <p className="text-xs text-destructive">{t("noToolsWarning")}</p>}
    </div>
  );
}

// ─── ConfigSection ────────────────────────────────────────────────────────────

function ConfigSection({
  title,
  description,
  children,
  error,
  dirty,
  saving,
  onSave,
  action,
}: {
  title: string;
  description: string;
  children: ReactNode;
  error: string | null;
  dirty: boolean;
  saving: boolean;
  /** Omit for a section whose controls apply immediately — it then has no Save footer. */
  onSave?: () => void;
  /** Rendered opposite the heading — used by Retain for the prompt tester. */
  action?: ReactNode;
}) {
  const t = useTranslations("bankConfig");
  return (
    <section className="space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        {action}
      </div>
      <Card className="bg-muted/20 border-border/40">
        <div className="divide-y divide-border/40">{children}</div>
        {error && (
          <div className="px-6 pb-2 pt-2">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        )}
        {onSave && (
          <div className="px-6 py-4 flex justify-end border-t border-border/40">
            <Button size="sm" disabled={!dirty || saving} onClick={onSave}>
              {saving ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  {t("saving")}
                </>
              ) : (
                t("saveChanges")
              )}
            </Button>
          </div>
        )}
      </Card>
    </section>
  );
}

// ─── BankAliasRows (the ids that reach this bank) ────────────────────────────

/**
 * The bank's aliases — extra ids that reach it, beside its own.
 *
 * Rows rather than a section of its own: it lives inside Access, next to the MCP
 * tool list, because both answer "how do clients get at this bank" — one is which
 * ids reach it, the other is what they may call once they do.
 *
 * Unlike its neighbours these rows are NOT part of the section's form: each add
 * and remove is its own request, applied immediately, so the section's Save
 * button neither covers nor waits for them. Its own errors therefore render here
 * instead of in the section's error slot.
 */
/** Sentinel for "no alias is shown" — Radix Select cannot hold an empty value,
 *  and the bank's own id is deliberately not one of the alias options. */
const OWN_ID = "__own_id__";

function BankAliasRows({ bankId }: { bankId: string | null }) {
  const t = useTranslations("bankAliases");
  const [aliases, setAliases] = useState<BankAliasEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Removal is the one destructive action here: the id stops routing the moment
  // it commits, so anything still calling it starts failing.
  const [pendingRemove, setPendingRemove] = useState<string | null>(null);

  useEffect(() => {
    if (!bankId) return;
    client
      .listBankAliases(bankId)
      .then((d) => setAliases(d.aliases ?? []))
      .catch((e) => {
        console.error("Failed to load bank aliases:", e);
        setError(t("loadFailed"));
      });
  }, [bankId, t]);

  const add = async () => {
    const alias = draft.trim();
    if (!alias || !bankId) return;
    setBusy(true);
    setError(null);
    try {
      // The response carries the whole list, so the chips show the server's view
      // rather than a locally appended guess.
      setAliases((await client.createBankAlias(bankId, alias)).aliases ?? []);
      setDraft("");
    } catch (e) {
      // Usually the name is already taken (409); that message names it.
      setError(e instanceof Error ? e.message : t("addFailed"));
    } finally {
      setBusy(false);
    }
  };

  /** Show the bank as `choice`, or as its own id when `choice` is OWN_ID. */
  const promote = async (choice: string) => {
    if (!bankId) return;
    setError(null);
    const current = aliases.find((a) => a.primary)?.alias ?? null;
    try {
      // The whole list comes back either way, so the demoted and promoted rows
      // update in the same render — the UI never shows two as shown.
      const next =
        choice === OWN_ID
          ? current && (await client.setBankAliasPrimary(bankId, current, false))
          : await client.setBankAliasPrimary(bankId, choice, true);
      if (next) setAliases(next.aliases ?? []);
    } catch (e) {
      console.error("Failed to set primary bank alias:", e);
      setError(t("primaryFailed"));
    }
  };

  const remove = async (alias: string) => {
    if (!bankId) return;
    setError(null);
    try {
      setAliases((await client.deleteBankAlias(bankId, alias)).aliases ?? []);
    } catch (e) {
      console.error("Failed to remove bank alias:", e);
      setError(t("removeFailed"));
    } finally {
      setPendingRemove(null);
    }
  };

  if (!bankId) return null;

  return (
    <>
      <FieldRow
        label={t("title")}
        description={t.rich("description", {
          bankId,
          // Italic, not the code style used for the aliases themselves: this one
          // names the bank you are already looking at, rather than an id to type.
          name: (chunks) => <em>{chunks}</em>,
        })}
      >
        <div className="flex gap-2">
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
            placeholder={t("placeholder")}
            className="h-8 text-sm"
            disabled={busy}
          />
          <Button size="sm" variant="outline" onClick={add} disabled={busy || !draft.trim()}>
            {t("add")}
          </Button>
        </div>
      </FieldRow>
      {aliases.length > 0 && (
        <FieldRow label={t("shownAsLabel")} description={t("shownAsDescription", { bankId })}>
          <Select
            value={aliases.find((a) => a.primary)?.alias ?? OWN_ID}
            onValueChange={(v) => promote(v)}
          >
            <SelectTrigger className="w-full h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {/* Always offered, and the default: a bank need not be shown as an
                  alias, and this is how you put it back to its own id. */}
              <SelectItem value={OWN_ID}>{bankId}</SelectItem>
              {aliases.map((entry) => (
                <SelectItem key={entry.alias} value={entry.alias}>
                  {entry.alias}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FieldRow>
      )}
      {(aliases.length > 0 || error) && (
        <div className="px-6 py-3 flex flex-wrap items-center gap-1.5">
          {aliases.map((entry) => (
            <IdChip
              key={entry.alias}
              id={entry.alias}
              size="xs"
              // Marks the one the bank is shown as. Promotion lives in the "Shown
              // as" row rather than on the chip: the chip shell renders a button
              // instead of the ✕ when given an onClick, so a clickable chip would
              // quietly lose its remove control.
              active={entry.primary}
              title={entry.primary ? t("primaryTitle") : undefined}
              onRemove={() => setPendingRemove(entry.alias)}
              removeLabel={t("removeAria", { alias: entry.alias })}
            />
          ))}
          {error && <p className="text-xs text-destructive">{error}</p>}
        </div>
      )}

      <AlertDialog
        open={pendingRemove !== null}
        onOpenChange={(open) => !open && setPendingRemove(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("removeTitle", { alias: pendingRemove ?? "" })}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("removeConfirm", { alias: pendingRemove ?? "", bankId })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("cancel")}</AlertDialogCancel>
            <AlertDialogAction onClick={() => pendingRemove && remove(pendingRemove)}>
              {t("remove")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

// ─── FieldRow (2-column layout for number / select / boolean) ─────────────────

function FieldRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="px-6 py-4">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="flex-1">
          <p className="text-sm font-medium">{label}</p>
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
        </div>
        <div className="md:w-64 shrink-0">{children}</div>
      </div>
    </div>
  );
}

// ─── TextareaRow (stacked layout) ─────────────────────────────────────────────

function TextareaRow({
  label,
  description,
  value,
  onChange,
  placeholder,
  rows,
  action,
}: {
  label: string;
  description?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
  /** Rendered opposite the label — used by the mission fields for "Preview prompt". */
  action?: React.ReactNode;
}) {
  return (
    <div className="px-6 py-4">
      <div className="space-y-2">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium">{label}</p>
            {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
          </div>
          {action}
        </div>
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={rows ?? 3}
          className="font-mono text-sm"
        />
      </div>
    </div>
  );
}

// ─── TraitRow (stacked layout with 1–5 selector) ──────────────────────────────

function TraitRow({
  label,
  description,
  lowLabel,
  highLabel,
  value,
  onChange,
}: {
  label: string;
  description?: string;
  lowLabel?: string;
  highLabel?: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="px-6 py-4">
      <div className="space-y-3">
        <div>
          <p className="text-sm font-medium">{label}</p>
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
        </div>
        <div className="flex items-center gap-1.5">
          {lowLabel && (
            <span className="text-xs text-muted-foreground w-16 text-right shrink-0">
              {lowLabel}
            </span>
          )}
          <div className="flex gap-0.5">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => onChange(n)}
                className={`w-4 h-4 rounded-full transition-colors hover:opacity-80 ${
                  n <= value ? "bg-primary" : "bg-muted"
                }`}
              />
            ))}
          </div>
          {highLabel && (
            <span className="text-xs text-muted-foreground w-20 shrink-0">{highLabel}</span>
          )}
          <span className="text-xs font-mono text-muted-foreground ml-1 shrink-0">{value}/5</span>
        </div>
      </div>
    </div>
  );
}

// ─── GeminiSafetyEditor ───────────────────────────────────────────────────────

// ─── ScopesEditor ─────────────────────────────────────────────────────────────

/** A strategy's rules: which observation scopes it applies to.
 *
 * Laid out as sentence-style filter rules (the pattern email and issue-tracker
 * filters use): each rule reads "Scopes that have [all of these tags ▾]" followed
 * by the tags, rules are separated by a literal "or", and each shows a one-line
 * summary of the existing scopes it matches, expandable to examples.
 *
 * History: this started as a textarea ("one scope per line, commas between
 * tags"), which hid the AND/OR logic entirely. A second version drew boxes but
 * stacked a label, a two-button toggle and a sentence restating the toggle in
 * every box, plus a permanent row of match chips — correct, but too dense to read.
 *
 * The match summaries come from the server (`rulePreviews`, see the panel), not
 * from matching in the browser: an earlier version matched a client-side copy of
 * the scope list with a TypeScript port of fnmatch, which silently capped at the
 * first 1000 scopes and was a second implementation to keep in sync.
 *
 * Tags are picked with the app's standard `TagFilterInput`, as in the document
 * and mental-model filters. Empty rules are kept while being filled in; the
 * server ignores them.
 */
function ScopesEditor({
  value,
  onChange,
  rulePreviews,
  complete,
  selfIndex,
  labelFor,
}: {
  value: ScopePattern[];
  onChange: (scopes: ScopePattern[]) => void;
  /** Server preview per rule, aligned by index; undefined while loading. */
  rulePreviews: StrategyRulePreview[] | undefined;
  complete: boolean;
  selfIndex: number;
  labelFor: (index: number) => string;
}) {
  const t = useTranslations("bankConfig");
  const { currentBank } = useBank();

  // The bank's real tags (same search the tag filters use) plus a `key:*`
  // wildcard per tag key — the pattern a strategy almost always wants.
  const fetchSuggestions = async (q: string): Promise<string[]> => {
    const bankTags = currentBank
      ? (await client.listTags(currentBank, q ? `${q}*` : undefined, 20)).items.map((i) => i.tag)
      : [];
    return suggestedTags([bankTags])
      .filter((tag) => tag.startsWith(q))
      .slice(0, 20);
  };

  const setPattern = (index: number, patch: Partial<ScopePattern>) =>
    onChange(value.map((pattern, i) => (i === index ? { ...pattern, ...patch } : pattern)));

  // A pasted "a, b" arrives as one tag; split it so it becomes two chips.
  const setTags = (index: number, tags: string[]) =>
    setPattern(index, {
      tags: [
        ...new Set(
          tags
            .flatMap((tag) => tag.split(","))
            .map((tag) => tag.trim())
            .filter(Boolean)
        ),
      ],
    });

  return (
    <div className="space-y-3">
      {value.map((pattern, index) => {
        const tags = pattern.tags;
        const mode = pattern.tags_match ?? "all";
        const preview = rulePreviews?.[index];
        return (
          <div key={index}>
            {index > 0 && (
              <div className="flex items-center gap-3 pb-3">
                <div className="h-px flex-1 bg-border" />
                <span className="rounded-full border border-border px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                  {t("consolidationStrategiesOr")}
                </span>
                <div className="h-px flex-1 bg-border" />
              </div>
            )}
            <div className="rounded-lg border border-border bg-card p-4 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span>{t("consolidationStrategiesRuleSentence")}</span>
                  <Select
                    value={mode}
                    onValueChange={(next) =>
                      setPattern(index, { tags_match: next as StrategyTagsMatch })
                    }
                  >
                    <SelectTrigger className="h-8 w-auto gap-1.5 text-sm font-medium">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">{t("consolidationStrategiesModeAll")}</SelectItem>
                      <SelectItem value="exact">{t("consolidationStrategiesModeExact")}</SelectItem>
                    </SelectContent>
                  </Select>
                  <span className="text-xs text-muted-foreground">
                    {mode === "exact"
                      ? t("consolidationStrategiesModeExactNote")
                      : t("consolidationStrategiesModeAllNote")}
                  </span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 shrink-0 p-0 text-muted-foreground hover:text-destructive"
                  aria-label={t("consolidationStrategiesRemoveRule")}
                  title={t("consolidationStrategiesRemoveRule")}
                  onClick={() => onChange(value.filter((_, i) => i !== index))}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>

              <TagFilterInput
                value={tags}
                onChange={(next) => setTags(index, next)}
                fetchSuggestions={fetchSuggestions}
                placeholder={t("consolidationStrategiesAddTagPlaceholder")}
                inline
              />

              {tags.length > 0 && (
                <MatchSummary
                  matchCount={preview?.match_count}
                  takenCount={preview?.taken_count ?? 0}
                  samples={preview?.samples ?? []}
                  complete={complete}
                  selfIndex={selfIndex}
                  labelFor={labelFor}
                />
              )}
            </div>
          </div>
        );
      })}
      <Button variant="outline" size="sm" onClick={() => onChange([...value, { tags: [] }])}>
        <Plus className="h-4 w-4" />
        {t("consolidationStrategiesAddRule")}
      </Button>
    </div>
  );
}

// ─── MatchSummary ─────────────────────────────────────────────────────────────

/** "Matches 4 existing scopes · 1 handled by an earlier strategy — Show".
 *
 * A summary first, examples on demand: a permanent row of chips (an earlier
 * version) was the densest part of the editor. Counts and examples come from the
 * server preview; `complete: false` means the bank has more scopes than the
 * preview scans, so the count is shown as "at least". With `selfIndex` set,
 * examples an earlier strategy wins are struck through with the winner named —
 * strategies are never combined, so this rule has no effect on them.
 */
function MatchSummary({
  matchCount,
  takenCount = 0,
  samples,
  complete,
  selfIndex,
  labelFor,
  emptyText,
}: {
  /** undefined while the preview is loading. */
  matchCount: number | undefined;
  takenCount?: number;
  samples: StrategyScopePreview[];
  complete: boolean;
  selfIndex?: number;
  labelFor?: (index: number) => string;
  emptyText?: string;
}) {
  const t = useTranslations("bankConfig");
  const [open, setOpen] = useState(false);
  const globalLabel = t("consolidationStrategiesGlobalScope");

  if (matchCount === undefined) {
    return <p className="text-xs text-muted-foreground">{t("consolidationStrategiesChecking")}</p>;
  }
  if (matchCount === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        {emptyText ?? t("consolidationStrategiesMatchesNone")}
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        <span className="font-medium">
          {complete
            ? t("consolidationStrategiesMatchesCount", { count: matchCount })
            : t("consolidationStrategiesMatchesCountAtLeast", { count: matchCount })}
        </span>
        {takenCount > 0 && (
          <span className="text-amber-700 dark:text-amber-400">
            · {t("consolidationStrategiesMatchesTaken", { count: takenCount })}
          </span>
        )}
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-primary hover:underline"
          aria-expanded={open}
        >
          {open ? t("consolidationStrategiesHide") : t("consolidationStrategiesShow")}
        </button>
      </div>
      {open && (
        <ul className="space-y-1">
          {samples.map((sample) => {
            const owned = selfIndex === undefined || sample.handled_by === selfIndex;
            const winnerName =
              sample.handled_by !== null && labelFor ? labelFor(sample.handled_by) : t("default");
            return (
              <li
                key={sample.tags.join("\u0000") || "__global__"}
                className="flex flex-wrap items-center gap-2 text-xs"
              >
                <span
                  className={`rounded-md border border-border px-1.5 py-0.5 font-mono ${
                    owned ? "bg-muted/40" : "text-muted-foreground line-through"
                  }`}
                >
                  {scopeChipLabel(sample.tags, globalLabel)}
                </span>
                <span className="text-muted-foreground">
                  {t("consolidationStrategiesObservationCount", { count: sample.count })}
                </span>
                {!owned && (
                  <span className="text-amber-700 dark:text-amber-400">
                    {t("consolidationStrategiesHandledBy", { name: winnerName })}
                  </span>
                )}
              </li>
            );
          })}
          {matchCount > samples.length && (
            <li className="text-xs text-muted-foreground">
              {t("consolidationStrategiesMoreScopes", { count: matchCount - samples.length })}
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

// ─── ConsolidationStrategiesPanel ─────────────────────────────────────────────

function scopeChipLabel(tags: string[], globalLabel: string): string {
  return tags.length ? tags.join(" + ") : globalLabel;
}

/** Per-scope consolidation settings, laid out like the retain strategies.
 *
 * The "Default" tab is the bank-level mission, cap and source-facts limits: they
 * apply to every scope no strategy claims, and fill in whatever a strategy leaves
 * empty. Each other tab is one `consolidation_strategies` entry, labelled by the
 * scopes it claims (the scopes are its name — there is no separate one).
 *
 * Each rule shows the bank's *existing* scopes it covers, so the effect of a glob
 * is visible before saving. That comes from the server's preview endpoint, asked
 * again 400ms after the last edit. Order matters on the server (first claiming
 * strategy wins, whole), so tabs read left to right in that order.
 *
 * Unlike the retain panel there is no local copy of the list: strategies have no
 * name to keep unique, so the stored array is the state and tabs are addressed by
 * position.
 */
function ConsolidationStrategiesPanel({
  defaults,
  onDefaultChange,
  strategies,
  onStrategiesChange,
  defaultAction,
}: {
  defaults: ConsolidationSettings;
  onDefaultChange: (patch: Partial<ConsolidationSettings>) => void;
  strategies: ConsolidationStrategy[] | null;
  onStrategiesChange: (next: ConsolidationStrategy[] | null) => void;
  /** Rendered next to the Default mission — the "Preview prompt" button. */
  defaultAction?: ReactNode;
}) {
  const t = useTranslations("bankConfig");
  const tCommon = useTranslations("common");
  const { currentBank } = useBank();
  const list = strategies ?? [];

  // Server preview of which existing scopes each rule matches, for the list as
  // currently edited (unsaved). Debounced so typing a tag does not send a request
  // per keystroke; a response for an older draft is dropped.
  const [preview, setPreview] = useState<ConsolidationStrategiesPreview | null>(null);
  const draftKey = JSON.stringify(list);
  useEffect(() => {
    if (!currentBank) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      client
        .previewConsolidationStrategies(currentBank, JSON.parse(draftKey))
        .then((result) => {
          if (!cancelled) setPreview(result);
        })
        .catch(() => {
          if (!cancelled) setPreview(null);
        });
    }, 400);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [currentBank, draftKey]);
  const [selected, setSelected] = useState<number | "default">("default");
  const [pendingDelete, setPendingDelete] = useState<number | null>(null);
  // A save or reset can shorten the list under an open tab.
  const active = selected === "default" ? null : (list[selected] ?? null);
  const selectedTab = active ? selected : "default";

  const store = (next: ConsolidationStrategy[]) =>
    onStrategiesChange(next.length ? next.map(compactStrategy) : null);

  const add = () => {
    store([...list, { scopes: [{ tags: [] }] }]);
    setSelected(list.length);
  };

  const update = (index: number, patch: Partial<ConsolidationStrategy>) =>
    store(list.map((s, i) => (i === index ? { ...s, ...patch } : s)));

  const remove = (index: number) => {
    store(list.filter((_, i) => i !== index));
    setSelected("default");
  };

  const globalLabel = t("consolidationStrategiesGlobalScope");
  const andWord = t("consolidationStrategiesAnd");
  const orWord = t("consolidationStrategiesOr");
  // Referenced with its priority number: two strategies can share a label (both
  // "company:*"), and "uses company:*" would not say which of them wins.
  const labelFor = (index: number) =>
    `#${index + 1} ${
      scopesLabel(list[index]?.scopes ?? [], andWord, orWord) ||
      t("consolidationStrategiesNoScopeTab")
    }`;

  // Tab order is priority order (first claiming strategy wins), so it must be
  // changeable without deleting and re-creating strategies.
  const move = (from: number, to: number) => {
    const next = [...list];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    store(next);
    setSelected(to);
  };

  return (
    <div>
      <div className="px-6 pt-6 pb-4 space-y-2">
        <p className="text-sm font-semibold">{t("consolidationStrategiesTitle")}</p>
        <p className="text-sm text-muted-foreground">{t("consolidationStrategiesLead")}</p>
        {/* The rules, one click away instead of a paragraph above the tabs. */}
        <details className="group">
          <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-primary hover:underline [&::-webkit-details-marker]:hidden">
            <ChevronRight className="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
            {t("consolidationStrategiesHowTitle")}
          </summary>
          <ul className="mt-2 ml-5 list-disc space-y-1.5 text-xs text-muted-foreground">
            <li>{t("consolidationStrategiesHowScopes")}</li>
            <li>{t("consolidationStrategiesHowRules")}</li>
            <li>{t("consolidationStrategiesHowPriority")}</li>
            <li>{t("consolidationStrategiesHowDefault")}</li>
          </ul>
        </details>
      </div>

      {/* Tab bar — same look as RetainStrategiesPanel. */}
      <div className="border-b border-border px-6 flex items-stretch gap-1 flex-wrap">
        <button
          type="button"
          onClick={() => setSelected("default")}
          className={`relative py-3 px-4 text-sm font-semibold transition-colors border-b-2 border-transparent -mb-px ${
            selectedTab === "default"
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground hover:border-border"
          }`}
        >
          {t("default")}
          {selectedTab === "default" && (
            <div className="absolute bottom-[-2px] left-0 right-0 h-0.5 bg-primary-gradient" />
          )}
        </button>

        {list.map((strategy, index) => {
          const label = scopesLabel(strategy.scopes, andWord, orWord);
          return (
            <div
              key={index}
              className={`relative flex items-center gap-2 py-3 px-4 text-sm font-semibold transition-colors border-b-2 border-transparent -mb-px cursor-pointer ${
                selectedTab === index
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground hover:border-border"
              }`}
              onClick={() => setSelected(index)}
            >
              {selectedTab === index && (
                <div className="absolute bottom-[-2px] left-0 right-0 h-0.5 bg-primary-gradient" />
              )}
              <span className="text-xs font-normal text-muted-foreground tabular-nums">
                {index + 1}
              </span>
              <span className="font-mono max-w-[220px] truncate" title={label}>
                {label || (
                  <span className="italic font-normal opacity-50">
                    {t("consolidationStrategiesNoScopeTab")}
                  </span>
                )}
              </span>
              <button
                type="button"
                aria-label={t("consolidationStrategiesRemove", { name: label || "—" })}
                onClick={(e) => {
                  e.stopPropagation();
                  setPendingDelete(index);
                }}
                className="opacity-40 hover:opacity-100 hover:text-destructive transition-opacity text-base leading-none"
              >
                ×
              </button>
            </div>
          );
        })}

        <button
          type="button"
          onClick={add}
          className="py-3 px-3 text-sm text-muted-foreground hover:text-primary transition-colors flex items-center gap-1.5"
        >
          <Plus className="h-3.5 w-3.5" />
          {t("addStrategy")}
        </button>
      </div>

      {active === null ? (
        <div>
          <div className="px-6 py-5 space-y-2 border-b border-border/40">
            <SectionHeading
              title={t("consolidationStrategiesDefaultTitle")}
              description={t("consolidationStrategiesDefaultLead")}
            />
            <MatchSummary
              matchCount={preview?.default.match_count}
              samples={preview?.default.samples ?? []}
              complete={preview?.complete ?? true}
              emptyText={t("consolidationStrategiesDefaultCoversNone")}
            />
          </div>
          <ConsolidationSettingsForm
            values={defaults}
            onChange={onDefaultChange}
            missionAction={defaultAction}
          />
        </div>
      ) : (
        <div>
          <div className="px-6 py-5 space-y-4 border-b border-border/40">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <SectionHeading
                title={`1 · ${t("consolidationStrategiesWhichScopes")}`}
                description={t("consolidationStrategiesWhichScopesLead")}
              />
              {/* Priority is tab order (first matching strategy wins). */}
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <span className="mr-1">
                  {t("consolidationStrategiesPriority", {
                    position: (selectedTab as number) + 1,
                    total: list.length,
                  })}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  disabled={selectedTab === 0}
                  aria-label={t("consolidationStrategiesMoveEarlier")}
                  title={t("consolidationStrategiesMoveEarlier")}
                  onClick={() => move(selectedTab as number, (selectedTab as number) - 1)}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  disabled={selectedTab === list.length - 1}
                  aria-label={t("consolidationStrategiesMoveLater")}
                  title={t("consolidationStrategiesMoveLater")}
                  onClick={() => move(selectedTab as number, (selectedTab as number) + 1)}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <ScopesEditor
              value={active.scopes}
              onChange={(scopes) => update(selectedTab as number, { scopes })}
              rulePreviews={preview?.strategies[selectedTab as number]?.rules}
              complete={preview?.complete ?? true}
              selfIndex={selectedTab as number}
              labelFor={labelFor}
            />
            {!active.scopes.some((pattern) => pattern.tags.length > 0) && (
              <p className="text-xs text-amber-700 dark:text-amber-400">
                {t("consolidationStrategiesNoScopes")}
              </p>
            )}
          </div>
          <div className="px-6 pt-5">
            <SectionHeading
              title={`2 · ${t("consolidationStrategiesSettingsTitle")}`}
              description={t("consolidationStrategiesSettingsLead")}
            />
            {!strategyOverridesSomething(active) && (
              <p className="mt-2 text-xs text-amber-700 dark:text-amber-400">
                {t("consolidationStrategiesOverridesNothing")}
              </p>
            )}
          </div>
          <ConsolidationSettingsForm
            values={active}
            onChange={(patch) => update(selectedTab as number, patch)}
            inherited={defaults}
          />
        </div>
      )}

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("deleteStrategyTitle", {
                name:
                  (pendingDelete !== null &&
                    scopesLabel(list[pendingDelete]?.scopes ?? [], andWord, orWord)) ||
                  t("consolidationStrategiesNoScopeTab"),
              })}
            </AlertDialogTitle>
            <AlertDialogDescription>{t("deleteStrategyDescription")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{tCommon("cancel")}</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                if (pendingDelete !== null) {
                  remove(pendingDelete);
                  setPendingDelete(null);
                }
              }}
            >
              {tCommon("delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function SectionHeading({ title, description }: { title: string; description?: string }) {
  return (
    <div className="space-y-0.5">
      <p className="text-sm font-semibold">{title}</p>
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
    </div>
  );
}

// ─── ConsolidationSettingsForm ────────────────────────────────────────────────

/** The four per-scope settings, for the Default tab or a strategy tab.
 *
 * With `inherited` set (a strategy tab), an empty field means "use Default", and
 * its placeholder shows the value that will actually apply — so nobody has to
 * flip back to the Default tab to know what a blank field does.
 */
function ConsolidationSettingsForm({
  values,
  onChange,
  inherited,
  missionAction,
}: {
  values: Partial<ConsolidationSettings>;
  onChange: (patch: Partial<ConsolidationSettings>) => void;
  inherited?: ConsolidationSettings;
  missionAction?: ReactNode;
}) {
  const t = useTranslations("bankConfig");

  const placeholderFor = (key: keyof ConsolidationSettings): string => {
    if (!inherited) return t("serverDefault");
    const value = inherited[key];
    if (value === null || value === undefined || value === "") {
      return t("consolidationStrategiesInheritedServerDefault");
    }
    // -1 is how every one of these limits spells "no limit".
    const shown = value === -1 ? t("consolidationStrategiesUnlimited") : String(value);
    return t("consolidationStrategiesInheritedValue", { value: shown });
  };

  const numberField = (
    key:
      | "max_observations_per_scope"
      | "consolidation_source_facts_max_tokens"
      | "consolidation_source_facts_max_tokens_per_observation",
    label: string,
    description: string
  ) => (
    <FieldRow label={label} description={description}>
      <Input
        type="number"
        min={-1}
        value={values[key] ?? ""}
        onChange={(e) => onChange({ [key]: e.target.value ? parseInt(e.target.value, 10) : null })}
        placeholder={placeholderFor(key)}
      />
    </FieldRow>
  );

  return (
    <div>
      <TextareaRow
        label={t("missionLabel")}
        description={
          inherited
            ? t("consolidationStrategyMissionDescription")
            : t("observationsMissionDescription")
        }
        value={values.observations_mission ?? ""}
        onChange={(v) => onChange({ observations_mission: v || null })}
        placeholder={
          inherited ? placeholderFor("observations_mission") : t("observationsMissionPlaceholder")
        }
        rows={3}
        action={missionAction}
      />
      {/* The bank-level descriptions say "blank = server default"; in a strategy tab
          blank means "inherit from Default", so those tabs get their own wording. */}
      {numberField(
        "max_observations_per_scope",
        t("maxObservationsPerScopeLabel"),
        inherited
          ? t("consolidationStrategyMaxObservationsDescription")
          : t("maxObservationsPerScopeDescription")
      )}
      {numberField(
        "consolidation_source_facts_max_tokens",
        t("sourceFactsMaxTokensLabel"),
        inherited
          ? t("consolidationStrategySourceFactsDescription")
          : t("sourceFactsMaxTokensDescription")
      )}
      {numberField(
        "consolidation_source_facts_max_tokens_per_observation",
        t("sourceFactsMaxTokensPerObservationLabel"),
        inherited
          ? t("consolidationStrategySourceFactsPerObservationDescription")
          : t("sourceFactsMaxTokensPerObservationDescription")
      )}
    </div>
  );
}

function GeminiSafetyEditor({
  value,
  onChange,
}: {
  value: GeminiSafetySetting[];
  onChange: (settings: GeminiSafetySetting[]) => void;
}) {
  const t = useTranslations("bankConfig");
  const harmCategories = getGeminiHarmCategories(t);
  const thresholds = getGeminiThresholds(t);
  const getThreshold = (category: string): string => {
    return value.find((s) => s.category === category)?.threshold ?? "BLOCK_MEDIUM_AND_ABOVE";
  };

  const setThreshold = (category: string, threshold: string) => {
    const next = harmCategories.map((c) => ({
      category: c.value,
      threshold: c.value === category ? threshold : getThreshold(c.value),
    }));
    onChange(next);
  };

  return (
    <div className="px-6 py-4 space-y-3">
      <p className="text-xs text-muted-foreground">
        {t("geminiSafetyEditorDescriptionPart1")}{" "}
        <a
          href="https://ai.google.dev/gemini-api/docs/safety-settings"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-foreground transition-colors"
        >
          {t("geminiSafetyEditorLearnMore")}
        </a>
      </p>
      <div className="space-y-2">
        {harmCategories.map((cat) => (
          <div key={cat.value} className="flex items-center justify-between gap-4">
            <span className="text-sm">{cat.label}</span>
            <Select
              value={getThreshold(cat.value)}
              onValueChange={(v) => setThreshold(cat.value, v)}
            >
              <SelectTrigger className="w-48 h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {thresholds.map((th) => (
                  <SelectItem key={th.value} value={th.value} className="text-xs">
                    {th.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        ))}
      </div>
    </div>
  );
}
