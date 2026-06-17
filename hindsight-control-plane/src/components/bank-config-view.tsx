"use client";

import { useState, useEffect, useRef, useMemo, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { client } from "@/lib/api";
import {
  deserializeRetainStrategies,
  serializeRetainStrategies,
  type RetainStrategy,
  type RetainStrategyValues,
} from "@/lib/retain-strategy-config";
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
import { Loader2, AlertCircle, Plus, Trash2, ChevronDown, ChevronRight } from "lucide-react";
import { Card } from "@/components/ui/card";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ProfileData {
  reflect_mission: string;
  disposition_skepticism: number;
  disposition_literalism: number;
  disposition_empathy: number;
}

type RetainEdits = {
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

type ObservationsEdits = {
  enable_observations: boolean | null;
  consolidation_llm_batch_size: number | null;
  consolidation_source_facts_max_tokens: number | null;
  consolidation_source_facts_max_tokens_per_observation: number | null;
  observations_mission: string | null;
  max_observations_per_scope: number | null;
};

type LabelValue = { value: string; description: string };
type MapField = {
  type: "text" | "value" | "multi-values" | "map";
  description: string;
  values?: LabelValue[];
  fields?: Record<string, MapField>;
};
type LabelGroup = {
  key: string;
  description: string;
  type: "value" | "multi-values" | "text" | "map";
  optional: boolean;
  tag: boolean;
  values: LabelValue[];
  fields: Record<string, MapField>;
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

function observationsSlice(config: Record<string, any>): ObservationsEdits {
  return {
    enable_observations: config.enable_observations ?? null,
    consolidation_llm_batch_size: config.consolidation_llm_batch_size ?? null,
    consolidation_source_facts_max_tokens: config.consolidation_source_facts_max_tokens ?? null,
    consolidation_source_facts_max_tokens_per_observation:
      config.consolidation_source_facts_max_tokens_per_observation ?? null,
    observations_mission: config.observations_mission ?? null,
    max_observations_per_scope: config.max_observations_per_scope ?? null,
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

const DEFAULT_PROFILE: ProfileData = {
  reflect_mission: "",
  disposition_skepticism: 3,
  disposition_literalism: 3,
  disposition_empathy: 3,
};

// ─── BankConfigView ───────────────────────────────────────────────────────────

export function BankConfigView() {
  const t = useTranslations("bankConfig");
  const { currentBank: bankId } = useBank();
  const { features } = useFeatures();
  const bankConfigEnabled = features?.bank_config_api ?? true; // optimistic default while loading
  const [loading, setLoading] = useState(true);

  // Source of truth
  const [baseConfig, setBaseConfig] = useState<Record<string, any>>({});
  const [baseProfile, setBaseProfile] = useState<ProfileData>(DEFAULT_PROFILE);

  // Per-section local edits
  const [retainEdits, setRetainEdits] = useState<RetainEdits>(retainSlice({}));
  const [strategiesEdits, setStrategiesEdits] = useState<StrategiesEdits>(strategiesSlice({}));
  const [observationsEdits, setObservationsEdits] = useState<ObservationsEdits>(
    observationsSlice({})
  );
  const [reflectEdits, setReflectEdits] = useState<ProfileData>(DEFAULT_PROFILE);
  const [mcpEdits, setMcpEdits] = useState<MCPEdits>(mcpSlice({}));
  const [geminiEdits, setGeminiEdits] = useState<GeminiEdits>(geminiSlice({}));

  // Per-section saving/error state
  const [retainSaving, setRetainSaving] = useState(false);
  const [observationsSaving, setObservationsSaving] = useState(false);
  const [reflectSaving, setReflectSaving] = useState(false);
  const [mcpSaving, setMcpSaving] = useState(false);
  const [geminiSaving, setGeminiSaving] = useState(false);
  const [retainError, setRetainError] = useState<string | null>(null);
  const [observationsError, setObservationsError] = useState<string | null>(null);
  const [reflectError, setReflectError] = useState<string | null>(null);
  const [mcpError, setMcpError] = useState<string | null>(null);
  const [geminiError, setGeminiError] = useState<string | null>(null);

  // Dirty tracking
  const retainDirty = useMemo(
    () =>
      JSON.stringify(retainEdits) !== JSON.stringify(retainSlice(baseConfig)) ||
      JSON.stringify(strategiesEdits) !== JSON.stringify(strategiesSlice(baseConfig)),
    [retainEdits, strategiesEdits, baseConfig]
  );
  const observationsDirty = useMemo(
    () => JSON.stringify(observationsEdits) !== JSON.stringify(observationsSlice(baseConfig)),
    [observationsEdits, baseConfig]
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
  useEffect(() => {
    if (bankId) loadAll();
  }, [bankId]);

  const loadAll = async () => {
    if (!bankId) return;
    setLoading(true);
    try {
      const [configResp, profileResp] = await Promise.all([
        client.getBankConfig(bankId),
        client.getBankProfile(bankId),
      ]);
      const cfg = configResp.config;
      const prof: ProfileData = {
        reflect_mission: profileResp.mission ?? "",
        disposition_skepticism:
          cfg.disposition_skepticism ?? profileResp.disposition?.skepticism ?? 3,
        disposition_literalism:
          cfg.disposition_literalism ?? profileResp.disposition?.literalism ?? 3,
        disposition_empathy: cfg.disposition_empathy ?? profileResp.disposition?.empathy ?? 3,
      };
      setBaseConfig(cfg);
      setBaseProfile(prof);
      setRetainEdits(retainSlice(cfg));
      setStrategiesEdits(strategiesSlice(cfg));
      setObservationsEdits(observationsSlice(cfg));
      setReflectEdits(prof);
      setMcpEdits(mcpSlice(cfg));
      setGeminiEdits(geminiSlice(cfg));
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
    try {
      await client.updateBankConfig(bankId, observationsEdits);
      setBaseConfig((prev) => ({ ...prev, ...observationsEdits }));
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
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
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
            <div className="flex justify-end">
              <Switch
                checked={observationsEdits.enable_observations ?? false}
                onCheckedChange={(v) =>
                  setObservationsEdits((prev) => ({ ...prev, enable_observations: v }))
                }
              />
            </div>
          </FieldRow>
          <TextareaRow
            label={t("missionLabel")}
            description={t("observationsMissionDescription")}
            value={observationsEdits.observations_mission ?? ""}
            onChange={(v) =>
              setObservationsEdits((prev) => ({ ...prev, observations_mission: v || null }))
            }
            placeholder={t("observationsMissionPlaceholder")}
            rows={3}
          />
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
          <FieldRow
            label={t("sourceFactsMaxTokensLabel")}
            description={t("sourceFactsMaxTokensDescription")}
          >
            <Input
              type="number"
              min={-1}
              value={observationsEdits.consolidation_source_facts_max_tokens ?? ""}
              onChange={(e) =>
                setObservationsEdits((prev) => ({
                  ...prev,
                  consolidation_source_facts_max_tokens: e.target.value
                    ? parseInt(e.target.value, 10)
                    : null,
                }))
              }
              placeholder={t("serverDefault")}
            />
          </FieldRow>
          <FieldRow
            label={t("sourceFactsMaxTokensPerObservationLabel")}
            description={t("sourceFactsMaxTokensPerObservationDescription")}
          >
            <Input
              type="number"
              min={-1}
              value={observationsEdits.consolidation_source_facts_max_tokens_per_observation ?? ""}
              onChange={(e) =>
                setObservationsEdits((prev) => ({
                  ...prev,
                  consolidation_source_facts_max_tokens_per_observation: e.target.value
                    ? parseInt(e.target.value, 10)
                    : null,
                }))
              }
              placeholder={t("serverDefault")}
            />
          </FieldRow>
          <FieldRow
            label={t("maxObservationsPerScopeLabel")}
            description={t("maxObservationsPerScopeDescription")}
          >
            <Input
              type="number"
              min={-1}
              value={observationsEdits.max_observations_per_scope ?? ""}
              onChange={(e) =>
                setObservationsEdits((prev) => ({
                  ...prev,
                  max_observations_per_scope: e.target.value ? parseInt(e.target.value, 10) : null,
                }))
              }
              placeholder={t("serverDefault")}
            />
          </FieldRow>
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
          />
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

        {/* MCP Tools Section */}
        <ConfigSection
          title={t("mcpToolsTitle")}
          description={t("mcpToolsDescription")}
          error={mcpError}
          dirty={mcpDirty}
          saving={mcpSaving}
          onSave={saveMCP}
        >
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

const EXTRACTION_MODES = ["concise", "verbose", "verbatim", "chunks", "custom"];
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
            {EXTRACTION_MODES.map((opt) => (
              <SelectItem key={opt} value={opt}>
                {opt}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
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
        <button
          type="button"
          onClick={() => setSelectedTab("default")}
          className={`relative py-3 px-4 text-sm font-semibold transition-colors border-b-2 -mb-px ${
            selectedTab === "default"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground hover:border-border"
          }`}
        >
          {t("default")}
        </button>

        {/* Named strategy tabs */}
        {local.map((s) => (
          <div
            key={s.id}
            className={`relative flex items-center gap-2 py-3 px-4 text-sm font-semibold transition-colors border-b-2 -mb-px cursor-pointer ${
              selectedTab === s.id
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground hover:border-border"
            }`}
            onClick={() => setSelectedTab(s.id)}
          >
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
}: {
  title: string;
  description: string;
  children: ReactNode;
  error: string | null;
  dirty: boolean;
  saving: boolean;
  onSave: () => void;
}) {
  const t = useTranslations("bankConfig");
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="text-sm text-muted-foreground">{description}</p>
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
        <div className="px-6 py-4 flex justify-end border-t border-border/40">
          <Button size="sm" disabled={!dirty || saving} onClick={onSave}>
            {saving ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                {t("saving")}
              </>
            ) : (
              t("saveChanges")
            )}
          </Button>
        </div>
      </Card>
    </section>
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
}: {
  label: string;
  description?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
}) {
  return (
    <div className="px-6 py-4">
      <div className="space-y-2">
        <div>
          <p className="text-sm font-medium">{label}</p>
          {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
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

// ─── MapFieldsEditor (recursive) ─────────────────────────────────────────────

/** Build an output-example string for the badge. */
function exampleBadge(
  key: string,
  attr: { type: string; values?: LabelValue[]; fields?: Record<string, MapField> }
): string {
  if (attr.type === "map" && attr.fields && Object.keys(attr.fields).length > 0)
    return `e.g. ${Object.keys(attr.fields)
      .slice(0, 2)
      .map((f) => `${key}:${f}:<value>`)
      .join(", ")}`;
  if (attr.type === "text") return `e.g. ${key}:<any text>`;
  if ((attr.values?.length ?? 0) > 0) return `e.g. ${key}:${attr.values![0].value || "<value>"}`;
  return `e.g. ${key}:<value>`;
}

function MapFieldsEditor({
  fields,
  onChange,
  depth,
  extraControls,
  examplePrefix,
}: {
  fields: Record<string, MapField>;
  onChange: (fields: Record<string, MapField>) => void;
  depth: number;
  extraControls?: React.ReactNode;
  examplePrefix?: string;
}) {
  const t = useTranslations("bankConfig");
  const FIELD_TYPE_LABELS: Record<MapField["type"], string> = {
    text: t("fieldTypeText"),
    value: t("fieldTypeValue"),
    "multi-values": t("fieldTypeMultiValues"),
    map: t("fieldTypeMap"),
  };
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  const updateField = (oldName: string, patch: Partial<MapField>) => {
    const newFields: Record<string, MapField> = {};
    for (const [k, v] of Object.entries(fields)) {
      newFields[k] = k === oldName ? { ...v, ...patch } : v;
    }
    onChange(newFields);
  };

  const renameField = (oldName: string, newName: string) => {
    const newFields: Record<string, MapField> = {};
    for (const [k, v] of Object.entries(fields)) {
      newFields[k === oldName ? newName : k] = v;
    }
    onChange(newFields);
  };

  const removeField = (name: string) => {
    const newFields = { ...fields };
    delete newFields[name];
    onChange(newFields);
  };

  const addField = () => {
    const newFields = { ...fields, "": { type: "text" as const, description: "" } };
    onChange(newFields);
  };

  const isRoot = depth === 0;

  return (
    <div
      className={
        isRoot ? "space-y-1.5 py-1" : "space-y-1.5 py-1 ml-3 border-l-2 border-border/40 pl-3"
      }
    >
      {Object.keys(fields).length === 0 && (
        <p className="text-xs text-muted-foreground italic">{t("noFieldsYet")}</p>
      )}
      {Object.entries(fields).map(([fieldName, field], fi) => {
        const isNestedMap = field.type === "map";
        const hasEnum = field.type === "value" || field.type === "multi-values";
        const isOpen = expanded[fi] ?? true;
        const hasExpandable = isNestedMap || hasEnum;
        return (
          <div key={fi} className="space-y-1">
            {/* Field row */}
            <div className="flex items-center gap-1.5">
              {hasExpandable ? (
                <button
                  type="button"
                  onClick={() => setExpanded((prev) => ({ ...prev, [fi]: !isOpen }))}
                  className="text-muted-foreground hover:text-foreground shrink-0 p-0.5 rounded hover:bg-muted/50"
                >
                  {isOpen ? (
                    <ChevronDown className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5" />
                  )}
                </button>
              ) : (
                <span className="w-[18px] shrink-0" />
              )}
              <Input
                placeholder={t("fieldNamePlaceholder")}
                value={fieldName}
                onChange={(e) => renameField(fieldName, e.target.value)}
                className="h-7 text-xs font-mono w-28 shrink-0"
              />
              <Input
                placeholder={t("extractorHintWhatPlaceholder")}
                value={field.description}
                onChange={(e) => updateField(fieldName, { description: e.target.value })}
                className="h-7 text-xs flex-1 min-w-0"
              />
              <Select
                value={field.type}
                onValueChange={(v: MapField["type"]) =>
                  updateField(fieldName, {
                    type: v,
                    ...(v === "map" ? { fields: field.fields ?? {}, values: undefined } : {}),
                    ...(v === "text" ? { fields: undefined, values: undefined } : {}),
                    ...(v === "value" || v === "multi-values"
                      ? { fields: undefined, values: field.values ?? [] }
                      : {}),
                  })
                }
              >
                <SelectTrigger className="h-7 text-xs w-[120px] shrink-0 px-2 py-0">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(FIELD_TYPE_LABELS).map(([val, label]) => (
                    <SelectItem key={val} value={val} className="text-xs">
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {extraControls}
              <button
                type="button"
                onClick={() => removeField(fieldName)}
                className="text-muted-foreground hover:text-destructive shrink-0 p-0.5 rounded hover:bg-destructive/10"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Example badge — only at root level to avoid clutter */}
            {isRoot && examplePrefix && fieldName && (
              <div className="ml-[18px] pl-1.5">
                <span className="text-[10px] font-mono text-muted-foreground/60 leading-none">
                  {exampleBadge(examplePrefix, field)}
                </span>
              </div>
            )}

            {/* Nested map fields */}
            {isOpen && isNestedMap && (
              <MapFieldsEditor
                fields={field.fields ?? {}}
                onChange={(subFields) => updateField(fieldName, { fields: subFields })}
                depth={depth + 1}
                examplePrefix={examplePrefix ? `${examplePrefix}:${fieldName}` : undefined}
              />
            )}

            {/* Enum values for value/multi-values fields */}
            {isOpen && hasEnum && (
              <div className="ml-6 space-y-0.5 py-1">
                {(field.values ?? []).length === 0 && (
                  <p className="text-[11px] text-muted-foreground italic">{t("noValuesYet")}</p>
                )}
                {(field.values ?? []).map((v, vi) => (
                  <div key={vi} className="flex items-center gap-1.5 group/val">
                    <span className="text-muted-foreground/50 text-[10px] shrink-0">&#x2022;</span>
                    <Input
                      placeholder={t("addValueShort")}
                      value={v.value}
                      onChange={(e) => {
                        const newValues = [...(field.values ?? [])];
                        newValues[vi] = { ...v, value: e.target.value };
                        updateField(fieldName, { values: newValues });
                      }}
                      className="h-6 text-[11px] font-mono w-24 shrink-0 border-dashed"
                    />
                    <Input
                      placeholder={t("extractorHintWhichPlaceholder")}
                      value={v.description}
                      onChange={(e) => {
                        const newValues = [...(field.values ?? [])];
                        newValues[vi] = { ...v, description: e.target.value };
                        updateField(fieldName, { values: newValues });
                      }}
                      className="h-6 text-[11px] flex-1 min-w-0 border-dashed"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const newValues = (field.values ?? []).filter((_, i) => i !== vi);
                        updateField(fieldName, { values: newValues });
                      }}
                      className="text-muted-foreground/40 hover:text-destructive shrink-0 p-0.5 rounded hover:bg-destructive/10 opacity-0 group-hover/val:opacity-100 transition-opacity"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={() => {
                    const newValues = [...(field.values ?? []), { value: "", description: "" }];
                    updateField(fieldName, { values: newValues });
                  }}
                  className="text-[11px] text-muted-foreground/60 hover:text-foreground inline-flex items-center gap-1 ml-2.5"
                >
                  <Plus className="h-2.5 w-2.5" />
                  {t("addValueShort")}
                </button>
              </div>
            )}
          </div>
        );
      })}
      <button
        type="button"
        onClick={addField}
        className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
      >
        <Plus className="h-3 w-3" />
        {t("addField")}
      </button>
    </div>
  );
}

// ─── EntityLabelsEditor ───────────────────────────────────────────────────────

function emptyAttribute(): LabelGroup {
  return {
    key: "",
    description: "",
    type: "value",
    optional: true,
    tag: false,
    values: [],
    fields: {},
  };
}

function EntityLabelsEditor({
  value,
  onChange,
}: {
  value: LabelGroup[];
  onChange: (attrs: LabelGroup[]) => void;
}) {
  const t = useTranslations("entityLabelsEditor");
  const updateAttr = (i: number, patch: Partial<LabelGroup>) => {
    const next = value.map((a, idx) => (idx === i ? { ...a, ...patch } : a));
    onChange(next);
  };

  const removeAttr = (i: number) => {
    onChange(value.filter((_, idx) => idx !== i));
  };

  const addAttr = () => {
    onChange([...value, emptyAttribute()]);
  };

  return (
    <div className="px-6 py-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">{t("entityLabelsTitle")}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{t("entityLabelsDescription")}</p>
        </div>
        {value.length > 0 && (
          <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full shrink-0">
            {t("labelCount", { count: value.length })}
          </span>
        )}
      </div>

      {value.length === 0 && (
        <p className="text-xs text-muted-foreground italic">{t("noEntityLabelsDefined")}</p>
      )}

      <div className="space-y-2">
        {value.map((attr, i) => (
          <div key={i} className="border border-border/50 rounded-md bg-background">
            {/* Rendered via MapFieldsEditor as a single-field editor */}
            <MapFieldsEditor
              fields={{
                [attr.key]: {
                  type: attr.type as MapField["type"],
                  description: attr.description,
                  values: attr.values,
                  fields: attr.fields,
                },
              }}
              onChange={(updated) => {
                const entries = Object.entries(updated);
                if (entries.length === 0) {
                  removeAttr(i);
                } else {
                  const [newKey, newField] = entries[0];
                  updateAttr(i, {
                    key: newKey,
                    type: newField.type as LabelGroup["type"],
                    description: newField.description,
                    values: newField.values ?? [],
                    fields: newField.fields ?? {},
                  });
                }
              }}
              depth={0}
              extraControls={
                <label
                  className="flex items-center gap-1.5 text-xs text-muted-foreground shrink-0 cursor-pointer select-none"
                  title={t("alsoStoreAsTagTooltip")}
                >
                  <Checkbox
                    checked={attr.tag}
                    onCheckedChange={(checked) => updateAttr(i, { tag: !!checked })}
                    className="h-4 w-4"
                  />
                  {t("plusTag")}
                </label>
              }
              examplePrefix={attr.key}
            />
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={addAttr}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
      >
        <Plus className="h-3.5 w-3.5" />
        {t("addLabel")}
      </button>
    </div>
  );
}

// ─── GeminiSafetyEditor ───────────────────────────────────────────────────────

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
