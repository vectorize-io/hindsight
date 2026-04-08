"use client";

import { useState, useEffect, useRef, useMemo, type ReactNode } from "react";
import { useBank } from "@/lib/bank-context";
import { useFeatures } from "@/lib/features-context";
import { client } from "@/lib/api";
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
type LabelGroup = {
  key: string;
  description: string;
  type: "value" | "multi-values" | "text";
  optional: boolean;
  tag: boolean;
  values: LabelValue[];
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

const GEMINI_HARM_CATEGORIES = [
  { value: "HARM_CATEGORY_HARASSMENT", label: "Harassment" },
  { value: "HARM_CATEGORY_HATE_SPEECH", label: "Hate Speech" },
  { value: "HARM_CATEGORY_SEXUALLY_EXPLICIT", label: "Sexually Explicit" },
  { value: "HARM_CATEGORY_DANGEROUS_CONTENT", label: "Dangerous Content" },
] as const;

const GEMINI_THRESHOLDS = [
  { value: "HARM_BLOCK_THRESHOLD_UNSPECIFIED", label: "Unspecified (use Gemini default)" },
  { value: "OFF", label: "Off (filter disabled)" },
  { value: "BLOCK_NONE", label: "Block none" },
  { value: "BLOCK_LOW_AND_ABOVE", label: "Block low & above" },
  { value: "BLOCK_MEDIUM_AND_ABOVE", label: "Block medium & above" },
  { value: "BLOCK_ONLY_HIGH", label: "Block only high" },
] as const;

const DEFAULT_GEMINI_SAFETY_SETTINGS: GeminiSafetySetting[] = GEMINI_HARM_CATEGORIES.map((c) => ({
  category: c.value,
  threshold: "BLOCK_NONE",
}));

// ─── MCP tool catalogue ───────────────────────────────────────────────────────

const MCP_TOOL_GROUPS: { label: string; tools: string[] }[] = [
  { label: "Core", tools: ["retain", "sync_retain", "recall", "reflect"] },
  {
    label: "Bank management",
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
    label: "Mental models",
    tools: [
      "list_mental_models",
      "get_mental_model",
      "create_mental_model",
      "update_mental_model",
      "delete_mental_model",
      "refresh_mental_model",
    ],
  },
  { label: "Directives", tools: ["list_directives", "create_directive", "delete_directive"] },
  { label: "Memories", tools: ["list_memories", "get_memory", "delete_memory"] },
  { label: "Documents", tools: ["list_documents", "get_document", "delete_document"] },
  { label: "Operations", tools: ["list_operations", "get_operation", "cancel_operation"] },
  { label: "Tags", tools: ["list_tags"] },
];

const ALL_TOOLS: string[] = MCP_TOOL_GROUPS.flatMap((g) => g.tools);

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
      setRetainError(err.message || "Failed to save retain settings");
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
      setObservationsError(err.message || "Failed to save observations settings");
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
      setReflectError(err.message || "Failed to save reflect settings");
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
      setMcpError(err.message || "Failed to save MCP settings");
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
      setGeminiError(err.message || "Failed to save Gemini settings");
    } finally {
      setGeminiSaving(false);
    }
  };

  if (!bankId) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-muted-foreground">No bank selected</p>
      </div>
    );
  }

  if (!bankConfigEnabled) {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
        <p className="text-base font-medium text-foreground">Bank configuration is disabled</p>
        <p className="text-sm text-muted-foreground max-w-sm">
          Set{" "}
          <code className="font-mono text-xs bg-muted px-1 py-0.5 rounded">
            HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true
          </code>{" "}
          to enable per-bank configuration.
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
          title="Retain"
          description="Default extraction settings and named strategies. Pass a strategy name on retain requests to override defaults per-item."
          error={retainError}
          dirty={retainDirty}
          saving={retainSaving}
          onSave={saveRetain}
        >
          <FieldRow
            label="Default strategy"
            description="Applied automatically when no strategy is specified on a request."
          >
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
                  <span className="text-muted-foreground italic">Default</span>
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
          title="Observations"
          description="Control how facts are synthesized into durable observations"
          error={observationsError}
          dirty={observationsDirty}
          saving={observationsSaving}
          onSave={saveObservations}
        >
          <FieldRow
            label="Enable Observations"
            description="Enable automatic consolidation of facts into observations"
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
            label="Mission"
            description="What this bank should synthesise into durable observations. Replaces the built-in consolidation rules — leave blank to use the server default."
            value={observationsEdits.observations_mission ?? ""}
            onChange={(v) =>
              setObservationsEdits((prev) => ({ ...prev, observations_mission: v || null }))
            }
            placeholder="e.g. Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events and ephemeral state."
            rows={3}
          />
          <FieldRow
            label="LLM Batch Size"
            description="Number of facts sent to the LLM in a single consolidation call. Higher values reduce LLM calls at the cost of larger prompts. Leave blank to use the server default."
          >
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
              placeholder="Server default"
            />
          </FieldRow>
          <FieldRow
            label="Source Facts Max Tokens"
            description="Total token budget for source facts included with observations during consolidation. -1 = unlimited."
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
              placeholder="Server default"
            />
          </FieldRow>
          <FieldRow
            label="Source Facts Max Tokens Per Observation"
            description="Per-observation token cap for source facts during consolidation. Each observation gets at most this many tokens of source facts. -1 = unlimited."
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
              placeholder="Server default"
            />
          </FieldRow>
          <FieldRow
            label="Max Observations Per Scope"
            description="Maximum number of observations allowed per tag scope. When the limit is reached, only updates and deletes are allowed. Observations with no tags are not subject to this limit. -1 = unlimited."
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
              placeholder="Server default"
            />
          </FieldRow>
        </ConfigSection>

        {/* Reflect Section */}
        <ConfigSection
          title="Reflect"
          description="Shape how the bank reasons and responds in reflect operations"
          error={reflectError}
          dirty={reflectDirty}
          saving={reflectSaving}
          onSave={saveReflect}
        >
          <TextareaRow
            label="Mission"
            description="Agent identity and purpose. Used as framing context in reflect."
            value={reflectEdits.reflect_mission}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, reflect_mission: v }))}
            placeholder="e.g. You are a senior engineering assistant. Always ground answers in documented decisions and rationale. Ignore speculation. Be direct and precise."
            rows={3}
          />
          <TraitRow
            label="Skepticism"
            description="How skeptical vs trusting when evaluating claims"
            lowLabel="Trusting"
            highLabel="Skeptical"
            value={reflectEdits.disposition_skepticism}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, disposition_skepticism: v }))}
          />
          <TraitRow
            label="Literalism"
            description="How literally to interpret information"
            lowLabel="Flexible"
            highLabel="Literal"
            value={reflectEdits.disposition_literalism}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, disposition_literalism: v }))}
          />
          <TraitRow
            label="Empathy"
            description="How much to weight emotional context"
            lowLabel="Detached"
            highLabel="Empathetic"
            value={reflectEdits.disposition_empathy}
            onChange={(v) => setReflectEdits((prev) => ({ ...prev, disposition_empathy: v }))}
          />
        </ConfigSection>

        {/* MCP Tools Section */}
        <ConfigSection
          title="MCP Tools"
          description="Restrict which MCP tools this bank exposes to agents"
          error={mcpError}
          dirty={mcpDirty}
          saving={mcpSaving}
          onSave={saveMCP}
        >
          <FieldRow
            label="Restrict tools"
            description="When off, all tools are available. When on, only the selected tools can be invoked for this bank."
          >
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
                {mcpEdits.mcp_enabled_tools !== null ? "Enabled" : "Disabled"}
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
          title="Models"
          description="Provider-specific model settings"
          error={geminiError}
          dirty={geminiDirty}
          saving={geminiSaving}
          onSave={saveGemini}
        >
          {/* Gemini subsection */}
          <div className="px-6 py-4 space-y-4">
            <p className="text-sm font-semibold">Gemini / Vertex AI</p>
            <div className="pl-4 border-l-2 border-border/40 space-y-4">
              <FieldRow
                label="Safety settings"
                description={
                  <>
                    When off, Gemini&apos;s default safety thresholds are used. When on, configure
                    thresholds per harm category.{" "}
                    <a
                      href="https://ai.google.dev/gemini-api/docs/safety-settings"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline hover:text-foreground transition-colors"
                    >
                      Learn more
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
                    {geminiEdits.llm_gemini_safety_settings !== null ? "Custom" : "Default"}
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

type RetainFormValues = {
  retain_extraction_mode: string | null;
  retain_chunk_size: number | null;
  retain_mission: string | null;
  retain_custom_instructions: string | null;
  entities_allow_free_form: boolean | null;
  entity_labels: LabelGroup[] | null;
};

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
  const modeValue = values.retain_extraction_mode ?? (isOverride ? INHERIT_SENTINEL : "");
  const showCustomField = values.retain_extraction_mode === "custom";

  return (
    <div className="divide-y divide-border/40">
      <FieldRow
        label="Extraction Mode"
        description="How aggressively to extract facts. concise = selective, verbose = capture everything, verbatim = store chunks as-is (still extract entities/time), chunks = no LLM, custom = write your own rules."
      >
        <Select
          value={modeValue}
          onValueChange={(val) =>
            onChange({ retain_extraction_mode: val === INHERIT_SENTINEL ? null : val || null })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder={isOverride ? "Inherited from default" : undefined} />
          </SelectTrigger>
          <SelectContent>
            {isOverride && (
              <SelectItem value={INHERIT_SENTINEL}>
                <span className="text-muted-foreground italic">inherited</span>
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
      <FieldRow label="Chunk Size" description="Size of text chunks for processing (characters)">
        <Input
          type="number"
          min={500}
          max={8000}
          value={values.retain_chunk_size ?? ""}
          onChange={(e) =>
            onChange({ retain_chunk_size: e.target.value ? parseFloat(e.target.value) : null })
          }
          placeholder={isOverride ? "Inherited from default" : undefined}
        />
      </FieldRow>
      <TextareaRow
        label="Mission"
        description="What this bank should pay attention to during extraction. Steers the LLM without replacing the extraction rules."
        value={values.retain_mission ?? ""}
        onChange={(v) => onChange({ retain_mission: v || null })}
        placeholder={
          isOverride
            ? "Inherited from default"
            : "e.g. Always include technical decisions, API design choices, and architectural trade-offs."
        }
        rows={3}
      />
      {showCustomField && (
        <TextareaRow
          label="Custom Extraction Prompt"
          description="Replaces the built-in extraction rules entirely. Only active when Extraction Mode is set to custom."
          value={values.retain_custom_instructions ?? ""}
          onChange={(v) => onChange({ retain_custom_instructions: v || null })}
          rows={5}
        />
      )}
      <FieldRow
        label="Free Form Entities"
        description="Extract regular named entities (people, places, concepts) alongside entity labels. Disable to restrict extraction to entity labels only."
      >
        <div className="flex justify-end items-center gap-2">
          <Label className="text-sm text-muted-foreground cursor-pointer select-none">
            {(values.entities_allow_free_form ?? true) ? "Enabled" : "Disabled"}
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

type LocalStrategy = { id: number; name: string; values: RetainFormValues };

function fromStrategiesDict(dict: Record<string, Record<string, any>> | null): LocalStrategy[] {
  if (!dict) return [];
  return Object.entries(dict).map(([name, overrides], i) => ({
    id: i,
    name,
    values: {
      retain_extraction_mode: overrides.retain_extraction_mode ?? null,
      retain_chunk_size: overrides.retain_chunk_size ?? null,
      retain_mission: overrides.retain_mission ?? null,
      retain_custom_instructions: overrides.retain_custom_instructions ?? null,
      entities_allow_free_form: overrides.entities_allow_free_form ?? null,
      entity_labels: parseEntityLabels(overrides.entity_labels),
    },
  }));
}

function toStrategiesDict(local: LocalStrategy[]): Record<string, Record<string, any>> | null {
  const dict: Record<string, Record<string, any>> = {};
  for (const s of local) {
    if (!s.name.trim()) continue;
    const overrides: Record<string, any> = {};
    if (s.values.retain_extraction_mode !== null)
      overrides.retain_extraction_mode = s.values.retain_extraction_mode;
    if (s.values.retain_chunk_size !== null)
      overrides.retain_chunk_size = s.values.retain_chunk_size;
    if (s.values.retain_mission) overrides.retain_mission = s.values.retain_mission;
    if (s.values.retain_custom_instructions)
      overrides.retain_custom_instructions = s.values.retain_custom_instructions;
    if (s.values.entities_allow_free_form !== null)
      overrides.entities_allow_free_form = s.values.entities_allow_free_form;
    if (s.values.entity_labels !== null) overrides.entity_labels = s.values.entity_labels;
    dict[s.name.trim()] = overrides;
  }
  return Object.keys(dict).length > 0 ? dict : null;
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
          Default
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
              {s.name || <span className="italic font-normal opacity-50">unnamed</span>}
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
          Add strategy
        </button>
      </div>

      {/* Form */}
      <div>
        {selectedTab === "default" ? (
          <RetainStrategyForm values={defaultValues} onChange={onDefaultChange} />
        ) : activeStrategy ? (
          <div>
            <div className="px-6 py-3 flex items-center gap-3 border-b border-border/40">
              <label className="text-xs text-muted-foreground shrink-0">Name</label>
              <div className="flex flex-col gap-1">
                <Input
                  value={activeStrategy.name}
                  onChange={(e) => updateStrategy(activeStrategy.id, { name: e.target.value })}
                  placeholder="strategy name (e.g. fast)"
                  className={`h-7 text-xs font-mono max-w-[200px] ${!activeStrategy.name.trim() ? "border-destructive focus-visible:ring-destructive" : ""}`}
                />
                {!activeStrategy.name.trim() && (
                  <p className="text-xs text-destructive">Name is required</p>
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
              Delete strategy &ldquo;{pendingDelete?.name || "unnamed"}&rdquo;?
            </AlertDialogTitle>
            <AlertDialogDescription>
              This will remove the strategy and all its overrides. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                if (pendingDelete) {
                  removeStrategy(pendingDelete.id);
                  setPendingDelete(null);
                }
              }}
            >
              Delete
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
  const selectedSet = new Set(selected);

  const toggleTool = (tool: string) => {
    const next = new Set(selectedSet);
    if (next.has(tool)) {
      next.delete(tool);
    } else {
      next.add(tool);
    }
    onChange(ALL_TOOLS.filter((t) => next.has(t)));
  };

  const allSelected = ALL_TOOLS.every((t) => selectedSet.has(t));
  const noneSelected = selected.length === 0;

  const toggleAll = () => {
    onChange(allSelected ? [] : [...ALL_TOOLS]);
  };

  return (
    <div className="px-6 py-4 space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {selected.length} of {ALL_TOOLS.length} tools enabled
        </p>
        <button type="button" onClick={toggleAll} className="text-xs text-primary hover:underline">
          {allSelected ? "Deselect all" : "Select all"}
        </button>
      </div>
      <div className="space-y-4">
        {MCP_TOOL_GROUPS.map((group) => {
          const groupSelected = group.tools.filter((t) => selectedSet.has(t)).length;
          const groupAll = groupSelected === group.tools.length;
          return (
            <div key={group.label}>
              <div className="flex items-center justify-between mb-1.5">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  {group.label}
                </p>
                <button
                  type="button"
                  onClick={() => {
                    const next = new Set(selectedSet);
                    if (groupAll) {
                      group.tools.forEach((t) => next.delete(t));
                    } else {
                      group.tools.forEach((t) => next.add(t));
                    }
                    onChange(ALL_TOOLS.filter((t) => next.has(t)));
                  }}
                  className="text-xs text-primary hover:underline"
                >
                  {groupAll ? "Deselect" : "Select all"}
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
      {noneSelected && (
        <p className="text-xs text-destructive">
          Warning: no tools selected — agents will be blocked from all MCP calls for this bank.
        </p>
      )}
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
                Saving...
              </>
            ) : (
              "Save changes"
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

// ─── EntityLabelsEditor ───────────────────────────────────────────────────────

function emptyAttribute(): LabelGroup {
  return {
    key: "",
    description: "",
    type: "value",
    optional: true,
    tag: false,
    values: [],
  };
}

function emptyValue(): LabelValue {
  return { value: "", description: "" };
}

function EntityLabelsEditor({
  value,
  onChange,
}: {
  value: LabelGroup[];
  onChange: (attrs: LabelGroup[]) => void;
}) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  const updateAttr = (i: number, patch: Partial<LabelGroup>) => {
    const next = value.map((a, idx) => (idx === i ? { ...a, ...patch } : a));
    onChange(next);
  };

  const removeAttr = (i: number) => {
    onChange(value.filter((_, idx) => idx !== i));
    setExpanded((prev) => {
      const next = { ...prev };
      delete next[i];
      return next;
    });
  };

  const addAttr = () => {
    const next = [...value, emptyAttribute()];
    onChange(next);
    setExpanded((prev) => ({ ...prev, [next.length - 1]: true }));
  };

  const updateVal = (attrIdx: number, valIdx: number, patch: Partial<LabelValue>) => {
    const newValues = value[attrIdx].values.map((v, vi) =>
      vi === valIdx ? { ...v, ...patch } : v
    );
    updateAttr(attrIdx, { values: newValues });
  };

  const removeVal = (attrIdx: number, valIdx: number) => {
    updateAttr(attrIdx, { values: value[attrIdx].values.filter((_, vi) => vi !== valIdx) });
  };

  const addVal = (attrIdx: number) => {
    updateAttr(attrIdx, { values: [...value[attrIdx].values, emptyValue()] });
  };

  return (
    <div className="px-6 py-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">Entity Labels</p>
          <p className="text-xs text-muted-foreground mt-0.5">
            Classification labels extracted at retain time. Leave empty to disable.
          </p>
        </div>
        {value.length > 0 && (
          <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full shrink-0">
            {value.length} group{value.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {value.length === 0 && (
        <p className="text-xs text-muted-foreground italic">No entity labels defined.</p>
      )}

      <div className="space-y-2">
        {value.map((attr, i) => {
          const isOpen = expanded[i] ?? false;
          const isText = attr.type === "text";
          const hasValues = !isText;
          return (
            <div key={i} className="border border-border/50 rounded-md bg-background">
              {/* Attribute header */}
              <div className="flex items-center gap-2 px-3 py-2">
                <button
                  type="button"
                  onClick={() => setExpanded((prev) => ({ ...prev, [i]: !isOpen }))}
                  className="text-muted-foreground hover:text-foreground shrink-0"
                  disabled={isText}
                >
                  {isOpen && hasValues ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className={`h-4 w-4 ${isText ? "opacity-30" : ""}`} />
                  )}
                </button>
                <Input
                  placeholder="key (e.g. pedagogy)"
                  value={attr.key}
                  onChange={(e) => updateAttr(i, { key: e.target.value })}
                  className="h-8 text-xs font-mono w-36 shrink-0"
                />
                <Input
                  placeholder={isText ? "description / examples" : "description"}
                  value={attr.description}
                  onChange={(e) => updateAttr(i, { description: e.target.value })}
                  className="h-8 text-xs flex-1 min-w-0"
                />
                {/* Type dropdown */}
                <Select
                  value={attr.type}
                  onValueChange={(v: "value" | "multi-values" | "text") =>
                    updateAttr(i, {
                      type: v,
                      // reset values when switching to free text
                      ...(v === "text" ? { values: [] } : {}),
                    })
                  }
                >
                  <SelectTrigger className="h-8 text-xs w-32 shrink-0">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="value" className="text-xs">
                      Single value
                    </SelectItem>
                    <SelectItem value="multi-values" className="text-xs">
                      Multi-values
                    </SelectItem>
                    <SelectItem value="text" className="text-xs">
                      Free text
                    </SelectItem>
                  </SelectContent>
                </Select>
                {/* Tag checkbox — also write extracted labels as tags */}
                <label className="flex items-center gap-1.5 text-xs text-muted-foreground shrink-0 cursor-pointer select-none">
                  <Checkbox
                    checked={attr.tag}
                    onCheckedChange={(checked) => updateAttr(i, { tag: !!checked })}
                    className="h-4 w-4"
                  />
                  tag
                </label>
                <button
                  type="button"
                  onClick={() => removeAttr(i)}
                  className="text-muted-foreground hover:text-destructive shrink-0"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>

              {/* Values list — enum and multi-values only */}
              {isOpen && hasValues && (
                <div className="px-3 pb-3 space-y-1 border-t border-border/30 pt-2">
                  {attr.values.length === 0 && (
                    <p className="text-xs text-muted-foreground italic pl-5">No values yet.</p>
                  )}
                  {attr.values.map((v, vi) => (
                    <div key={vi} className="flex items-center gap-2 pl-5">
                      <Input
                        placeholder="value"
                        value={v.value}
                        onChange={(e) => updateVal(i, vi, { value: e.target.value })}
                        className="h-8 text-xs font-mono w-32 shrink-0"
                      />
                      <Input
                        placeholder="description"
                        value={v.description}
                        onChange={(e) => updateVal(i, vi, { description: e.target.value })}
                        className="h-8 text-xs flex-1 min-w-0"
                      />
                      <button
                        type="button"
                        onClick={() => removeVal(i, vi)}
                        className="text-muted-foreground hover:text-destructive shrink-0"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() => addVal(i)}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground pl-5 mt-1"
                  >
                    <Plus className="h-3 w-3" />
                    Add value
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <button
        type="button"
        onClick={addAttr}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
      >
        <Plus className="h-3.5 w-3.5" />
        Add attribute
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
  const getThreshold = (category: string): string => {
    return value.find((s) => s.category === category)?.threshold ?? "BLOCK_MEDIUM_AND_ABOVE";
  };

  const setThreshold = (category: string, threshold: string) => {
    const next = GEMINI_HARM_CATEGORIES.map((c) => ({
      category: c.value,
      threshold: c.value === category ? threshold : getThreshold(c.value),
    }));
    onChange(next);
  };

  return (
    <div className="px-6 py-4 space-y-3">
      <p className="text-xs text-muted-foreground">
        Set the blocking threshold for each harm category. "Off" disables the filter entirely
        (default for Gemini 2.5+). Lower thresholds block more content.{" "}
        <a
          href="https://ai.google.dev/gemini-api/docs/safety-settings"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-foreground transition-colors"
        >
          Learn more
        </a>
      </p>
      <div className="space-y-2">
        {GEMINI_HARM_CATEGORIES.map((cat) => (
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
                {GEMINI_THRESHOLDS.map((t) => (
                  <SelectItem key={t.value} value={t.value} className="text-xs">
                    {t.label}
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
