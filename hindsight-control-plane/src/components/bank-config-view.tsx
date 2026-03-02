"use client";

import { useState, useEffect, useMemo, type ReactNode } from "react";
import { useBank } from "@/lib/bank-context";
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
import { Switch } from "@/components/ui/switch";
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
};

type ObservationsEdits = {
  enable_observations: boolean | null;
  observations_mission: string | null;
};

type LabelValue = { value: string; description: string };
type LabelGroup = {
  key: string;
  description: string;
  type: "value" | "multi-values" | "text";
  optional: boolean;
  values: LabelValue[];
};

type EntityLabelsEdits = {
  entity_labels: LabelGroup[] | null;
  entities_allow_free_form: boolean;
};

type MCPEdits = {
  mcp_enabled_tools: string[] | null;
};

// ─── MCP tool catalogue ───────────────────────────────────────────────────────

const MCP_TOOL_GROUPS: { label: string; tools: string[] }[] = [
  { label: "Core", tools: ["retain", "recall", "reflect"] },
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

function retainSlice(config: Record<string, any>): RetainEdits {
  return {
    retain_chunk_size: config.retain_chunk_size ?? null,
    retain_extraction_mode: config.retain_extraction_mode ?? null,
    retain_mission: config.retain_mission ?? null,
    retain_custom_instructions: config.retain_custom_instructions ?? null,
  };
}

function observationsSlice(config: Record<string, any>): ObservationsEdits {
  return {
    enable_observations: config.enable_observations ?? null,
    observations_mission: config.observations_mission ?? null,
  };
}

function entityLabelsSlice(config: Record<string, any>): EntityLabelsEdits {
  const raw = config.entity_labels;
  let attrs: LabelGroup[] | null = null;
  if (Array.isArray(raw)) {
    attrs = raw as LabelGroup[];
  } else if (raw && typeof raw === "object" && Array.isArray(raw.attributes)) {
    attrs = raw.attributes as LabelGroup[];
  }
  return {
    entity_labels: attrs,
    entities_allow_free_form: config.entities_allow_free_form ?? true,
  };
}

function mcpSlice(config: Record<string, any>): MCPEdits {
  return {
    mcp_enabled_tools: config.mcp_enabled_tools ?? null,
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
  const [loading, setLoading] = useState(true);

  // Source of truth
  const [baseConfig, setBaseConfig] = useState<Record<string, any>>({});
  const [baseProfile, setBaseProfile] = useState<ProfileData>(DEFAULT_PROFILE);

  // Per-section local edits
  const [retainEdits, setRetainEdits] = useState<RetainEdits>(retainSlice({}));
  const [observationsEdits, setObservationsEdits] = useState<ObservationsEdits>(
    observationsSlice({})
  );
  const [entityLabelsEdits, setEntityLabelsEdits] = useState<EntityLabelsEdits>(
    entityLabelsSlice({})
  );
  const [reflectEdits, setReflectEdits] = useState<ProfileData>(DEFAULT_PROFILE);
  const [mcpEdits, setMcpEdits] = useState<MCPEdits>(mcpSlice({}));

  // Per-section saving/error state
  const [retainSaving, setRetainSaving] = useState(false);
  const [observationsSaving, setObservationsSaving] = useState(false);
  const [entityLabelsSaving, setEntityLabelsSaving] = useState(false);
  const [reflectSaving, setReflectSaving] = useState(false);
  const [mcpSaving, setMcpSaving] = useState(false);
  const [retainError, setRetainError] = useState<string | null>(null);
  const [observationsError, setObservationsError] = useState<string | null>(null);
  const [entityLabelsError, setEntityLabelsError] = useState<string | null>(null);
  const [reflectError, setReflectError] = useState<string | null>(null);
  const [mcpError, setMcpError] = useState<string | null>(null);

  // Reset dialog

  // Dirty tracking
  const retainDirty = useMemo(
    () => JSON.stringify(retainEdits) !== JSON.stringify(retainSlice(baseConfig)),
    [retainEdits, baseConfig]
  );
  const observationsDirty = useMemo(
    () => JSON.stringify(observationsEdits) !== JSON.stringify(observationsSlice(baseConfig)),
    [observationsEdits, baseConfig]
  );
  const entityLabelsDirty = useMemo(
    () => JSON.stringify(entityLabelsEdits) !== JSON.stringify(entityLabelsSlice(baseConfig)),
    [entityLabelsEdits, baseConfig]
  );
  const reflectDirty = useMemo(
    () => JSON.stringify(reflectEdits) !== JSON.stringify(baseProfile),
    [reflectEdits, baseProfile]
  );
  const mcpDirty = useMemo(
    () => JSON.stringify(mcpEdits) !== JSON.stringify(mcpSlice(baseConfig)),
    [mcpEdits, baseConfig]
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
      setObservationsEdits(observationsSlice(cfg));
      setEntityLabelsEdits(entityLabelsSlice(cfg));
      setReflectEdits(prof);
      setMcpEdits(mcpSlice(cfg));
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
      await client.updateBankConfig(bankId, retainEdits);
      setBaseConfig((prev) => ({ ...prev, ...retainEdits }));
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

  const saveEntityLabels = async () => {
    if (!bankId) return;
    setEntityLabelsSaving(true);
    setEntityLabelsError(null);
    try {
      const payload = {
        entity_labels: entityLabelsEdits.entity_labels,
        entities_allow_free_form: entityLabelsEdits.entities_allow_free_form,
      };
      await client.updateBankConfig(bankId, payload);
      setBaseConfig((prev) => ({ ...prev, ...payload }));
    } catch (err: any) {
      setEntityLabelsError(err.message || "Failed to save entity labels settings");
    } finally {
      setEntityLabelsSaving(false);
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

  if (!bankId) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-muted-foreground">No bank selected</p>
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
        {/* Retain Section */}
        <ConfigSection
          title="Retain"
          description="Control what gets extracted and stored from content"
          error={retainError}
          dirty={retainDirty}
          saving={retainSaving}
          onSave={saveRetain}
        >
          <FieldRow
            label="Chunk Size"
            description="Size of text chunks for processing (characters)"
          >
            <Input
              type="number"
              min={500}
              max={8000}
              value={retainEdits.retain_chunk_size ?? ""}
              onChange={(e) =>
                setRetainEdits((prev) => ({
                  ...prev,
                  retain_chunk_size: e.target.value ? parseFloat(e.target.value) : null,
                }))
              }
            />
          </FieldRow>
          <TextareaRow
            label="Mission"
            description="What this bank should pay attention to during extraction. Steers the LLM without replacing the extraction rules — works alongside any extraction mode."
            value={retainEdits.retain_mission ?? ""}
            onChange={(v) => setRetainEdits((prev) => ({ ...prev, retain_mission: v || null }))}
            placeholder="e.g. Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics, greetings, and social exchanges."
            rows={3}
          />
          <FieldRow
            label="Extraction Mode"
            description="How aggressively to extract facts: concise (default, selective), verbose (capture everything), custom (write your own extraction rules)"
          >
            <Select
              value={retainEdits.retain_extraction_mode ?? ""}
              onValueChange={(val) =>
                setRetainEdits((prev) => ({ ...prev, retain_extraction_mode: val }))
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {["concise", "verbose", "custom"].map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {opt}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldRow>
          {retainEdits.retain_extraction_mode === "custom" && (
            <TextareaRow
              label="Custom Extraction Prompt"
              description="Replaces the built-in extraction rules entirely. Only active when Extraction Mode is set to custom."
              value={retainEdits.retain_custom_instructions ?? ""}
              onChange={(v) =>
                setRetainEdits((prev) => ({ ...prev, retain_custom_instructions: v || null }))
              }
              rows={5}
            />
          )}
        </ConfigSection>

        {/* Entity Labels Section */}
        <ConfigSection
          title="Entities"
          description="Control entity extraction and define a controlled vocabulary of key:value classification labels (e.g. pedagogy:scaffolding, interest:active)"
          error={entityLabelsError}
          dirty={entityLabelsDirty}
          saving={entityLabelsSaving}
          onSave={saveEntityLabels}
        >
          <FieldRow
            label="Free Form Entities"
            description="Extract regular named entities (people, places, concepts) alongside label groups. Disable to restrict extraction to label groups only."
          >
            <div className="flex justify-end items-center gap-2">
              <Label
                htmlFor="entities-allow-free-form"
                className="text-sm text-muted-foreground cursor-pointer select-none"
              >
                {entityLabelsEdits.entities_allow_free_form ? "Enabled" : "Disabled"}
              </Label>
              <Switch
                id="entities-allow-free-form"
                checked={entityLabelsEdits.entities_allow_free_form}
                onCheckedChange={(v) =>
                  setEntityLabelsEdits((prev) => ({ ...prev, entities_allow_free_form: v }))
                }
              />
            </div>
          </FieldRow>
          <EntityLabelsEditor
            value={entityLabelsEdits.entity_labels ?? []}
            onChange={(attrs) =>
              setEntityLabelsEdits((prev) => ({
                ...prev,
                entity_labels: attrs.length > 0 ? attrs : null,
              }))
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
      </div>
    </>
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
  description?: string;
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
          <p className="text-sm font-medium">Label Groups</p>
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
        <p className="text-xs text-muted-foreground italic">No label groups defined.</p>
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
