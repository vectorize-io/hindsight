"use client";

import { useId, useState, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import cronstrue from "cronstrue";
import { CalendarClock, FilePenLine, FileText, Hand, Zap } from "lucide-react";
import { client, type TagGroup, type TagsMatch } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FactType, FactTypeCheckboxGroup } from "@/components/fact-type-filter";
import { ResponseSchemaField } from "./response-schema-field";
import { CronSchedulePreview } from "./cron-schedule-preview";
import { DisclosureButton, Hint, OptionCards, Row, Section } from "./form-layout";

type MentalModelTrigger = NonNullable<Parameters<typeof client.updateMentalModel>[2]["trigger"]>;

// The trigger as the form edits it: free-text inputs stay strings until
// triggerFromForm parses them, "" meaning "inherit the bank/global default".
export type TriggerForm = {
  refreshTrigger: "manual" | "auto" | "scheduled";
  mode: "full" | "delta";
  refreshCron: string;
  minRefreshIntervalSeconds: string;
  factTypes: FactType[];
  excludeMentalModels: boolean;
  excludeMentalModelIds: string;
  tagsMatch: string;
  tagGroups: string;
  includeChunks: "" | "true" | "false";
  recallMaxTokens: string;
  recallChunksMaxTokens: string;
  observationsMaxTokens: string;
  observationsIncludeEntities: "" | "true" | "false";
  responseSchema: string;
  keepTrace: boolean;
};

export function triggerFormFromTrigger(trigger?: Partial<MentalModelTrigger> | null): TriggerForm {
  return {
    refreshTrigger: trigger?.refresh_cron
      ? "scheduled"
      : trigger?.refresh_after_consolidation
        ? "auto"
        : "manual",
    mode: trigger?.mode || "full",
    refreshCron: trigger?.refresh_cron || "",
    minRefreshIntervalSeconds:
      trigger?.min_refresh_interval_seconds != null
        ? String(trigger.min_refresh_interval_seconds)
        : "",
    factTypes: (trigger?.fact_types as FactType[] | null | undefined) || [],
    excludeMentalModels: trigger?.exclude_mental_models || false,
    excludeMentalModelIds: (trigger?.exclude_mental_model_ids || []).join(", "),
    tagsMatch: trigger?.tags_match || "",
    tagGroups: trigger?.tag_groups ? JSON.stringify(trigger.tag_groups, null, 2) : "",
    includeChunks:
      trigger?.include_chunks === true ? "true" : trigger?.include_chunks === false ? "false" : "",
    recallMaxTokens: trigger?.recall_max_tokens != null ? String(trigger.recall_max_tokens) : "",
    recallChunksMaxTokens:
      trigger?.recall_chunks_max_tokens != null ? String(trigger.recall_chunks_max_tokens) : "",
    observationsMaxTokens:
      trigger?.reflect_search_observations_max_tokens != null
        ? String(trigger.reflect_search_observations_max_tokens)
        : "",
    observationsIncludeEntities:
      trigger?.reflect_search_observations_include_entities === true
        ? "true"
        : trigger?.reflect_search_observations_include_entities === false
          ? "false"
          : "",
    responseSchema: trigger?.response_schema
      ? JSON.stringify(trigger.response_schema, null, 2)
      : "",
    keepTrace: trigger?.keep_trace || false,
  };
}

/**
 * The trigger a form describes, or null when its tag-groups JSON does not parse.
 *
 * The schedule, fact types and exclude flag are sent explicitly (null/false)
 * rather than omitted: the server MERGES a trigger over the stored one (and over
 * the knowledge-page default), so an omitted field keeps its old value — which
 * made it impossible to switch a model off a cron, clear its fact types, or
 * untick "exclude mental models".
 */
export function triggerFromForm(form: TriggerForm): MentalModelTrigger | null {
  let tagGroups: TagGroup[] | undefined;
  if (form.tagGroups.trim()) {
    try {
      tagGroups = JSON.parse(form.tagGroups.trim());
    } catch {
      return null;
    }
  }
  const excludeIds = form.excludeMentalModelIds
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const toInt = (v: string) => (v.trim() ? parseInt(v, 10) : undefined);
  return {
    mode: form.mode,
    refresh_after_consolidation: form.refreshTrigger === "auto",
    refresh_cron: form.refreshTrigger === "scheduled" ? form.refreshCron.trim() || null : null,
    min_refresh_interval_seconds: toInt(form.minRefreshIntervalSeconds),
    fact_types: form.factTypes.length > 0 ? form.factTypes : null,
    exclude_mental_models: form.excludeMentalModels,
    exclude_mental_model_ids: excludeIds.length > 0 ? excludeIds : undefined,
    tags_match: (form.tagsMatch as TagsMatch) || undefined,
    tag_groups: tagGroups,
    include_chunks:
      form.includeChunks === "true" ? true : form.includeChunks === "false" ? false : undefined,
    recall_max_tokens: toInt(form.recallMaxTokens),
    recall_chunks_max_tokens: toInt(form.recallChunksMaxTokens),
    reflect_search_observations_max_tokens: toInt(form.observationsMaxTokens),
    reflect_search_observations_include_entities:
      form.observationsIncludeEntities === "true"
        ? true
        : form.observationsIncludeEntities === "false"
          ? false
          : undefined,
    // response_schema is only ever set through the schema builder, which
    // guarantees valid, usable JSON.
    response_schema: form.responseSchema.trim()
      ? JSON.parse(form.responseSchema.trim())
      : undefined,
    keep_trace: form.keepTrace,
  };
}

// Fields rarely touched; the Advanced panel starts open only when one is set,
// so a model configured there never hides its own settings.
function hasAdvancedValues(form: TriggerForm): boolean {
  return Boolean(
    form.excludeMentalModelIds.trim() ||
    form.tagGroups.trim() ||
    form.includeChunks ||
    form.recallMaxTokens.trim() ||
    form.recallChunksMaxTokens.trim() ||
    form.observationsMaxTokens.trim() ||
    form.observationsIncludeEntities ||
    form.responseSchema.trim() ||
    form.keepTrace
  );
}

const CRON_PRESETS = [
  { key: "triggerCronPresetHourly", cron: "0 * * * *" },
  { key: "triggerCronPresetDaily", cron: "0 3 * * *" },
  { key: "triggerCronPresetWeekly", cron: "0 3 * * 1" },
] as const;

// "0 * * * *" -> "Every hour"; the raw expression when cronstrue can't read it.
function describeCron(cron: string): string {
  try {
    return cronstrue.toString(cron, { throwExceptionOnParseError: true });
  } catch {
    return cron;
  }
}

/** One-line chips describing a trigger, e.g. for a settings row that opens the form. */
export function TriggerSummary({ form }: { form: TriggerForm }) {
  const t = useTranslations("mentalModels");
  const tFact = useTranslations("factTypeFilter");
  const chips = [
    form.refreshTrigger === "scheduled"
      ? describeCron(form.refreshCron)
      : form.refreshTrigger === "auto"
        ? t("optionsRefreshTriggerAuto")
        : t("optionsRefreshTriggerManual"),
    form.mode === "delta" ? t("triggerModeDelta") : t("triggerModeFull"),
    form.factTypes.length > 0
      ? form.factTypes.map((ft) => tFact(ft)).join(", ")
      : t("triggerFactTypesAll"),
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

/**
 * The trigger options of a mental model: when it refreshes, how it rewrites,
 * what it reads, plus an Advanced panel. Shared by the create/update dialogs and
 * the bank's knowledge-page default so the three can never drift apart.
 *
 * ``tagsField`` renders at the top of "What it reads": a model's own tags are
 * not part of the trigger, but belong next to how they match.
 */
export function MentalModelTriggerFields({
  value: form,
  onChange: setForm,
  tagsField,
}: {
  value: TriggerForm;
  onChange: (form: TriggerForm) => void;
  tagsField?: ReactNode;
}) {
  const t = useTranslations("mentalModels");
  const id = useId();
  const [advancedOpen, setAdvancedOpen] = useState(() => hasAdvancedValues(form));

  return (
    <div className="space-y-7">
      <Section title={t("triggerSectionWhen")}>
        <OptionCards
          value={form.refreshTrigger}
          onChange={(refreshTrigger) =>
            setForm({
              ...form,
              refreshTrigger,
              // Drop any cron when leaving the scheduled option.
              refreshCron: refreshTrigger === "scheduled" ? form.refreshCron : "",
            })
          }
          options={[
            {
              value: "auto",
              icon: Zap,
              label: t("optionsRefreshTriggerAuto"),
              description: t("triggerCardAuto"),
              hint: t("optionsRefreshTriggerAutoDesc"),
            },
            {
              value: "scheduled",
              icon: CalendarClock,
              label: t("triggerScheduled"),
              description: t("triggerCardScheduled"),
              hint: t("optionsRefreshTriggerScheduledDesc"),
            },
            {
              value: "manual",
              icon: Hand,
              label: t("optionsRefreshTriggerManual"),
              description: t("triggerCardManual"),
              hint: t("optionsRefreshTriggerManualDesc"),
            },
          ]}
        />
        {form.refreshTrigger === "scheduled" && (
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Input
                aria-label={t("optionsRefreshCronLabel")}
                value={form.refreshCron}
                onChange={(e) => setForm({ ...form, refreshCron: e.target.value })}
                placeholder={t("optionsRefreshCronPlaceholder")}
                className="h-8 flex-1 min-w-[12rem] font-mono text-sm"
              />
              {CRON_PRESETS.map((preset) => (
                <button
                  key={preset.cron}
                  type="button"
                  onClick={() => setForm({ ...form, refreshCron: preset.cron })}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-xs transition-colors",
                    form.refreshCron.trim() === preset.cron
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                  )}
                >
                  {t(preset.key)}
                </button>
              ))}
            </div>
            <CronSchedulePreview cron={form.refreshCron} compact />
          </div>
        )}
        {form.refreshTrigger !== "manual" && (
          <Row
            label={t("triggerMinIntervalLabel")}
            description={t("optionsMinRefreshIntervalDescription")}
            htmlFor={`${id}-min-interval`}
          >
            <div className="relative">
              <Input
                id={`${id}-min-interval`}
                type="number"
                min="0"
                value={form.minRefreshIntervalSeconds}
                onChange={(e) => setForm({ ...form, minRefreshIntervalSeconds: e.target.value })}
                placeholder={t("optionsMinRefreshIntervalPlaceholder")}
                className="h-8 pr-16"
              />
              <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-xs text-muted-foreground">
                {t("triggerSeconds")}
              </span>
            </div>
          </Row>
        )}
      </Section>

      <Section title={t("triggerSectionHow")}>
        <OptionCards
          value={form.mode}
          onChange={(mode) => setForm({ ...form, mode })}
          options={[
            {
              value: "delta",
              icon: FilePenLine,
              label: t("triggerModeDelta"),
              description: t("triggerCardDelta"),
              hint: t("triggerModeDeltaDesc"),
            },
            {
              value: "full",
              icon: FileText,
              label: t("triggerModeFull"),
              description: t("triggerCardFull"),
              hint: t("triggerModeFullDesc"),
            },
          ]}
        />
      </Section>

      <Section title={t("triggerSectionReads")}>
        {tagsField}
        <div className="space-y-2">
          <div className="flex items-center gap-1.5">
            <p className="text-sm font-medium text-foreground">{t("optionsFactTypesLabel")}</p>
            <Hint text={t("optionsFactTypesEmpty")} />
          </div>
          <FactTypeCheckboxGroup
            value={form.factTypes}
            onChange={(v) => setForm({ ...form, factTypes: v as FactType[] })}
          />
          {/* An empty selection is not "no types" — it means no filter, i.e. all
              types (it saves as fact_types: null). The "?" Hint says so, but empty
              checkboxes read as "none"; surface the effective value inline, matching
              the read-only summary/detail view, so the editor can't be misread. */}
          {form.factTypes.length === 0 && (
            <p className="text-xs text-muted-foreground italic">{t("triggerFactTypesAll")}</p>
          )}
        </div>
        <Row
          label={t("optionsExcludeAllLabel")}
          description={t("triggerExcludeAllDescription")}
          htmlFor={`${id}-exclude`}
        >
          <div className="flex sm:justify-end">
            <Switch
              id={`${id}-exclude`}
              checked={form.excludeMentalModels}
              onCheckedChange={(checked) => setForm({ ...form, excludeMentalModels: checked })}
            />
          </div>
        </Row>
        <Row label={t("optionsTagsMatchLabel")} description={t("optionsTagsMatchDescription")}>
          <Select
            value={form.tagsMatch || "default"}
            onValueChange={(v) => setForm({ ...form, tagsMatch: v === "default" ? "" : v })}
          >
            <SelectTrigger className="h-8">
              <SelectValue placeholder={t("optionsTagsMatchDefaultPlaceholder")} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">{t("optionsTagsMatchDefault")}</SelectItem>
              <SelectItem value="any">{t("optionsTagsMatchAny")}</SelectItem>
              <SelectItem value="all">{t("optionsTagsMatchAll")}</SelectItem>
              <SelectItem value="any_strict">{t("optionsTagsMatchAnyStrict")}</SelectItem>
              <SelectItem value="all_strict">{t("optionsTagsMatchAllStrict")}</SelectItem>
              <SelectItem value="exact">{t("optionsTagsMatchExact")}</SelectItem>
            </SelectContent>
          </Select>
        </Row>
      </Section>

      <section className="space-y-3">
        <DisclosureButton
          open={advancedOpen}
          onToggle={() => setAdvancedOpen((open) => !open)}
          label={t("triggerSectionAdvanced")}
        />
        {advancedOpen && (
          <div className="space-y-4">
            <Row
              label={t("optionsExcludeIdsLabel")}
              description={t("triggerHelpExcludeIds")}
              htmlFor={`${id}-exclude-ids`}
            >
              <Input
                id={`${id}-exclude-ids`}
                value={form.excludeMentalModelIds}
                onChange={(e) => setForm({ ...form, excludeMentalModelIds: e.target.value })}
                placeholder={t("optionsExcludeIdsPlaceholder")}
                className="h-8"
              />
            </Row>
            <Row label={t("optionsIncludeChunksLabel")} description={t("triggerHelpIncludeChunks")}>
              <Select
                value={form.includeChunks || "default"}
                onValueChange={(v) =>
                  setForm({
                    ...form,
                    includeChunks: v === "default" ? "" : (v as "true" | "false"),
                  })
                }
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">{t("optionsIncludeChunksDefault")}</SelectItem>
                  <SelectItem value="true">{t("optionsIncludeChunksYes")}</SelectItem>
                  <SelectItem value="false">{t("optionsIncludeChunksNo")}</SelectItem>
                </SelectContent>
              </Select>
            </Row>
            <Row
              label={t("optionsRecallMaxTokensLabel")}
              description={t("optionsRecallMaxTokensDescription")}
              htmlFor={`${id}-recall-max`}
            >
              <Input
                id={`${id}-recall-max`}
                type="number"
                min="0"
                value={form.recallMaxTokens}
                onChange={(e) => setForm({ ...form, recallMaxTokens: e.target.value })}
                placeholder={t("optionsRecallMaxTokensPlaceholder")}
                className="h-8"
              />
            </Row>
            <Row
              label={t("optionsRecallChunksMaxTokensLabel")}
              description={t("optionsRecallChunksMaxTokensDescription")}
              htmlFor={`${id}-chunks-max`}
            >
              <Input
                id={`${id}-chunks-max`}
                type="number"
                min="0"
                value={form.recallChunksMaxTokens}
                onChange={(e) => setForm({ ...form, recallChunksMaxTokens: e.target.value })}
                placeholder={t("optionsRecallChunksMaxTokensPlaceholder")}
                className="h-8"
              />
            </Row>
            <Row
              label={t("optionsObservationsMaxTokensLabel")}
              description={t("optionsObservationsMaxTokensDescription")}
              htmlFor={`${id}-observations-max`}
            >
              <Input
                id={`${id}-observations-max`}
                type="number"
                min="1"
                value={form.observationsMaxTokens}
                onChange={(e) => setForm({ ...form, observationsMaxTokens: e.target.value })}
                placeholder={t("optionsRecallMaxTokensPlaceholder")}
                className="h-8"
              />
            </Row>
            <Row
              label={t("optionsObservationsEntitiesLabel")}
              description={t("optionsObservationsEntitiesDescription")}
            >
              <Select
                value={form.observationsIncludeEntities || "default"}
                onValueChange={(v) =>
                  setForm({
                    ...form,
                    observationsIncludeEntities: v === "default" ? "" : (v as "true" | "false"),
                  })
                }
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">{t("optionsIncludeChunksDefault")}</SelectItem>
                  <SelectItem value="true">{t("optionsObservationsEntitiesYes")}</SelectItem>
                  <SelectItem value="false">{t("optionsObservationsEntitiesNo")}</SelectItem>
                </SelectContent>
              </Select>
            </Row>
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <label htmlFor={`${id}-tag-groups`} className="text-sm font-medium text-foreground">
                  {t("optionsTagGroupsLabel")}
                </label>
                <Hint text={t("optionsTagGroupsDescription")} />
              </div>
              <Textarea
                id={`${id}-tag-groups`}
                value={form.tagGroups}
                onChange={(e) => setForm({ ...form, tagGroups: e.target.value })}
                placeholder='e.g., [{"or": [{"tags": ["user:alice"], "match": "all_strict"}, {"tags": ["shared"]}]}]'
                rows={3}
                className="font-mono text-xs"
              />
            </div>
            <ResponseSchemaField
              value={form.responseSchema}
              onChange={(json) => setForm({ ...form, responseSchema: json })}
            />
            <Row
              label={t("optionsKeepTraceLabel")}
              description={t("optionsKeepTraceDescription")}
              htmlFor={`${id}-keep-trace`}
            >
              <div className="flex sm:justify-end">
                <Switch
                  id={`${id}-keep-trace`}
                  checked={form.keepTrace}
                  onCheckedChange={(checked) => setForm({ ...form, keepTrace: checked })}
                />
              </div>
            </Row>
          </div>
        )}
      </section>
    </div>
  );
}
