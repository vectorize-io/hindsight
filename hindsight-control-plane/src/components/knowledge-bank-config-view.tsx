"use client";

// A knowledge bank's own ingest settings, edited like a memory bank's configuration:
// the same sections, rows and per-section Save. An empty field (or "Server default")
// follows the server; saving it sends null, which clears that one override. Only the
// per-bank settings the server lets a bank change are here — never models or providers.

import * as React from "react";
import { useTranslations } from "next-intl";

import { client } from "@/lib/api";
import {
  MEDIA_TYPES,
  type KnowledgeBankConfig,
  type KnowledgeBankConfigUpdates,
  type KnowledgeIngestSettings,
  type KnowledgeSettingName,
  type MediaType,
} from "@/lib/knowledge-banks";
import { ConfigSection, FieldRow } from "@/components/bank-config-view";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Spinner } from "@/components/ui/spinner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type NumberSetting = Exclude<KnowledgeSettingName, "pdf_bands" | "disabled_media_types">;

/** The server's bounds, so the browser's number field steps inside them. The server still
 *  checks, and also that a limit is not above its own and the overlap is under both windows. */
const NUMBER_BOUNDS: Record<NumberSetting, { min: number; max?: number }> = {
  text_passage_tokens: { min: 16, max: 8192 },
  pdf_page_pixels: { min: 256, max: 8192 },
  max_pdf_pages: { min: 1 },
  image_tokens: { min: 70, max: 1120 },
  audio_window_seconds: { min: 1, max: 3600 },
  video_window_seconds: { min: 1, max: 3600 },
  media_overlap_seconds: { min: 0, max: 3599 },
  max_media_seconds: { min: 1 },
};

type SectionId = "text" | "pdf" | "images" | "media" | "types";

const SECTIONS: { id: SectionId; fields: KnowledgeSettingName[] }[] = [
  { id: "text", fields: ["text_passage_tokens"] },
  { id: "pdf", fields: ["pdf_page_pixels", "pdf_bands", "max_pdf_pages"] },
  { id: "images", fields: ["image_tokens"] },
  {
    id: "media",
    fields: [
      "audio_window_seconds",
      "video_window_seconds",
      "media_overlap_seconds",
      "max_media_seconds",
    ],
  },
  { id: "types", fields: ["disabled_media_types"] },
];

/** What each field is set to in the form; null follows the server. */
type Edits = { [K in KnowledgeSettingName]: KnowledgeIngestSettings[K] | null };

function editsOf(overrides: KnowledgeBankConfig["overrides"]): Edits {
  return {
    text_passage_tokens: overrides.text_passage_tokens ?? null,
    pdf_page_pixels: overrides.pdf_page_pixels ?? null,
    pdf_bands: overrides.pdf_bands ?? null,
    image_tokens: overrides.image_tokens ?? null,
    audio_window_seconds: overrides.audio_window_seconds ?? null,
    video_window_seconds: overrides.video_window_seconds ?? null,
    media_overlap_seconds: overrides.media_overlap_seconds ?? null,
    max_media_seconds: overrides.max_media_seconds ?? null,
    max_pdf_pages: overrides.max_pdf_pages ?? null,
    disabled_media_types: overrides.disabled_media_types ?? null,
  };
}

/** Equal settings; a media type list is compared as a set (the server sorts it). */
function same(a: unknown, b: unknown): boolean {
  const norm = (v: unknown) => (Array.isArray(v) ? [...v].sort() : v);
  return JSON.stringify(norm(a)) === JSON.stringify(norm(b));
}

const INHERIT = "__inherit__";

type SectionState = { saving: boolean; error: string | null };

export function KnowledgeBankConfigView({ bankId }: { bankId: string }) {
  const t = useTranslations("knowledgeBanks.config");
  const tKinds = useTranslations("knowledgeBanks.kinds");
  const [data, setData] = React.useState<KnowledgeBankConfig | null>(null);
  const [loadError, setLoadError] = React.useState<string | null>(null);
  const [edits, setEdits] = React.useState<Edits | null>(null);
  const [sections, setSections] = React.useState<Partial<Record<SectionId, SectionState>>>({});

  React.useEffect(() => {
    let live = true;
    client
      .getKnowledgeBankConfig(bankId)
      .then((res) => {
        if (!live) return;
        setData(res);
        setEdits(editsOf(res.overrides));
      })
      .catch((e) => live && setLoadError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [bankId]);

  if (loadError) {
    return (
      <p className="text-sm text-destructive">
        {t("loadFailed")} <span className="text-muted-foreground">{loadError}</span>
      </p>
    );
  }
  if (!data || !edits) {
    return (
      <div className="flex justify-center py-12">
        <Spinner size="lg" variant="jump" />
      </div>
    );
  }

  const saved = editsOf(data.overrides);
  const set = <K extends KnowledgeSettingName>(field: K, value: Edits[K]) =>
    setEdits((prev) => prev && { ...prev, [field]: value });
  const changed = (fields: KnowledgeSettingName[]) =>
    fields.filter((f) => !same(edits[f], saved[f]));

  const save = async (id: SectionId, fields: KnowledgeSettingName[]) => {
    // Only what changed, so saving one section never rewrites another's overrides.
    const updates = Object.fromEntries(
      changed(fields).map((f) => [f, edits[f]])
    ) as KnowledgeBankConfigUpdates;
    setSections((prev) => ({ ...prev, [id]: { saving: true, error: null } }));
    try {
      const res = await client.updateKnowledgeBankConfig(bankId, updates);
      const next = editsOf(res.overrides);
      setData(res);
      // Other sections keep their unsaved edits.
      setEdits(
        (prev) => prev && { ...prev, ...Object.fromEntries(fields.map((f) => [f, next[f]])) }
      );
      setSections((prev) => ({ ...prev, [id]: { saving: false, error: null } }));
    } catch (e) {
      // The server names the bad field and why (out of range, above its own limit, ...).
      const error = e instanceof Error ? e.message : t("saveFailed");
      setSections((prev) => ({ ...prev, [id]: { saving: false, error } }));
    }
  };

  /** "Set for this bank · Use server default", under a field the bank overrides. */
  const overrideNote = (field: KnowledgeSettingName) =>
    field in data.overrides &&
    edits[field] !== null && (
      <p className="mt-1.5 flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
        <span className="rounded bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
          {t("overridden")}
        </span>
        <button
          type="button"
          className="text-primary hover:underline"
          onClick={() => set(field, null)}
        >
          {t("useServerDefault")}
        </button>
      </p>
    );

  const numberRow = (field: NumberSetting) => (
    <FieldRow
      key={field}
      label={t(`fields.${field}.label`)}
      description={t(`fields.${field}.description`)}
    >
      <Input
        type="number"
        inputMode="numeric"
        aria-label={t(`fields.${field}.label`)}
        min={NUMBER_BOUNDS[field].min}
        max={NUMBER_BOUNDS[field].max}
        value={edits[field] ?? ""}
        onChange={(e) => set(field, e.target.value === "" ? null : Number(e.target.value))}
        // Without an override the resolved value is the server's; with one it is not sent.
        placeholder={
          field in data.overrides
            ? t("serverDefault")
            : t("serverDefaultValue", { value: data.config[field] })
        }
      />
      {overrideNote(field)}
    </FieldRow>
  );

  const bandsRow = () => (
    <FieldRow
      key="pdf_bands"
      label={t("fields.pdf_bands.label")}
      description={t("fields.pdf_bands.description")}
    >
      {/* Tri-state, as on memory banks: follow the server, or turn it on or off for this bank. */}
      <Select
        value={edits.pdf_bands === null ? INHERIT : String(edits.pdf_bands)}
        onValueChange={(v) => set("pdf_bands", v === INHERIT ? null : v === "true")}
      >
        <SelectTrigger aria-label={t("fields.pdf_bands.label")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={INHERIT}>
            {"pdf_bands" in data.overrides
              ? t("serverDefault")
              : t("serverDefaultValue", { value: data.config.pdf_bands ? t("on") : t("off") })}
          </SelectItem>
          <SelectItem value="true">{t("on")}</SelectItem>
          <SelectItem value="false">{t("off")}</SelectItem>
        </SelectContent>
      </Select>
    </FieldRow>
  );

  // A bank may only add to the server's refused types, so those show off and locked. They are
  // what the bank refuses beyond its own list (the resolved list is the two together).
  const ownOff = edits.disabled_media_types ?? [];
  const serverOff = data.config.disabled_media_types.filter(
    (m) => !(data.overrides.disabled_media_types ?? []).includes(m)
  );
  const toggleType = (kind: MediaType, accept: boolean) => {
    const next = MEDIA_TYPES.filter((m) => (m === kind ? !accept : ownOff.includes(m)));
    // Refusing nothing beyond the server's is the server's default: clear the override.
    set("disabled_media_types", next.length ? next : null);
  };
  const typeRows = () => (
    <React.Fragment key="disabled_media_types">
      {MEDIA_TYPES.map((kind) => {
        const locked = serverOff.includes(kind);
        return (
          <FieldRow
            key={kind}
            label={tKinds(kind)}
            description={locked ? t("serverOff") : undefined}
          >
            <div className="flex md:justify-end">
              <Switch
                aria-label={t("accept", { kind: tKinds(kind) })}
                checked={!locked && !ownOff.includes(kind)}
                disabled={locked}
                onCheckedChange={(accept) => toggleType(kind, accept)}
              />
            </div>
          </FieldRow>
        );
      })}
      {overrideNote("disabled_media_types") && (
        <div className="px-6 py-3">{overrideNote("disabled_media_types")}</div>
      )}
    </React.Fragment>
  );

  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">{t("intro")}</p>
      {SECTIONS.map(({ id, fields }) => (
        <ConfigSection
          key={id}
          title={t(`sections.${id}.title`)}
          description={t(`sections.${id}.description`)}
          error={sections[id]?.error ?? null}
          dirty={changed(fields).length > 0}
          saving={sections[id]?.saving ?? false}
          onSave={() => save(id, fields)}
        >
          {fields.map((field) =>
            field === "pdf_bands"
              ? bandsRow()
              : field === "disabled_media_types"
                ? typeRows()
                : numberRow(field)
          )}
        </ConfigSection>
      ))}
    </div>
  );
}
