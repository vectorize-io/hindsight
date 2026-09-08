"use client";

import { useState } from "react";

import { ChevronDown, ChevronRight, Plus, Trash2 } from "lucide-react";
import { useTranslations } from "next-intl";

import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/**
 * The controlled-vocabulary editor for `entity_labels`, lifted out of
 * bank-config-view so the Extraction Lab can offer it too — its block is the one
 * setting whose value has structure, and sending the reader off to another screen to
 * change it broke the loop the lab exists for.
 */
export type LabelValue = { value: string; description: string };

export type MapField = {
  type: "text" | "multi-text" | "value" | "multi-values" | "map";
  description: string;
  values?: LabelValue[];
  fields?: Record<string, MapField>;
};

export type LabelGroup = {
  key: string;
  description: string;
  type: "value" | "multi-values" | "text" | "multi-text" | "map";
  optional: boolean;
  tag: boolean;
  values: LabelValue[];
  fields: Record<string, MapField>;
};

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
  if (attr.type === "multi-text") return `e.g. ${key}:<any text>, ${key}:<any text>`;
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
    "multi-text": t("fieldTypeMultiText"),
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
                    ...(v === "text" || v === "multi-text"
                      ? { fields: undefined, values: undefined }
                      : {}),
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

export function EntityLabelsEditor({
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
    <div className="px-6 py-4 space-y-3" data-config-field="entity_labels">
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
