"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Check, X } from "lucide-react";

export interface EditMemoryFields {
  text: string;
  context: string;
  factType: "world" | "experience";
  occurredStart: string; // YYYY-MM-DD or "" to clear
  occurredEnd: string;
  entities: string[];
}

interface EditMemoryFormProps {
  memory: {
    text: string;
    context?: string | null;
    type?: string;
    fact_type?: string;
    occurred_start?: string | null;
    occurred_end?: string | null;
    // Graph-node memories carry entities as a ", "-joined string; the detail
    // endpoint returns an array. Accept both and normalize.
    entities?: string[] | string | null;
  };
  busy?: boolean;
  onCancel: () => void;
  onSave: (fields: EditMemoryFields) => void;
}

// ISO timestamp -> YYYY-MM-DD for <input type="date">; "" when absent.
function toDateInput(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toISOString().slice(0, 10);
}

// Form to correct what the LLM extracted: text, type, occurred dates, context,
// and entities. Entities are edited as removable chips; clearing them all
// detaches every entity from the fact.
export function EditMemoryForm({ memory, busy, onCancel, onSave }: EditMemoryFormProps) {
  const t = useTranslations("memoryDetailPanel");
  const [text, setText] = useState(memory.text ?? "");
  const [context, setContext] = useState(memory.context ?? "");
  const [factType, setFactType] = useState<"world" | "experience">(
    (memory.fact_type ?? memory.type) === "experience" ? "experience" : "world"
  );
  const [occurredStart, setOccurredStart] = useState(toDateInput(memory.occurred_start));
  const [occurredEnd, setOccurredEnd] = useState(toDateInput(memory.occurred_end));
  const [entities, setEntities] = useState<string[]>(
    Array.isArray(memory.entities)
      ? memory.entities
      : memory.entities
        ? String(memory.entities).split(", ").filter(Boolean)
        : []
  );
  const [entityDraft, setEntityDraft] = useState("");

  const addEntity = (raw: string) => {
    const name = raw.trim();
    if (!name) return;
    // De-dup case-insensitively, mirroring the server-side normalization.
    if (!entities.some((e) => e.toLowerCase() === name.toLowerCase())) {
      setEntities([...entities, name]);
    }
    setEntityDraft("");
  };

  return (
    <div className="space-y-3">
      <div>
        <label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          {t("sectionFullText")}
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="mt-1 w-full min-h-[100px] p-3 bg-muted/50 rounded-lg border border-border text-sm leading-relaxed text-card-foreground resize-y"
          autoFocus
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            {t("editFieldType")}
          </label>
          <div className="mt-1 flex items-center gap-1 bg-muted rounded-lg p-1 w-fit">
            {(["world", "experience"] as const).map((ft) => (
              <button
                key={ft}
                type="button"
                onClick={() => setFactType(ft)}
                className={`px-3 py-1 rounded-md text-sm font-medium capitalize transition-all ${
                  factType === ft
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {ft}
              </button>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              {t("editFieldOccurredStart")}
            </label>
            <Input
              type="date"
              value={occurredStart}
              onChange={(e) => setOccurredStart(e.target.value)}
              className="mt-1 h-8 text-sm"
            />
          </div>
          <div>
            <label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              {t("editFieldOccurredEnd")}
            </label>
            <Input
              type="date"
              value={occurredEnd}
              onChange={(e) => setOccurredEnd(e.target.value)}
              className="mt-1 h-8 text-sm"
            />
          </div>
        </div>
      </div>

      <div>
        <label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          {t("editFieldContext")}
        </label>
        <Input
          value={context}
          onChange={(e) => setContext(e.target.value)}
          className="mt-1 h-8 text-sm"
        />
      </div>

      <div>
        <label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          {t("editFieldEntities")}
        </label>
        <div className="mt-1 flex flex-wrap items-center gap-1.5 p-2 bg-muted/50 rounded-lg border border-border min-h-[2.5rem]">
          {entities.map((entity) => (
            <span
              key={entity}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-background border border-border text-xs"
            >
              {entity}
              <button
                type="button"
                onClick={() => setEntities(entities.filter((e) => e !== entity))}
                className="text-muted-foreground hover:text-foreground"
                aria-label={t("editEntityRemove", { entity })}
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
          <input
            value={entityDraft}
            onChange={(e) => setEntityDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === ",") {
                e.preventDefault();
                addEntity(entityDraft);
              } else if (e.key === "Backspace" && !entityDraft && entities.length) {
                setEntities(entities.slice(0, -1));
              }
            }}
            onBlur={() => addEntity(entityDraft)}
            placeholder={t("editEntityPlaceholder")}
            className="flex-1 min-w-[8rem] bg-transparent text-sm outline-none"
          />
        </div>
      </div>

      <div className="flex items-center justify-end gap-2">
        <Button variant="ghost" size="sm" disabled={busy} onClick={onCancel}>
          {t("curationCancel")}
        </Button>
        <Button
          size="sm"
          disabled={busy || !text.trim()}
          onClick={() => {
            // Fold any unsubmitted draft into the list before saving.
            const finalEntities = entityDraft.trim()
              ? entities.some((e) => e.toLowerCase() === entityDraft.trim().toLowerCase())
                ? entities
                : [...entities, entityDraft.trim()]
              : entities;
            onSave({
              text: text.trim(),
              context,
              factType,
              occurredStart,
              occurredEnd,
              entities: finalEntities,
            });
          }}
        >
          {busy ? (
            <span className="animate-spin mr-1.5">⏳</span>
          ) : (
            <Check className="h-3.5 w-3.5 mr-1.5" />
          )}
          {t("curationSave")}
        </Button>
      </div>
    </div>
  );
}
