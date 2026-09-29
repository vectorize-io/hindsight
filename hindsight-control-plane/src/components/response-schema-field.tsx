"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";
import { Wand2, Pencil, X } from "lucide-react";
import { SchemaBuilderDialog } from "./schema-builder-dialog";
import { validateResponseSchema } from "@/lib/response-schema";

/**
 * The single control for a structured-output `response_schema`, shared by the
 * reflect view and the mental-model dialogs. It shows only *whether* a schema is
 * set (with a field summary) — all editing happens in the SchemaBuilderDialog, so
 * there is no raw-JSON textarea on the page. `value` is the JSON string; an empty
 * string means "no schema".
 */
export function ResponseSchemaField({
  value,
  onChange,
  className,
}: {
  value: string;
  onChange: (json: string) => void;
  className?: string;
}) {
  const t = useTranslations("schemaBuilder");
  const [open, setOpen] = useState(false);

  const info = useMemo(() => {
    const text = value.trim();
    if (!text) return { state: "empty" as const };
    try {
      const parsed = JSON.parse(text);
      if (validateResponseSchema(parsed) !== null) return { state: "invalid" as const };
      const props = (parsed as { properties?: Record<string, unknown> }).properties ?? {};
      return { state: "set" as const, names: Object.keys(props) };
    } catch {
      return { state: "invalid" as const };
    }
  }, [value]);

  return (
    <div className={className}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-muted-foreground">{t("title")}</span>
        {info.state === "empty" ? (
          <Button variant="outline" size="sm" className="h-7" onClick={() => setOpen(true)}>
            <Wand2 className="w-3.5 h-3.5 mr-1" />
            {t("openBuilder")}
          </Button>
        ) : (
          <div className="flex items-center gap-1">
            <Button variant="outline" size="sm" className="h-7" onClick={() => setOpen(true)}>
              <Pencil className="w-3.5 h-3.5 mr-1" />
              {t("edit")}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => onChange("")}
              aria-label={t("clear")}
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        )}
      </div>

      {info.state === "set" && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">
            {t("fieldsSummary", { count: info.names.length })}
          </span>
          {info.names.slice(0, 8).map((name) => (
            <span
              key={name}
              className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
            >
              {name}
            </span>
          ))}
        </div>
      )}

      {info.state === "invalid" && (
        <p className="mt-2 text-xs text-red-600 dark:text-red-400">{t("invalidSchema")}</p>
      )}

      {info.state === "empty" && (
        <p className="mt-2 text-xs text-muted-foreground">{t("fieldDescription")}</p>
      )}

      <SchemaBuilderDialog
        open={open}
        value={value}
        onClose={() => setOpen(false)}
        onSave={onChange}
      />
    </div>
  );
}
