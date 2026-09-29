"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Plus, Trash2 } from "lucide-react";
import {
  SchemaField,
  SchemaFieldType,
  SchemaNode,
  SCHEMA_FIELD_TYPES,
  validateResponseSchema,
  fieldsToSchema,
  schemaToFields,
  emptyField,
  nodeForType,
} from "@/lib/response-schema";

type Mode = "visual" | "code";

/** Type dropdown shared by fields and array items. */
function TypeSelect({
  value,
  onChange,
}: {
  value: SchemaFieldType;
  onChange: (t: SchemaFieldType) => void;
}) {
  return (
    <Select value={value} onValueChange={(v) => onChange(v as SchemaFieldType)}>
      <SelectTrigger className="w-28 h-8">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {SCHEMA_FIELD_TYPES.map((ty) => (
          <SelectItem key={ty} value={ty}>
            {ty}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/** Renders the nested part of a node: object → child fields, array → item node. */
function NodeChildren({
  node,
  onChange,
}: {
  node: SchemaNode;
  onChange: (node: SchemaNode) => void;
}) {
  const t = useTranslations("schemaBuilder");
  if (node.type === "object") {
    return (
      <div className="ml-2 pl-3 border-l-2 border-border/70">
        <FieldRows
          fields={node.fields ?? []}
          onChange={(fields) => onChange({ ...node, fields })}
        />
      </div>
    );
  }
  if (node.type === "array") {
    const items = node.items ?? { type: "string" };
    return (
      <div className="ml-2 pl-3 border-l-2 border-border/70 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">{t("itemsLabel")}</span>
          <TypeSelect
            value={items.type}
            onChange={(ty) => onChange({ ...node, items: nodeForType(ty) })}
          />
        </div>
        <NodeChildren
          node={items}
          onChange={(itemNode) => onChange({ ...node, items: itemNode })}
        />
      </div>
    );
  }
  return null;
}

/** A recursive editable list of fields (top level, or the properties of an object). */
function FieldRows({
  fields,
  onChange,
}: {
  fields: SchemaField[];
  onChange: (fields: SchemaField[]) => void;
}) {
  const t = useTranslations("schemaBuilder");
  const update = (i: number, patch: Partial<SchemaField>) =>
    onChange(fields.map((f, idx) => (idx === i ? { ...f, ...patch } : f)));
  const remove = (i: number) => onChange(fields.filter((_, idx) => idx !== i));
  const add = () => onChange([...fields, emptyField()]);

  return (
    <div className="space-y-2">
      {fields.length === 0 && (
        <p className="text-xs text-muted-foreground py-1">{t("emptyState")}</p>
      )}
      {fields.map((field, i) => (
        <div key={i} className="rounded-lg border border-border p-2.5 space-y-2">
          <div className="flex items-center gap-2">
            <Input
              value={field.name}
              onChange={(e) => update(i, { name: e.target.value })}
              placeholder={t("namePlaceholder")}
              className="flex-1 h-8"
            />
            <TypeSelect
              value={field.node.type}
              onChange={(ty) => update(i, { node: nodeForType(ty) })}
            />
            <label className="flex items-center gap-1.5 text-xs cursor-pointer whitespace-nowrap">
              <Checkbox
                checked={field.required}
                onCheckedChange={(c) => update(i, { required: c === true })}
              />
              {t("required")}
            </label>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              onClick={() => remove(i)}
              aria-label={t("removeField")}
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
          <Input
            value={field.description}
            onChange={(e) => update(i, { description: e.target.value })}
            placeholder={t("descriptionPlaceholder")}
            className="h-8 text-xs"
          />
          <NodeChildren node={field.node} onChange={(node) => update(i, { node })} />
        </div>
      ))}
      <Button variant="outline" size="sm" onClick={add}>
        <Plus className="w-4 h-4 mr-1" />
        {t("addField")}
      </Button>
    </div>
  );
}

/**
 * No-code editor for a structured-output `response_schema`. Visual mode edits a
 * recursive field tree (objects nest fields, arrays nest an item type); Code mode
 * edits the raw JSON. The two stay in sync on tab switch, and Apply is blocked
 * until the current representation is a usable schema (same contract as the
 * backend's validate_response_schema()).
 */
export function SchemaBuilderDialog({
  open,
  value,
  onClose,
  onSave,
}: {
  open: boolean;
  value: string;
  onClose: () => void;
  onSave: (json: string) => void;
}) {
  const t = useTranslations("schemaBuilder");
  const [mode, setMode] = useState<Mode>("visual");
  const [fields, setFields] = useState<SchemaField[]>([emptyField()]);
  const [code, setCode] = useState("");
  // Set only when a Code→Visual switch fails (invalid or too-complex JSON); cleared on any edit.
  const [switchError, setSwitchError] = useState<string | null>(null);

  // Seed the editor from the incoming value each time the dialog opens.
  useEffect(() => {
    if (!open) return;
    setSwitchError(null);
    const text = value.trim();
    if (!text) {
      setMode("visual");
      setFields([emptyField()]);
      setCode("");
      return;
    }
    try {
      const parsed = JSON.parse(text);
      const asFields =
        typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
          ? schemaToFields(parsed as Record<string, unknown>)
          : null;
      if (asFields !== null) {
        setFields(asFields.length > 0 ? asFields : [emptyField()]);
        setMode("visual");
      } else {
        setMode("code");
      }
      setCode(JSON.stringify(parsed, null, 2));
    } catch {
      // Not valid JSON — let the user fix it in code mode.
      setMode("code");
      setCode(text);
    }
  }, [open, value]);

  // Validate the current representation; returns a message or null.
  const visualError = (() => {
    const named = fields.filter((f) => f.name.trim());
    if (named.length === 0) return t("emptyState");
    const names = named.map((f) => f.name.trim());
    if (new Set(names).size !== names.length) return t("duplicateNames");
    return validateResponseSchema(fieldsToSchema(fields));
  })();

  const codeError = (() => {
    const text = code.trim();
    if (!text) return t("emptyState");
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return t("invalidJson");
    }
    return validateResponseSchema(parsed);
  })();

  const currentError = mode === "visual" ? visualError : codeError;

  const switchTo = (next: Mode) => {
    if (next === mode) return;
    setSwitchError(null);
    if (next === "code") {
      setCode(JSON.stringify(fieldsToSchema(fields), null, 2));
      setMode("code");
      return;
    }
    // Code → Visual: only allowed when the JSON maps cleanly to the field tree.
    let parsed: unknown;
    try {
      parsed = JSON.parse(code);
    } catch {
      setSwitchError(t("invalidJson"));
      return;
    }
    const asFields =
      typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
        ? schemaToFields(parsed as Record<string, unknown>)
        : null;
    if (asFields === null) {
      setSwitchError(t("visualUnavailable"));
      return;
    }
    // An empty schema is representable — start the visual editor with a blank field.
    setFields(asFields.length > 0 ? asFields : [emptyField()]);
    setMode("visual");
  };

  const handleApply = () => {
    if (mode === "visual") {
      if (visualError) return;
      onSave(JSON.stringify(fieldsToSchema(fields), null, 2));
    } else {
      if (codeError) return;
      onSave(JSON.stringify(JSON.parse(code), null, 2));
    }
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{t("title")}</DialogTitle>
          <DialogDescription>{t("subtitle")}</DialogDescription>
        </DialogHeader>

        <Tabs
          value={mode}
          onValueChange={(v) => switchTo(v as Mode)}
          className="flex-1 flex flex-col min-h-0 overflow-hidden"
        >
          <TabsList className="w-full">
            <TabsTrigger value="visual" className="flex-1">
              {t("visualTab")}
            </TabsTrigger>
            <TabsTrigger value="code" className="flex-1">
              {t("codeTab")}
            </TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto mt-3 px-0.5">
            <TabsContent value="visual">
              <FieldRows fields={fields} onChange={setFields} />
            </TabsContent>

            <TabsContent value="code">
              <Textarea
                value={code}
                onChange={(e) => {
                  setCode(e.target.value);
                  setSwitchError(null);
                }}
                rows={16}
                className="font-mono text-xs"
                placeholder='{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}'
              />
            </TabsContent>
          </div>
        </Tabs>

        {(switchError || currentError) && (
          <p className="text-xs text-red-600 dark:text-red-400">{switchError || currentError}</p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            {t("cancel")}
          </Button>
          <Button onClick={handleApply} disabled={!!currentError}>
            {t("apply")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
