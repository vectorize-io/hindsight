"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslations } from "next-intl";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { cn } from "@/lib/utils";

type PromptPreviewOperation = "retain" | "observations" | "reflect";

type BlockSource = "config" | "builtin";
type InputKind = "text" | "boolean" | "choice" | "complex";
type Block = {
  label: string;
  text: string;
  source: BlockSource;
  field: string;
  active: boolean;
  note?: string | null;
  value?: string | null;
  kind: InputKind;
  choices?: string[] | null;
  editable: boolean;
};
type Message = { role: "system" | "user"; blocks: Block[] };
type Preview = {
  operation: string;
  messages: Message[];
  response_schema?: Record<string, unknown> | null;
  skipped_reason?: string | null;
};

/**
 * Colour is the point of this screen: a reader should tell at a glance which blocks
 * come from a setting they can change and which are Hindsight's own.
 *
 * Hue is keyed on the *config field*, not on position, so a setting keeps its colour
 * across operations and across re-renders — the mission is always amber, the language
 * rule always violet. Built-in text is colourless and runtime stand-ins are dashed:
 * neither is something this screen can change, and a hue of their own would say
 * otherwise.
 */
const FIELD_BAR: Record<string, string> = {
  retain_mission: "bg-amber-400",
  observations_mission: "bg-amber-400",
  reflect_mission: "bg-amber-400",
  llm_output_language: "bg-violet-400",
  entity_labels: "bg-sky-400",
  retain_custom_instructions: "bg-emerald-400",
  retain_extract_causal_links: "bg-rose-400",
  retain_extraction_mode: "bg-blue-400",
};

function barFor(block: Block): string {
  if (!block.active) return "bg-transparent";
  if (block.source === "builtin" && !block.field) return "bg-border";
  return FIELD_BAR[block.field] ?? "bg-border";
}

/**
 * First line of `text` with actual words in it.
 *
 * The prompts separate their sections with long rules of `═`, so a naive
 * first-N-characters preview of a collapsed block is often a row of box-drawing
 * characters that says nothing about what the block contains.
 */
function firstMeaningfulLine(text: string): string {
  const line = text.split("\n").find((candidate) => /[\p{L}\p{N}]/u.test(candidate));
  return (line ?? text).trim();
}

/**
 * The control for the setting a block belongs to, shown on the block's own header row.
 *
 * Saving PATCHes only this one field, so other unsaved edits in the surrounding
 * Configuration form are left alone — but that form has to be told, or it would keep
 * showing (and later re-save) the value this replaced. That is what `onSaved` is for.
 */
function BlockControl({
  bankId,
  block,
  onSaved,
  disabledReason,
  onClose,
}: {
  bankId: string;
  block: Block;
  onSaved: (field: string, value: string | null) => void;
  disabledReason?: string;
  /** Closes the dialog, for a setting whose editor lives on the page behind it. */
  onClose: () => void;
}) {
  const t = useTranslations("bankConfig");
  const [saving, setSaving] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(block.value ?? "");
  const [error, setError] = useState<string | null>(null);

  // The preview refetches after every save, so the block's value is the source of
  // truth; re-seed the draft whenever it changes underneath us.
  useEffect(() => setDraft(block.value ?? ""), [block.value]);

  async function save(next: string | null) {
    setSaving(true);
    setError(null);
    try {
      await client.updateBankConfig(bankId, {
        // An empty box means "unset" — a null override, so the bank falls back to the
        // server default, exactly as clearing the field in the Configuration form does.
        [block.field]: block.kind === "boolean" ? next === "true" : next === "" ? null : next,
      });
      setEditing(false);
      onSaved(block.field, next === "" ? null : next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  if (saving) return <Spinner size="sm" />;

  // Entity labels are groups of key/value pairs with their own editor on the page
  // behind this dialog. "Edit in the section above" was useless while the dialog
  // covered that section — take the reader there instead.
  if (block.kind === "complex" && !disabledReason) {
    return (
      <Button
        variant="outline"
        size="sm"
        className="h-7 text-xs"
        onClick={() => {
          onClose();
          document
            .querySelector(`[data-config-field="${block.field}"]`)
            ?.scrollIntoView({ behavior: "smooth", block: "center" });
        }}
      >
        {t("promptPreviewGoToSetting")}
      </Button>
    );
  }

  if (!block.editable || disabledReason) {
    return (
      <span className="text-[11px] italic text-muted-foreground">
        {disabledReason ?? t("promptPreviewServerLevel")}
      </span>
    );
  }

  if (block.kind === "boolean") {
    return (
      <Switch
        checked={block.value === "true"}
        onCheckedChange={(checked) => void save(checked ? "true" : "false")}
      />
    );
  }

  if (block.kind === "choice") {
    return (
      <select
        className="h-7 rounded-md border border-border bg-background px-2 text-xs"
        value={block.value ?? ""}
        onChange={(e) => void save(e.target.value)}
      >
        {(block.choices ?? []).map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    );
  }

  if (!editing) {
    return (
      <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => setEditing(true)}>
        {block.active ? t("promptPreviewEdit") : t("promptPreviewWrite")}
      </Button>
    );
  }

  return (
    <div className="w-64 space-y-1">
      <textarea
        autoFocus
        rows={3}
        className="w-full resize-y rounded-md border border-border bg-background px-2 py-1 font-mono text-[11px]"
        value={draft}
        placeholder={t("promptPreviewUnset")}
        onChange={(e) => setDraft(e.target.value)}
        // Enter saves, Shift+Enter keeps a newline — missions are usually one line but
        // sometimes a short list.
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void save(draft);
          }
          if (e.key === "Escape") setEditing(false);
        }}
      />
      {error ? <p className="text-[10px] text-destructive">{error}</p> : null}
      <div className="flex justify-end gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="h-6 text-[11px]"
          onClick={() => setEditing(false)}
        >
          {t("promptPreviewCancel")}
        </Button>
        <Button size="sm" className="h-6 text-[11px]" onClick={() => void save(draft)}>
          {t("promptPreviewSave")}
        </Button>
      </div>
    </div>
  );
}

function BlockCard({
  block,
  showControl,
  control,
}: {
  block: Block;
  /** False when an earlier block already carries this field's control — the label
      then says so rather than repeating it. */
  showControl: boolean;
  control: React.ReactNode;
}) {
  const t = useTranslations("bankConfig");
  // Everything starts collapsed: the point of this screen is the *shape* of the
  // prompt — which blocks it has and what decides them — and that only fits on one
  // screen with the bodies folded away. Open the one you came for.
  const [open, setOpen] = useState(false);
  const summary = useMemo(() => firstMeaningfulLine(block.text), [block.text]);
  const Chevron = open ? ChevronDown : ChevronRight;

  return (
    // No card border: with a dozen blocks stacked, a box around each one drew a dozen
    // competing rectangles and the coloured bar — the thing that actually says where a
    // block comes from — was the quietest mark on the row. The bar and the row spacing
    // carry the grouping instead. An inactive block still gets a dashed outline, since
    // "this is not in the prompt" has to read as different in kind, not just in colour.
    <div
      className={cn(
        "rounded-lg",
        !block.active && "border border-dashed border-border/70 bg-muted/20"
      )}
    >
      <div className="flex items-start gap-2 px-2 py-2">
        <span className={cn("mt-0.5 h-8 w-1 shrink-0 rounded-full", barFor(block))} />
        <button
          type="button"
          disabled={!block.active}
          onClick={() => setOpen((v) => !v)}
          className="flex min-w-0 flex-1 items-start gap-1.5 text-left disabled:cursor-default"
        >
          {block.active ? (
            <Chevron className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <span className="w-3.5 shrink-0" />
          )}
          <span className="min-w-0">
            <span
              className={cn(
                "block truncate text-xs font-semibold",
                !block.active && "text-muted-foreground"
              )}
            >
              {block.label}
            </span>
            {block.field ? (
              <span className="block truncate font-mono text-[10px] text-muted-foreground">
                {block.field}
              </span>
            ) : null}
            {!open && block.active && (
              <span className="mt-0.5 block truncate font-mono text-[11px] text-muted-foreground">
                {summary}
              </span>
            )}
          </span>
        </button>
        <span className="shrink-0 whitespace-nowrap pt-0.5 text-[10px] tabular-nums text-muted-foreground">
          {block.active
            ? t("promptPreviewChars", { count: block.text.length })
            : t("promptPreviewUnset")}
        </span>
        <div className="flex w-32 shrink-0 justify-end pt-0.5">
          {/* A block with no field has no setting to show — only one whose control is
              deliberately suppressed says "same setting". */}
          {showControl && control}
          {!showControl && block.field ? (
            <span className="text-[11px] italic text-muted-foreground">
              {t("promptPreviewSameSetting")}
            </span>
          ) : null}
        </div>
      </div>

      {!block.active && block.note ? (
        <p className="mx-2 mb-2 rounded border border-border/60 bg-background/60 px-2 py-1.5 text-[11px] text-muted-foreground">
          {block.note}
        </p>
      ) : null}

      {open && block.active && (
        <pre className="mx-2 mb-2 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border/60 bg-muted/20 px-3 py-2 font-mono text-xs leading-relaxed">
          {block.text.replace(/^\n+|\n+$/g, "")}
        </pre>
      )}
    </div>
  );
}

/**
 * Shows the exact prompts an operation would send for this bank, and lets the
 * settings behind them be changed in place — nothing is previewed that is not sent,
 * and no LLM is called.
 *
 * Which message carries the mission depends on the operation — user for retain and
 * observations, whose system prompt is deliberately bank-agnostic so one
 * provider-side cache serves every bank; system for reflect — so the message picker
 * names both and shows how many blocks each has, rather than hiding one.
 */
function PromptPreviewDialog({
  open,
  onOpenChange,
  bankId,
  operation,
  overrides,
  onSaved,
  editDisabledReason,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bankId: string;
  operation: PromptPreviewOperation;
  overrides: Record<string, unknown>;
  /** Told the new value after an inline save, so the Configuration form stays in step. */
  onSaved?: (field: string, value: string | null) => void;
  /** Set to explain why editing is unavailable here. */
  editDisabledReason?: string;
}) {
  const t = useTranslations("bankConfig");
  const tCommon = useTranslations("common");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [role, setRole] = useState<"system" | "user">("system");
  // The whole message as one plain panel — for reading it end to end, or copying
  // it out, which the blocks make awkward.
  const [raw, setRaw] = useState(false);
  // Bumped after a save, to refetch the preview against the value now stored.
  const [reloads, setReloads] = useState(0);

  // `overrides` is a fresh object on every keystroke in the editor, so it cannot be a
  // dependency of the load effect without refetching mid-typing. Reading it through a
  // ref keeps the effect keyed on `open` alone while still sending what the user has
  // typed by the time they open the dialog.
  const overridesRef = useRef(overrides);
  overridesRef.current = overrides;

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setPreview(null);
    setLoading(true);
    setError(null);
    client
      .previewPrompt(bankId, operation, overridesRef.current)
      .then((result) => {
        if (!cancelled) setPreview(result);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, bankId, operation, reloads]);

  function handleSaved(field: string, value: string | null) {
    // The pending override for this field is now what the bank holds; leaving it in
    // would mask the saved value on the refetch below.
    const { [field]: _saved, ...rest } = overridesRef.current;
    overridesRef.current = rest;
    onSaved?.(field, value);
    setReloads((n) => n + 1);
  }

  const message = preview?.messages.find((m) => m.role === role) ?? preview?.messages[0];
  const totalChars =
    preview?.messages.reduce(
      (sum, m) => sum + m.blocks.reduce((s, b) => s + (b.active ? b.text.length : 0), 0),
      0
    ) ?? 0;

  // A field can decide several blocks (the extraction mode picks every built-in block
  // of the system prompt). Its control belongs on the first of them; repeating it
  // would suggest the blocks could be set independently.
  const seenFields = new Set<string>();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* Fixed height: the two messages have different block counts, and a dialog that
          resized on every switch made the picker jump out from under the cursor. */}
      <DialogContent className="flex h-[85vh] max-w-5xl flex-col">
        <DialogHeader>
          <DialogTitle>{t("promptPreviewTitle")}</DialogTitle>
          <DialogDescription>{t("promptPreviewDescription")}</DialogDescription>
        </DialogHeader>

        {preview && !preview.skipped_reason && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-muted-foreground">
            <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]">
              {preview.operation}
            </span>
            <span>·</span>
            <span>{t("promptPreviewMessageCount", { count: preview.messages.length })}</span>
            <span>·</span>
            <span>{t("promptPreviewChars", { count: totalChars })}</span>
          </div>
        )}

        {error ? (
          <p className="text-sm text-destructive">{error}</p>
        ) : loading || !preview ? (
          <div className="flex min-h-0 flex-1 items-center justify-center">
            <Spinner size="sm" />
          </div>
        ) : preview.skipped_reason ? (
          <p className="min-h-0 flex-1 rounded-md border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
            {preview.skipped_reason}
          </p>
        ) : (
          <>
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
              <div className="flex gap-1 rounded-lg bg-muted p-1">
                {preview.messages.map((m) => (
                  <button
                    key={m.role}
                    type="button"
                    onClick={() => setRole(m.role)}
                    className={cn(
                      "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                      m.role === role
                        ? "bg-background shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {m.role === "system"
                      ? t("promptPreviewSystemLabel")
                      : t("promptPreviewUserLabel")}
                    <span className="ml-1.5 text-muted-foreground">· {m.blocks.length}</span>
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <span className="h-3 w-1 rounded-full bg-amber-400" />
                  {t("promptPreviewLegendActive")}
                </span>
                <span className="flex items-center gap-1">
                  <span className="h-3 w-2.5 rounded-sm border border-dashed border-border" />
                  {t("promptPreviewLegendInactive")}
                </span>
                <span className="flex items-center gap-1">
                  <span className="font-mono">«…»</span>
                  {t("promptPreviewLegendRuntime")}
                </span>
              </div>
            </div>

            {raw ? (
              <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted/20 p-3 font-mono text-xs leading-relaxed">
                {message?.blocks
                  .filter((b) => b.active)
                  .map((b) => b.text)
                  .join("")}
              </pre>
            ) : (
              <div className="min-h-0 flex-1 divide-y divide-border/40 overflow-y-auto pr-1">
                {message?.blocks.map((block, i) => {
                  const showControl = !block.field || !seenFields.has(block.field);
                  if (block.field) seenFields.add(block.field);
                  return (
                    <BlockCard
                      key={`${block.field}-${i}`}
                      block={block}
                      showControl={showControl}
                      control={
                        block.field ? (
                          <BlockControl
                            bankId={bankId}
                            block={block}
                            onSaved={handleSaved}
                            disabledReason={editDisabledReason}
                            onClose={() => onOpenChange(false)}
                          />
                        ) : null
                      }
                    />
                  );
                })}
              </div>
            )}
          </>
        )}

        <DialogFooter className="shrink-0 sm:items-center sm:justify-between">
          <p className="text-[11px] text-muted-foreground">{t("promptPreviewFooterNote")}</p>
          {preview && !preview.skipped_reason && (
            <Button variant="outline" size="sm" onClick={() => setRaw((v) => !v)}>
              {raw ? t("promptPreviewShowBlocks") : t("promptPreviewShowRaw")}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/**
 * "Preview prompt" affordance for a mission field: opens {@link PromptPreviewDialog}
 * against the bank currently in context, with the unsaved edits in `overrides`.
 */
export function PreviewPromptButton({
  operation,
  overrides,
  onSaved,
  editDisabledReason,
}: {
  operation: PromptPreviewOperation;
  overrides: Record<string, unknown>;
  /** Told the new value after an inline save, so the Configuration form stays in step. */
  onSaved?: (field: string, value: string | null) => void;
  /** Set to explain why inline editing is unavailable — named retain strategies do
      not live in the bank's top-level config, so saving one from here would write
      the wrong key. */
  editDisabledReason?: string;
}) {
  const t = useTranslations("bankConfig");
  const { currentBank } = useBank();
  const [open, setOpen] = useState(false);

  if (!currentBank) return null;

  return (
    <>
      <Button variant="outline" size="sm" className="shrink-0" onClick={() => setOpen(true)}>
        {t("promptPreviewAction")}
      </Button>
      <PromptPreviewDialog
        open={open}
        onOpenChange={setOpen}
        bankId={currentBank}
        operation={operation}
        overrides={overrides}
        onSaved={onSaved}
        editDisabledReason={editDisabledReason}
      />
    </>
  );
}
