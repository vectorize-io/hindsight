"use client";

import { useEffect, useRef, useState } from "react";

import {
  CalendarArrowDown,
  CalendarArrowUp,
  ChevronDown,
  ChevronRight,
  Eye,
  FlaskConical,
  Globe,
  User,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { toast } from "sonner";

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
import { EntityLabelsEditor, type LabelGroup } from "@/components/entity-labels-editor";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { cn } from "@/lib/utils";

type PromptPreviewOperation = "retain" | "consolidation" | "reflect";

type BlockSource = "config" | "builtin";
type InputKind = "text" | "boolean" | "choice" | "complex";
type Block = {
  text: string;
  source: BlockSource;
  field: string;
  section: string;
  heading: string;
  active: boolean;
  value?: string | null;
  kind: InputKind;
  choices?: string[] | null;
  editable: boolean;
};
type Message = { role: "system" | "user"; blocks: Block[] };
type Preview = {
  messages: Message[];
  strategy?: string | null;
  strategies?: string[];
  run_settings?: RunSetting[];
  response_schema?: Record<string, unknown> | null;
  skipped_reason?: string | null;
};
type Fact = {
  text: string;
  fact_type: string;
  entities: string[];
  occurred_start?: string | null;
  occurred_end?: string | null;
  chunk_index?: number | null;
};
type RunSetting = {
  field: string;
  value?: string | null;
  kind: InputKind;
  editable: boolean;
};
type Chunk = { text: string; fact_count: number };
type DryRunResult = { facts: Fact[]; chunks?: Chunk[]; usage?: Record<string, unknown> | null };

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

/**
 * What to call a block, and — when it is switched off — what turning it on would do.
 *
 * The API deliberately ships no display copy: a block is identified by its config
 * `field`, by a `section` slug for the parts no field owns, or by the `heading` the
 * prompt text itself carries. All three are machine values, so the wording lives
 * here where next-intl can translate it.
 */
function useBlockCopy() {
  const t = useTranslations("bankConfig");

  function name(block: Block): string {
    if (block.field) return t(`promptBlock.field.${block.field}` as never);
    if (block.section) return t(`promptBlock.section.${block.section}` as never);
    return block.heading || t("promptBlock.instructions");
  }

  function inactiveNote(block: Block): string {
    const key = block.field || block.section;
    return t(`promptBlock.off.${key}` as never);
  }

  return { name, inactiveNote };
}

/**
 * The surface for prompt text — the block bodies and the raw view.
 *
 * Deliberately not the muted grey the rest of the dialog uses: the extraction results
 * sit beside it in that grey, and the two are different kinds of thing. This is what
 * gets sent; that is what came back.
 */
const PROMPT_SURFACE =
  "border-indigo-200/70 bg-indigo-50/50 dark:border-indigo-500/25 dark:bg-indigo-950/25";

function barFor(block: Block): string {
  if (!block.active) return "bg-transparent";
  if (block.source === "builtin" && !block.field) return "bg-border";
  return FIELD_BAR[block.field] ?? "bg-border";
}

/**
 * The control for the setting a block belongs to, shown on the block's own header row.
 *
 * Saving PATCHes only this one field, so other unsaved edits in the surrounding
 * Configuration form are left alone — but that form has to be told, or it would keep
 * showing (and later re-save) the value this replaced. That is what `onSaved` is for.
 *
 * It carries the value in the shape the form holds it, not the string shape the block
 * reports: `entity_labels` is an array, and its display rendering is not something
 * that can be assigned back.
 */
type FieldEditor = {
  draft: string;
  setDraft: (v: string) => void;
  saving: boolean;
  error: string | null;
  save: (next: string) => void;
};

/** Draft state and the save call for one setting. */
function useFieldEditor(
  bankId: string,
  block: Block,
  name: string,
  onSaved: (field: string, value: unknown) => void,
  onDone: () => void
): FieldEditor {
  const t = useTranslations("bankConfig");
  const [draft, setDraft] = useState(block.value ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The preview refetches after every save, so the block's value is the source of
  // truth; re-seed the draft whenever it changes underneath us.
  useEffect(() => setDraft(block.value ?? ""), [block.value]);

  function save(next: string) {
    setSaving(true);
    setError(null);
    client
      .updateBankConfig(bankId, {
        // An empty box means "unset" — a null override, so the bank falls back to the
        // server default, exactly as clearing the field in the Configuration form does.
        [block.field]: block.kind === "boolean" ? next === "true" : next === "" ? null : next,
      })
      .then(() => {
        onDone();
        toast.success(t("promptPreviewSaved", { name }));
        onSaved(block.field, next === "" ? null : next);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setSaving(false));
  }

  return { draft, setDraft, saving, error, save };
}

/**
 * The trigger for a block's setting, sitting on the block's own header row.
 *
 * Text settings only open the editor here — the editor itself is rendered below the
 * row by {@link BlockCard}, full width. Inline it was a 16rem box wedged into a
 * header, which is the wrong shape for a mission that runs to several lines.
 */
function BlockControl({
  block,
  editor,
  editing,
  setEditing,
  onClose,
  name,
}: {
  block: Block;
  editor: FieldEditor;
  editing: boolean;
  setEditing: (v: boolean) => void;
  /** Closes the dialog, for a setting whose editor lives on the page behind it. */
  onClose: () => void;
  /** The block's display name, for "Add <name>". */
  name: string;
}) {
  const t = useTranslations("bankConfig");

  if (editor.saving) return <Spinner size="sm" />;

  if (!block.editable) {
    return (
      <span className="text-[11px] italic text-muted-foreground">
        {t("promptPreviewServerLevel")}
      </span>
    );
  }

  if (block.kind === "boolean") {
    return (
      <Switch
        checked={block.value === "true"}
        onCheckedChange={(checked) => editor.save(checked ? "true" : "false")}
      />
    );
  }

  if (block.kind === "choice") {
    return (
      <select
        className="h-7 rounded-md border border-border bg-background px-2 text-xs"
        value={block.value ?? ""}
        onChange={(e) => editor.save(e.target.value)}
      >
        {(block.choices ?? []).map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    );
  }

  return (
    <Button
      variant="outline"
      size="sm"
      className="h-7 text-xs"
      onClick={() => setEditing(!editing)}
    >
      {block.active
        ? t("promptPreviewEdit")
        : t("promptPreviewAddNamed", { name: name.toLowerCase() })}
    </Button>
  );
}

/**
 * Entity labels edited in place, using the same editor the Configuration form does.
 *
 * Its value has structure, so it cannot go through the plain textarea — and sending
 * the reader to another screen for it broke the loop the lab exists for. Saved
 * explicitly rather than on every keystroke: the editor emits a new array for every
 * character typed into a label description.
 */
function EntityLabelsBlockEditor({
  bankId,
  block,
  onSaved,
  onCancel,
}: {
  bankId: string;
  block: Block;
  onSaved: (field: string, value: unknown) => void;
  onCancel: () => void;
}) {
  const t = useTranslations("bankConfig");
  const [groups, setGroups] = useState<LabelGroup[]>(() => {
    try {
      const parsed = JSON.parse(block.value ?? "[]");
      // Guarded, not just parsed: the editor maps over this, so anything that is not
      // an array takes the whole dialog down rather than showing an empty editor.
      return Array.isArray(parsed) ? (parsed as LabelGroup[]) : [];
    } catch {
      return [];
    }
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function save() {
    setSaving(true);
    setError(null);
    client
      .updateBankConfig(bankId, { entity_labels: groups.length ? groups : null })
      .then(() => {
        toast.success(t("promptPreviewSaved", { name: t("promptBlock.field.entity_labels") }));
        onSaved("entity_labels", groups.length ? groups : null);
        onCancel();
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setSaving(false));
  }

  return (
    <div className="mx-2 mb-2 space-y-2 rounded-md border border-border bg-background/60 p-2">
      <EntityLabelsEditor value={groups} onChange={setGroups} />
      {error ? <p className="text-[10px] text-destructive">{error}</p> : null}
      <div className="flex justify-end gap-1">
        <Button variant="ghost" size="sm" className="h-6 text-[11px]" onClick={onCancel}>
          {t("promptPreviewCancel")}
        </Button>
        <Button size="sm" className="h-6 text-[11px]" disabled={saving} onClick={save}>
          {saving ? <Spinner size="sm" /> : t("promptPreviewSave")}
        </Button>
      </div>
    </div>
  );
}

/** The setting's editor, below its block and the full width of it. */
function BlockEditor({ editor, onCancel }: { editor: FieldEditor; onCancel: () => void }) {
  const t = useTranslations("bankConfig");
  return (
    <div className="mx-2 mb-2 space-y-1">
      <textarea
        autoFocus
        rows={4}
        className="w-full resize-y rounded-md border border-border bg-background px-2 py-1.5 font-mono text-xs"
        value={editor.draft}
        placeholder={t("promptPreviewUnset")}
        onChange={(e) => editor.setDraft(e.target.value)}
        // Enter saves, Shift+Enter keeps a newline — missions are usually one line but
        // sometimes a short list.
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            editor.save(editor.draft);
          }
          if (e.key === "Escape") onCancel();
        }}
      />
      {editor.error ? <p className="text-[10px] text-destructive">{editor.error}</p> : null}
      <div className="flex justify-end gap-1">
        <Button variant="ghost" size="sm" className="h-6 text-[11px]" onClick={onCancel}>
          {t("promptPreviewCancel")}
        </Button>
        <Button size="sm" className="h-6 text-[11px]" onClick={() => editor.save(editor.draft)}>
          {t("promptPreviewSave")}
        </Button>
      </div>
    </div>
  );
}

function BlockCard({
  block,
  showControl,
  bankId,
  onSaved,
  onClose,
}: {
  block: Block;
  /** False when an earlier block already carries this field's control — the label
      then says so rather than repeating it. */
  showControl: boolean;
  bankId: string;
  onSaved: (field: string, value: unknown) => void;
  onClose: () => void;
}) {
  const t = useTranslations("bankConfig");
  const copy = useBlockCopy();
  const [editing, setEditing] = useState(false);
  const editor = useFieldEditor(bankId, block, copy.name(block), onSaved, () => setEditing(false));
  // Everything starts collapsed: the point of this screen is the *shape* of the
  // prompt — which blocks it has and what decides them — and that only fits on one
  // screen with the bodies folded away. Open the one you came for.
  const [open, setOpen] = useState(false);
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
      {/* A fixed grid, not a flex row: the bar, the chevron, the name and the control
          each get their own column, so a block with no control and one with a select
          still line up. Under flex, an absent control let the name stretch and every
          row started somewhere different. */}
      <div className="grid grid-cols-[4px_14px_1fr_9rem] items-center gap-2 px-2 py-2.5">
        <span className={cn("h-7 w-1 rounded-full", barFor(block))} />
        {block.active ? (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="flex h-full items-center justify-center text-muted-foreground"
            aria-label={copy.name(block)}
          >
            <Chevron className="h-3.5 w-3.5" />
          </button>
        ) : (
          <span />
        )}
        <button
          type="button"
          disabled={!block.active}
          onClick={() => setOpen((v) => !v)}
          className="min-w-0 text-left disabled:cursor-default"
        >
          {/* The name alone. The text, the config key and the size all live inside the
              panel: a dozen rows each carrying a truncated line of prompt, a mono
              field name and a char count read as a table of metadata rather than as
              the shape of a prompt. */}
          <span
            className={cn(
              "block truncate text-sm font-semibold",
              !block.active && "text-muted-foreground"
            )}
          >
            {copy.name(block)}
          </span>
        </button>
        <div className="flex justify-end">
          {/* A block with no field has no setting to show — only one whose control is
              deliberately suppressed says "same setting". */}
          {showControl && block.field ? (
            <BlockControl
              block={block}
              editor={editor}
              editing={editing}
              setEditing={setEditing}
              onClose={onClose}
              name={copy.name(block)}
            />
          ) : null}
          {!showControl && block.field ? (
            <span className="text-xs italic text-muted-foreground">
              {t("promptPreviewSameSetting")}
            </span>
          ) : null}
        </div>
      </div>

      {editing ? (
        block.kind === "complex" ? (
          <EntityLabelsBlockEditor
            bankId={bankId}
            block={block}
            onSaved={onSaved}
            onCancel={() => setEditing(false)}
          />
        ) : (
          <BlockEditor editor={editor} onCancel={() => setEditing(false)} />
        )
      ) : null}

      {!block.active && !editing ? (
        <div className="mx-2 mb-2 space-y-1 rounded border border-border/60 bg-background/60 px-2 py-1.5">
          <p className="text-xs leading-relaxed text-muted-foreground">
            {copy.inactiveNote(block)}
          </p>
          {block.field ? (
            <p className="font-mono text-[11px] text-muted-foreground">{block.field}</p>
          ) : null}
        </div>
      ) : null}

      {open && block.active && (
        <div className="mx-2 mb-2 space-y-1">
          <p className="flex flex-wrap items-center gap-x-2 text-[11px] text-muted-foreground">
            {block.field ? <span className="font-mono">{block.field}</span> : null}
            {block.field ? <span>·</span> : null}
            <span className="tabular-nums">
              {t("promptPreviewChars", { count: block.text.length })}
            </span>
          </p>
          <pre
            className={cn(
              "max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border px-3 py-2 font-mono text-xs leading-relaxed",
              PROMPT_SURFACE
            )}
          >
            {block.text.replace(/^\n+|\n+$/g, "")}
          </pre>
        </div>
      )}
    </div>
  );
}

/**
 * The paid half of the tester: sample text in, real extracted facts out.
 *
 * Deliberately on demand. Editing a setting re-renders the blocks for free, but a
 * run is a real LLM call — re-running it on every toggle would spend one per
 * keystroke-sized change, on whatever model the bank is pointed at.
 */
/** One run setting: its name, and a box to change it. */
/** One extracted fact. */
/**
 * One chunk and the facts it produced, collapsible.
 *
 * Only the first opens on a run: a long input cuts into many chunks, and a page of
 * expanded ones is a wall to scroll rather than a list to navigate. The header keeps
 * the count visible so a chunk that yielded nothing is obvious while closed.
 */
function ChunkCard({
  chunk,
  index,
  facts,
  defaultOpen,
}: {
  chunk: Chunk;
  index: number;
  facts: Fact[];
  defaultOpen: boolean;
}) {
  const t = useTranslations("bankConfig");
  const [open, setOpen] = useState(defaultOpen);
  const Chevron = open ? ChevronDown : ChevronRight;

  return (
    <div className="overflow-hidden rounded-md border border-border bg-muted/10">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-muted/20"
      >
        <Chevron className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
          {t("testerChunkLabel", { index: index + 1 })}
        </span>
        <span className="text-[11px] text-muted-foreground">·</span>
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
          {t("testerFactCount", { count: chunk.fact_count })}
        </span>
        {!open ? (
          <span className="ml-2 min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground">
            {chunk.text}
          </span>
        ) : null}
      </button>

      {open ? (
        <div className="space-y-2 px-3 pb-3">
          <ChunkText text={chunk.text} />
          <ul className="space-y-2">
            {facts.map((fact, i) => (
              <FactRow key={i} fact={fact} />
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

/**
 * A chunk's text, clamped to two lines until clicked.
 *
 * A chunk runs to the configured chunk size — thousands of characters by default —
 * and printing it in full pushed the facts it produced off the screen, which are the
 * thing being read. Two lines is enough to recognise where in the input you are.
 */
function ChunkText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <button
      type="button"
      onClick={() => setExpanded((v) => !v)}
      title={expanded ? undefined : text}
      className={cn(
        "block w-full whitespace-pre-wrap break-words text-left font-mono text-xs leading-relaxed text-muted-foreground hover:text-foreground",
        !expanded && "line-clamp-2"
      )}
    >
      {text}
    </button>
  );
}

/**
 * An occurrence bound as the badges show it: `YYYY-MM-DD HH:mm`.
 *
 * Kept in UTC rather than the viewer's zone, because these are compared against the
 * raw payload right beside them — a local rendering would shift the day for anyone
 * east or west of Greenwich and disagree with the JSON on screen. The full instant
 * goes in the tooltip, seconds included.
 */
function factDate(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString().slice(0, 16).replace("T", " ");
}

/** One extracted fact: what it is, when it happened, and what it mentions. */
function FactRow({ fact }: { fact: Fact }) {
  const t = useTranslations("bankConfig");
  const isExperience = fact.fact_type === "experience";

  return (
    <li className="space-y-1.5 rounded-md border border-border bg-background/60 px-3 py-2.5 text-sm">
      <p className="leading-relaxed">{fact.text}</p>
      <div className="flex flex-wrap items-center gap-1.5">
        {/* World and experience are the two perspectives a fact can take, and which
            one it got is the classification most worth checking — so it is coloured
            rather than left as another grey chip. */}
        <span
          className={cn(
            "inline-flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium",
            isExperience
              ? "bg-purple-100 text-purple-900 dark:bg-purple-500/25 dark:text-purple-100"
              : "bg-sky-100 text-sky-900 dark:bg-sky-500/25 dark:text-sky-100"
          )}
        >
          {isExperience ? <User className="h-3 w-3" /> : <Globe className="h-3 w-3" />}
          {fact.fact_type}
        </span>

        {/* Dates only when the fact carries them: most do not, and an empty "—" badge
            on every row is noise standing in for absence. */}
        {fact.occurred_start ? (
          <span
            className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5 text-[11px] tabular-nums text-muted-foreground"
            title={`${t("testerOccurredStart")} — ${fact.occurred_start}`}
          >
            <CalendarArrowUp className="h-3 w-3" />
            {factDate(fact.occurred_start)}
          </span>
        ) : null}
        {fact.occurred_end ? (
          <span
            className="inline-flex items-center gap-1 rounded bg-muted px-2 py-0.5 text-[11px] tabular-nums text-muted-foreground"
            title={`${t("testerOccurredEnd")} — ${fact.occurred_end}`}
          >
            <CalendarArrowDown className="h-3 w-3" />
            {factDate(fact.occurred_end)}
          </span>
        ) : null}

        {fact.entities.length ? (
          <span className="ml-1 font-mono text-[11px] text-muted-foreground">
            {fact.entities.join(", ")}
          </span>
        ) : null}
      </div>
    </li>
  );
}

function RunSettingRow({
  bankId,
  setting,
  onSaved,
}: {
  bankId: string;
  setting: RunSetting;
  onSaved: () => void;
}) {
  const t = useTranslations("bankConfig");
  const [draft, setDraft] = useState(setting.value ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => setDraft(setting.value ?? ""), [setting.value]);

  function save() {
    if (draft === (setting.value ?? "")) return;
    setSaving(true);
    client
      .updateBankConfig(bankId, {
        // Blank means "unset" — fall back to the server default, as clearing the field
        // in the Configuration form does. Chunk sizes are numbers on the wire.
        [setting.field]: draft === "" ? null : Number(draft),
      })
      .then(() => {
        toast.success(
          t("promptPreviewSaved", { name: t(`promptBlock.field.${setting.field}` as never) })
        );
        onSaved();
      })
      .finally(() => setSaving(false));
  }

  return (
    <label className="flex items-center gap-2 text-[11px]">
      <span className="min-w-0 flex-1 truncate text-muted-foreground">
        {t(`promptBlock.field.${setting.field}` as never)}
      </span>
      <input
        type="number"
        className="h-6 w-24 rounded border border-border bg-background px-1.5 text-right text-[11px] disabled:opacity-60"
        value={draft}
        disabled={!setting.editable || saving}
        placeholder={t("promptPreviewUnset")}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
        }}
      />
    </label>
  );
}

function ExtractionPanel({
  bankId,
  strategy,
  raw,
  settings,
  onSaved,
}: {
  bankId: string;
  strategy?: string | null;
  /** Driven by the dialog's single Show raw switch, so prompt and result flip together. */
  raw: boolean;
  /** Settings that shape the run without appearing in the prompt — the chunk sizes. */
  settings: RunSetting[];
  onSaved: () => void;
}) {
  const t = useTranslations("bankConfig");
  // Seeded, not left empty: the run button is disabled without text, so an empty box
  // makes the panel look broken until you think of something to paste.
  const [content, setContent] = useState(() => t("testerSampleDefault"));
  const [result, setResult] = useState<DryRunResult | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [runId, setRunId] = useState(0);
  const [error, setError] = useState<string | null>(null);

  function run() {
    setRunning(true);
    setError(null);
    // Wall-clock around the whole call, which is what a reader is judging: extraction
    // is the slow part of a retain, and the number that matters is how long this
    // bank's configuration takes, not the provider's own accounting.
    const startedAt = performance.now();
    client
      .dryRunExtract(bankId, content, strategy)
      .then((value) => {
        setResult(value);
        setElapsedMs(Math.round(performance.now() - startedAt));
        setRunId((n) => n + 1);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setRunning(false));
  }

  const facts = result?.facts ?? null;

  return (
    // Its own column beside the prompt, not a strip beneath it: the sample box and
    // the run button are the point of the panel, and scrolling past a 6000-line
    // prompt to reach them would defeat it. Only the results scroll.
    <section className="flex min-h-0 flex-col gap-3 border-t border-border pt-3 lg:border-l lg:border-t-0 lg:pl-5 lg:pt-0">
      {settings.length ? (
        // Beside the sample rather than among the prompt blocks: these decide how the
        // input is cut before a prompt exists, so they belong to the run, not the text.
        // Folded away by default — this column is the sample box, the button and the
        // results, and a chunk size you set once should not sit above all three.
        <details className="group shrink-0">
          <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium uppercase tracking-wide text-muted-foreground hover:text-foreground">
            <ChevronRight className="h-3.5 w-3.5 transition-transform group-open:rotate-90" />
            {t("testerRunSettingsLabel")}
          </summary>
          <div className="mt-1.5 space-y-1.5 pl-4">
            {settings.map((setting) => (
              <RunSettingRow
                key={setting.field}
                bankId={bankId}
                setting={setting}
                onSaved={onSaved}
              />
            ))}
          </div>
        </details>
      ) : null}

      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {t("testerSampleLabel")}
      </p>
      <textarea
        rows={3}
        className="w-full shrink-0 resize-y rounded-md border border-border bg-background px-3 py-2 text-sm"
        placeholder={t("testerSamplePlaceholder")}
        value={content}
        onChange={(e) => setContent(e.target.value)}
      />
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs leading-relaxed text-muted-foreground">{t("testerRunNote")}</p>
        <Button size="sm" className="shrink-0" disabled={running || !content.trim()} onClick={run}>
          {running ? <Spinner size="sm" className="mr-2" /> : null}
          {t("testerRunAction")}
        </Button>
      </div>

      {error ? <p className="text-xs text-destructive">{error}</p> : null}

      {result ? (
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
          <span>{t("testerFactCount", { count: result.facts.length })}</span>
          {result.chunks?.length ? (
            <>
              <span>·</span>
              <span>{t("testerChunkCount", { count: result.chunks.length })}</span>
            </>
          ) : null}
          {elapsedMs !== null ? (
            <>
              <span>·</span>
              <span className="tabular-nums">{t("testerElapsed", { ms: elapsedMs })}</span>
            </>
          ) : null}
        </p>
      ) : null}

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto">
        {/* The dialog's one Show raw switch drives both halves: the prompt on the left
            and the whole dry-run payload here, `usage` included. */}
        {raw && result ? (
          <pre className="whitespace-pre-wrap break-words rounded-md border border-border bg-muted/20 p-3 font-mono text-xs leading-relaxed">
            {JSON.stringify(result, null, 2)}
          </pre>
        ) : facts ? (
          <>
            {(result?.chunks ?? []).map((chunk, index) => (
              <ChunkCard
                // Keyed by the run as well as the position, so a new run remounts the
                // cards and the "first one open" rule applies again rather than
                // leaving whatever the last run was left scrolled open.
                key={`${runId}-${index}`}
                chunk={chunk}
                index={index}
                facts={facts.filter((fact) => fact.chunk_index === index)}
                defaultOpen={index === 0}
              />
            ))}

            {/* Anything the counts could not attribute — never expected, but dropping
                facts on the floor because of it would be worse than an odd heading. */}
            {facts.some((fact) => fact.chunk_index == null) ? (
              <ul className="space-y-1">
                {facts
                  .filter((fact) => fact.chunk_index == null)
                  .map((fact, i) => (
                    <FactRow key={i} fact={fact} />
                  ))}
              </ul>
            ) : null}
          </>
        ) : null}
      </div>
    </section>
  );
}

/**
 * Shows the exact prompts an operation would send for this bank, and lets the
 * settings behind them be changed in place — nothing is previewed that is not sent,
 * and no LLM is called.
 *
 * What it shows is always the bank's *saved* configuration: the endpoint takes no
 * overrides, so an unsaved edit in the Configuration form behind this dialog is not
 * reflected here. Editing a block saves it, and the blocks then re-render against
 * what the bank actually holds.
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
  onSaved,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bankId: string;
  operation: PromptPreviewOperation;
  /** Told the new value after an inline save, so the Configuration form stays in step. */
  onSaved?: (field: string, value: unknown) => void;
}) {
  const t = useTranslations("bankConfig");
  const tCommon = useTranslations("common");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [loading, setLoading] = useState(false);
  // Whether anything is on screen yet. A ref, not `preview`, because the load effect
  // must not re-run when the preview it just set changes.
  const hasPreview = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [role, setRole] = useState<"system" | "user">("system");
  // null means "whatever the bank defaults to" — the API resolves it and reports
  // back which one applied, so the picker shows the truth rather than a blank.
  const [strategy, setStrategy] = useState<string | null>(null);
  // Only retain can be *run*: it owns the extraction panel. Consolidation and reflect
  // render a prompt and stop there, so they are a preview, not a lab.
  const isLab = operation === "retain";
  // The whole message as one plain panel — for reading it end to end, or copying
  // it out, which the blocks make awkward.
  const [raw, setRaw] = useState(false);
  // Bumped after a save, to refetch the preview against the value now stored.
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    // Only show the spinner when there is nothing to look at. A refetch after a save
    // used to clear the preview and drop a spinner in, so every toggle flashed the
    // whole dialog empty and back; the blocks now stay put and swap under you.
    setLoading(!hasPreview.current);
    setError(null);
    client
      .previewPrompt(bankId, operation, strategy)
      .then((result) => {
        if (cancelled) return;
        hasPreview.current = true;
        setPreview(result);
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
  }, [open, bankId, operation, reloads, strategy]);

  // Closing discards what was on screen, so the next open starts from the spinner
  // rather than briefly showing another bank's or operation's blocks.
  useEffect(() => {
    if (!open) {
      hasPreview.current = false;
      setPreview(null);
    }
  }, [open]);

  function handleSaved(field: string, value: unknown) {
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
      <DialogContent className="flex h-[85vh] max-w-7xl flex-col">
        <DialogHeader>
          <DialogTitle>{t(isLab ? "extractionTesterTitle" : "promptPreviewTitle")}</DialogTitle>
          <DialogDescription>
            {t(isLab ? "extractionTesterDescription" : "promptPreviewDescription")}
          </DialogDescription>
        </DialogHeader>

        {preview && !preview.skipped_reason && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
            <span className="rounded bg-muted px-2 py-0.5 font-mono text-[11px]">{operation}</span>
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
        ) : (
          <>
            {preview.skipped_reason ? (
              <p className="shrink-0 rounded-md border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
                {preview.skipped_reason}
              </p>
            ) : null}

            <div className="flex shrink-0 flex-wrap items-center justify-between gap-x-3 gap-y-2">
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
              {operation === "retain" ? (
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  {t("testerStrategyLabel")}
                  <select
                    className="h-7 rounded-md border border-border bg-background px-1.5 text-xs disabled:opacity-60"
                    // Bound to the local selection, not the resolved one: picking
                    // "default" must stay on "default" rather than jumping to the
                    // name the server resolved it to.
                    value={strategy ?? ""}
                    disabled={(preview.strategies?.length ?? 0) === 0}
                    onChange={(e) => setStrategy(e.target.value || null)}
                  >
                    <option value="">
                      {(preview.strategies?.length ?? 0) === 0
                        ? t("testerNoStrategies")
                        : preview.strategy
                          ? t("testerDefaultStrategyNamed", { name: preview.strategy })
                          : t("testerDefaultStrategy")}
                    </option>
                    {preview.strategies?.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
            </div>

            <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="flex min-h-0 flex-col">
                {raw && !preview.skipped_reason ? (
                  <pre
                    className={cn(
                      "min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words rounded-md border p-3 font-mono text-xs leading-relaxed",
                      PROMPT_SURFACE
                    )}
                  >
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
                          bankId={bankId}
                          onSaved={handleSaved}
                          onClose={() => onOpenChange(false)}
                        />
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Retain only: the other operations have no dry run to pair with. */}
              {/* Shown in chunks mode too: no prompt is sent, but the mode still
                  produces memories — one per chunk, verbatim — and that is exactly
                  what a reader would want to check. */}
              {operation === "retain" ? (
                <ExtractionPanel
                  bankId={bankId}
                  strategy={preview.strategy}
                  raw={raw}
                  settings={preview.run_settings ?? []}
                  onSaved={() => setReloads((n) => n + 1)}
                />
              ) : null}
            </div>
          </>
        )}

        <DialogFooter className="shrink-0 sm:items-center sm:justify-end">
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
 * against the bank currently in context.
 */
export function PreviewPromptButton({
  operation,
  onSaved,
}: {
  operation: PromptPreviewOperation;
  /** Told the new value after an inline save, so the Configuration form stays in step. */
  onSaved?: (field: string, value: unknown) => void;
}) {
  const t = useTranslations("bankConfig");
  const { currentBank } = useBank();
  const [open, setOpen] = useState(false);
  const isLab = operation === "retain";

  if (!currentBank) return null;

  return (
    <>
      <Button
        variant={isLab ? "default" : "outline"}
        size="sm"
        className={cn(
          "shrink-0",
          // The emerald the Hindsight Cloud call-to-action uses in Memory Defense, so
          // the one prominent button on a config screen looks like the others.
          isLab && "bg-emerald-600 text-white shadow-sm hover:bg-emerald-700"
        )}
        onClick={() => setOpen(true)}
      >
        {isLab ? <FlaskConical className="mr-2 h-4 w-4" /> : <Eye className="mr-2 h-4 w-4" />}
        {t(isLab ? "extractionTesterAction" : "promptPreviewAction")}
      </Button>
      <PromptPreviewDialog
        open={open}
        onOpenChange={setOpen}
        bankId={currentBank}
        operation={operation}
        onSaved={onSaved}
      />
    </>
  );
}
