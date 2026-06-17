/**
 * Chat side panel. Renders a message stream where each assistant turn comes
 * from `reflect`, with collapsible citations (clickable back to the source
 * note — DESIGN.md §0.5) and a reasoning disclosure.
 */

import { ItemView, MarkdownRenderer, Notice, type WorkspaceLeaf, setIcon } from "obsidian";
import { HINDSIGHT_ICON_ID, HINDSIGHT_MARK_DATA_URI } from "./branding";
import { runChatTurn } from "./chat";
import type HindsightPlugin from "./main";
import { collectDocIds, groundedNotes, type RetrievedNote } from "./reflect-util";
import { renderSyncStatus } from "./status-bar";
import type { Budget, ReflectResponse, ReflectToolCall, TagGroup, TagLeaf } from "./types";

const SNIPPET_MAX = 160;

// Grounded notes shown before the "show all" toggle — keeps a long retrieval
// list from burying the answer (the rest are one click away).
const NOTES_VISIBLE = 3;

// px — the composer grows with the message up to this height, then scrolls.
const INPUT_MAX_HEIGHT = 240;

export const VIEW_TYPE_CHAT = "hindsight-chat";

const ALL = ""; // sentinel for "no filter" in the dropdowns

function callQuery(call: ReflectToolCall): string {
  const q = call.input?.query;
  return typeof q === "string" ? q : "";
}

export class ChatView extends ItemView {
  private messagesEl!: HTMLElement;
  private input!: HTMLTextAreaElement;
  private sending = false;
  private vaultFilter = ALL;
  private folderFilter = ALL;
  private emptyState: HTMLElement | null = null;
  private syncStatusEl: HTMLElement | null = null;
  private syncButtonEl: HTMLElement | null = null;

  constructor(
    leaf: WorkspaceLeaf,
    private readonly plugin: HindsightPlugin
  ) {
    super(leaf);
  }

  getViewType(): string {
    return VIEW_TYPE_CHAT;
  }

  getDisplayText(): string {
    return "Hindsight chat";
  }

  getIcon(): string {
    return HINDSIGHT_ICON_ID;
  }

  async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("hindsight-chat");

    // "New chat" lives in the view's header action bar (top-right).
    this.addAction("square-pen", "New chat", () => this.newChat());

    this.buildHeader(root);
    this.messagesEl = root.createDiv({ cls: "hindsight-chat__messages" });
    this.renderEmptyState();

    // Filters sit just above the ask bar (right-aligned), not at the top.
    this.buildFilterBar(root);
    const composer = root.createDiv({ cls: "hindsight-chat__composer" });
    this.input = composer.createEl("textarea", {
      attr: { placeholder: "Ask about your vault…", rows: "2" },
    });
    const send = composer.createEl("button", { text: "Ask" });

    this.input.addEventListener("keydown", (evt) => {
      if (evt.key === "Enter" && !evt.shiftKey) {
        evt.preventDefault();
        void this.submit();
      }
    });
    // Grow the box to fit multi-line input (Shift+Enter), capped by CSS max-height.
    this.input.addEventListener("input", () => this.autoGrowInput());
    send.addEventListener("click", () => void this.submit());
  }

  /** Resize the composer to fit its content, up to INPUT_MAX_HEIGHT (then it scrolls). */
  private autoGrowInput(): void {
    const el = this.input;
    // Use Obsidian's setCssStyles (not el.style.x =) per the plugin guidelines.
    // Reset to auto first so scrollHeight reflects the content's natural height.
    el.setCssStyles({ height: "auto" });
    el.setCssStyles({ height: `${Math.min(el.scrollHeight, INPUT_MAX_HEIGHT)}px` });
  }

  private buildHeader(root: HTMLElement): void {
    const header = root.createDiv({ cls: "hindsight-chat__header" });
    header.createEl("img", {
      cls: "hindsight-chat__brand-mark",
      attr: { src: HINDSIGHT_MARK_DATA_URI, alt: "Hindsight" },
    });
    header.createEl("span", { cls: "hindsight-chat__brand-name", text: "Hindsight" });
    // Sync status (label) + an explicit refresh button that triggers a sync.
    this.syncStatusEl = header.createEl("span", { cls: "hindsight-chat__sync" });
    this.syncButtonEl = header.createEl("span", {
      cls: "hindsight-chat__sync-btn clickable-icon",
      attr: { "aria-label": "Sync vault now" },
    });
    setIcon(this.syncButtonEl, "refresh-cw");
    this.syncButtonEl.addEventListener("click", () => void this.plugin.syncVault());
    this.refreshSyncStatus();
  }

  /** Repaint the header sync label + refresh button from the current sync state. */
  refreshSyncStatus(): void {
    if (!this.syncStatusEl) return;
    const status = this.plugin.getSyncStatus();
    const view = renderSyncStatus(status, Date.now(), "");
    this.syncStatusEl.setText(view.text);
    this.syncStatusEl.title = view.tooltip;
    this.syncButtonEl?.setAttribute("aria-label", view.tooltip);
    this.syncButtonEl?.toggleClass("is-syncing", status.syncing > 0);
  }

  /** Reset the conversation to a blank slate. */
  newChat(): void {
    this.messagesEl.empty();
    this.emptyState = null;
    this.renderEmptyState();
    this.input?.focus();
  }

  private renderEmptyState(): void {
    this.emptyState = this.messagesEl.createDiv({ cls: "hindsight-chat__empty" });
    this.emptyState.createEl("img", {
      cls: "hindsight-chat__empty-mark",
      attr: { src: HINDSIGHT_MARK_DATA_URI, alt: "" },
    });
    this.emptyState.createEl("div", {
      cls: "hindsight-chat__empty-text",
      text: "Ask anything about your vault. Answers are grounded on your notes and cite them.",
    });
  }

  private buildFilterBar(root: HTMLElement): void {
    const bar = root.createDiv({ cls: "hindsight-chat__filters" });

    const vaultName = this.app.vault.getName();
    const vaultSel = bar.createEl("select", { cls: "hindsight-chat__filter" });
    vaultSel.createEl("option", { text: "All vaults", value: ALL });
    vaultSel.createEl("option", { text: `Vault: ${vaultName}`, value: vaultName });
    vaultSel.value = this.vaultFilter;
    vaultSel.addEventListener("change", () => {
      this.vaultFilter = vaultSel.value;
    });

    const folderSel = bar.createEl("select", { cls: "hindsight-chat__filter" });
    folderSel.createEl("option", { text: "All folders", value: ALL });
    for (const folder of this.folderOptions()) {
      folderSel.createEl("option", { text: folder, value: folder });
    }
    folderSel.value = this.folderFilter;
    folderSel.addEventListener("change", () => {
      this.folderFilter = folderSel.value;
    });

    // Chat depth (reflect budget) — settable inline, and written back to the
    // persisted default so the chat-window choice sticks across turns/sessions.
    const depthSel = bar.createEl("select", { cls: "hindsight-chat__filter" });
    const depths: Record<Budget, string> = {
      low: "Depth: Low",
      mid: "Depth: Medium",
      high: "Depth: High",
    };
    for (const [value, label] of Object.entries(depths)) {
      depthSel.createEl("option", { text: label, value });
    }
    depthSel.value = this.plugin.settings.defaultBudget;
    depthSel.addEventListener("change", () => {
      this.plugin.settings.defaultBudget = depthSel.value as Budget;
      void this.plugin.saveSettings();
    });
  }

  /** Distinct folder paths in the current vault (mirrors the folder: tags we emit). */
  private folderOptions(): string[] {
    const folders = new Set<string>();
    for (const file of this.app.vault.getMarkdownFiles()) {
      const dir = file.path.includes("/") ? file.path.slice(0, file.path.lastIndexOf("/")) : "";
      const parts = dir ? dir.split("/") : [];
      for (let i = 0; i < parts.length; i++) {
        folders.add(parts.slice(0, i + 1).join("/"));
      }
    }
    return [...folders].sort();
  }

  /** Build the reflect scope filter from the dropdown selections (undefined = whole bank). */
  private scopeTagGroups(): TagGroup[] | undefined {
    const leaves: TagLeaf[] = [];
    if (this.vaultFilter) leaves.push({ tags: [`vault:${this.vaultFilter}`], match: "all_strict" });
    if (this.folderFilter)
      leaves.push({ tags: [`folder:${this.folderFilter}`], match: "all_strict" });
    if (leaves.length === 0) return undefined;
    // Single leaf, or AND the leaves; pass a one-element list so the server's
    // list-level combination is unambiguous.
    return leaves.length === 1 ? [leaves[0]] : [{ and: leaves }];
  }

  private addMessage(role: "user" | "assistant", text: string): HTMLElement {
    this.emptyState?.remove();
    this.emptyState = null;
    const el = this.messagesEl.createDiv({
      cls: `hindsight-chat__msg hindsight-chat__msg--${role}`,
    });
    // sourcePath = the active note so any [[wikilinks]] in the answer resolve.
    const sourcePath = this.app.workspace.getActiveFile()?.path ?? "";
    void MarkdownRenderer.render(this.app, text, el, sourcePath, this);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    return el;
  }

  private openNote(docId: string): void {
    void this.app.workspace.openLinkText(this.plugin.stripDocPrefix(docId), "");
  }

  /** Actions under an assistant answer. */
  private renderActions(container: HTMLElement, text: string): void {
    const actions = container.createDiv({ cls: "hindsight-chat__actions" });

    const copy = actions.createEl("button", { cls: "hindsight-chat__action", text: "Copy" });
    copy.addEventListener("click", () => {
      void navigator.clipboard.writeText(text);
      new Notice("Hindsight: answer copied");
    });
  }

  /** Render one grounded note (citation link + snippet); returns the row element. */
  private renderNoteItem(parent: HTMLElement, note: RetrievedNote): HTMLElement {
    const item = parent.createDiv({ cls: "hindsight-chat__note" });
    const link = item.createEl("a", {
      cls: "hindsight-chat__citation",
      text: this.plugin.stripDocPrefix(note.docId),
      href: "#",
    });
    link.addEventListener("click", (evt) => {
      evt.preventDefault();
      this.openNote(note.docId);
    });
    const snippet = note.snippets[0];
    if (snippet) {
      item.createDiv({
        cls: "hindsight-chat__snippet",
        text: snippet.length > SNIPPET_MAX ? `${snippet.slice(0, SNIPPET_MAX)}…` : snippet,
      });
    }
    return item;
  }

  /** The notes the answer is grounded on (citations), capped with a "show all". */
  private renderRetrievedNotes(container: HTMLElement, response: ReflectResponse): void {
    const notes = groundedNotes(response);
    const models = response.based_on?.mental_models ?? [];
    if (notes.length === 0 && models.length === 0) return;

    const details = container.createEl("details", { cls: "hindsight-chat__disclosure" });
    // Collapsed by default; thereafter remembers the user's last choice (persisted).
    details.open = this.plugin.settings.notesExpanded;
    this.persistDisclosure(details, "notesExpanded");
    details.createEl("summary", { text: `Notes retrieved (${notes.length})` });

    // Render all rows but hide the overflow; "show all" reveals them in place.
    const overflow: HTMLElement[] = [];
    notes.forEach((note, i) => {
      const item = this.renderNoteItem(details, note);
      if (i >= NOTES_VISIBLE) {
        item.hidden = true;
        overflow.push(item);
      }
    });
    if (overflow.length > 0) {
      const more = details.createEl("a", {
        cls: "hindsight-chat__show-all",
        text: `Show all (${overflow.length} more)`,
        href: "#",
      });
      more.addEventListener("click", (evt) => {
        evt.preventDefault();
        more.remove();
        for (const el of overflow) el.hidden = false;
      });
    }

    for (const model of models) {
      details.createEl("span", {
        cls: "hindsight-chat__citation",
        text: `🧠 ${model.name ?? "mental model"}`,
      });
    }
  }

  /** Persist a disclosure's open/closed state to settings (last value wins). */
  private persistDisclosure(
    details: HTMLDetailsElement,
    key: "notesExpanded" | "reasoningExpanded"
  ): void {
    details.addEventListener("toggle", () => {
      this.plugin.settings[key] = details.open;
      void this.plugin.saveSettings();
    });
  }

  private renderReasoning(container: HTMLElement, response: ReflectResponse): void {
    const calls = response.trace?.tool_calls ?? [];
    if (calls.length === 0) return;
    const details = container.createEl("details", { cls: "hindsight-chat__disclosure" });
    details.open = this.plugin.settings.reasoningExpanded;
    this.persistDisclosure(details, "reasoningExpanded");
    details.createEl("summary", { text: `Reasoning (${calls.length} steps)` });
    for (const call of calls) {
      const ids = new Set<string>();
      collectDocIds(call.output, ids);
      const query = callQuery(call);
      const noteCount = ids.size ? ` — ${ids.size} note${ids.size === 1 ? "" : "s"}` : "";
      const label = `• ${call.tool}${query ? ` · “${query}”` : ""}${noteCount}`;
      details.createEl("div", { cls: "hindsight-chat__citation", text: label });
    }
  }

  private async submit(): Promise<void> {
    if (this.sending) return;
    const message = this.input.value.trim();
    if (!message) return;

    const client = this.plugin.getClient();
    if (!client) {
      new Notice("Hindsight: set your API URL in settings first.");
      return;
    }

    this.sending = true;
    this.input.value = "";
    this.autoGrowInput();
    this.addMessage("user", message);
    const pending = this.messagesEl.createDiv({
      cls: "hindsight-chat__msg hindsight-chat__msg--assistant hindsight-chat__pending",
      text: "Thinking…",
    });

    try {
      const response = await runChatTurn(
        {
          client,
          bankId: this.plugin.getBankId(),
          budget: this.plugin.settings.defaultBudget,
          rememberConversations: this.plugin.settings.rememberConversations,
          tagGroups: this.scopeTagGroups(),
          debug: this.plugin.settings.debugLogging,
        },
        message
      );
      pending.remove();
      const assistant = this.addMessage("assistant", response.text || "_(no answer)_");
      this.renderRetrievedNotes(assistant, response);
      this.renderReasoning(assistant, response);
      if (response.text) this.renderActions(assistant, response.text);
    } catch (err) {
      pending.remove();
      const detail = err instanceof Error ? err.message : String(err);
      this.addMessage("assistant", `⚠️ ${detail}`);
    } finally {
      this.sending = false;
    }
  }
}
