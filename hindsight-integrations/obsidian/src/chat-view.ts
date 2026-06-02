/**
 * Chat side panel. Renders a message stream where each assistant turn comes
 * from `reflect`, with collapsible citations (clickable back to the source
 * note — DESIGN.md §0.5) and a reasoning disclosure.
 */

import { ItemView, MarkdownRenderer, Notice, type WorkspaceLeaf } from "obsidian";
import { runChatTurn } from "./chat";
import type HindsightPlugin from "./main";
import type { ReflectResponse } from "./types";

export const VIEW_TYPE_CHAT = "hindsight-chat";

export class ChatView extends ItemView {
  private messagesEl!: HTMLElement;
  private input!: HTMLTextAreaElement;
  private sending = false;

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
    return "brain-circuit";
  }

  async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("hindsight-chat");

    this.messagesEl = root.createDiv({ cls: "hindsight-chat__messages" });

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
    send.addEventListener("click", () => void this.submit());
  }

  private addMessage(role: "user" | "assistant", text: string): HTMLElement {
    const el = this.messagesEl.createDiv({
      cls: `hindsight-chat__msg hindsight-chat__msg--${role}`,
    });
    void MarkdownRenderer.render(this.app, text, el, "", this);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    return el;
  }

  private renderCitations(container: HTMLElement, response: ReflectResponse): void {
    const memories = response.based_on?.memories ?? [];
    const models = response.based_on?.mental_models ?? [];
    const notePaths = [
      ...new Set(memories.map((m) => m.document_id).filter((d): d is string => !!d)),
    ];
    if (notePaths.length === 0 && models.length === 0) return;

    const details = container.createEl("details", { cls: "hindsight-chat__disclosure" });
    details.createEl("summary", {
      text: `Sources (${notePaths.length + models.length})`,
    });
    for (const path of notePaths) {
      const link = details.createEl("a", {
        cls: "hindsight-chat__citation",
        text: path,
        href: "#",
      });
      link.addEventListener("click", (evt) => {
        evt.preventDefault();
        void this.app.workspace.openLinkText(this.plugin.stripDocPrefix(path), "");
      });
    }
    for (const model of models) {
      details.createEl("span", {
        cls: "hindsight-chat__citation",
        text: `🧠 ${model.name ?? "mental model"}`,
      });
    }
  }

  private renderReasoning(container: HTMLElement, response: ReflectResponse): void {
    const calls = response.trace?.tool_calls ?? [];
    if (calls.length === 0) return;
    const details = container.createEl("details", { cls: "hindsight-chat__disclosure" });
    details.createEl("summary", { text: `Reasoning (${calls.length} steps)` });
    for (const call of calls) {
      details.createEl("div", { cls: "hindsight-chat__citation", text: `• ${call.tool}` });
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
        },
        message
      );
      pending.remove();
      const assistant = this.addMessage("assistant", response.text || "_(no answer)_");
      this.renderCitations(assistant, response);
      this.renderReasoning(assistant, response);
    } catch (err) {
      pending.remove();
      const detail = err instanceof Error ? err.message : String(err);
      this.addMessage("assistant", `⚠️ ${detail}`);
    } finally {
      this.sending = false;
    }
  }
}
