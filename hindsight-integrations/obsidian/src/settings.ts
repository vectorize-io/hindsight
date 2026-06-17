import { type App, Notice, PluginSettingTab, Setting } from "obsidian";
import type HindsightPlugin from "./main";
import type { Budget } from "./types";

export interface HindsightSettings {
  apiUrl: string;
  apiKey: string;
  /** Blank → the shared default bank `obsidian` (all vaults, separated by vault: tags). */
  bankId: string;
  includeFolders: string[];
  excludeFolders: string[];
  syncOnEdit: boolean;
  defaultBudget: Budget;
  /** DESIGN.md §0.5: OFF by default — keeps Hindsight from becoming a 2nd source of truth. */
  rememberConversations: boolean;
  prefixDocId: boolean;
  /** Log reflect requests/responses to the console (open devtools to view). */
  debugLogging: boolean;
  /** Chat disclosure open/closed state — remembered across sessions (last value wins). */
  notesExpanded: boolean;
  reasoningExpanded: boolean;
}

export const DEFAULT_SETTINGS: HindsightSettings = {
  apiUrl: "https://api.hindsight.vectorize.io",
  apiKey: "",
  bankId: "",
  includeFolders: [],
  excludeFolders: [],
  syncOnEdit: true,
  defaultBudget: "low",
  rememberConversations: false,
  // On by default: all vaults share one bank, so document ids must be vault-prefixed
  // to avoid cross-vault collisions (e.g. two vaults both having Notes/todo.md).
  prefixDocId: true,
  debugLogging: false,
  // Notes default to collapsed (the panel was previously always-open and noisy);
  // thereafter both disclosures remember the user's last open/closed choice.
  notesExpanded: false,
  reasoningExpanded: false,
};

function parseFolders(value: string): string[] {
  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export class HindsightSettingTab extends PluginSettingTab {
  constructor(
    app: App,
    private readonly plugin: HindsightPlugin
  ) {
    super(app, plugin);
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName("API URL")
      .setDesc("Hindsight Cloud default; use http://localhost:8888 for self-hosted.")
      .addText((t) =>
        t
          .setPlaceholder("https://api.hindsight.vectorize.io")
          .setValue(this.plugin.settings.apiUrl)
          .onChange(async (v) => {
            this.plugin.settings.apiUrl = v.trim();
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("API key")
      .setDesc("Hindsight Cloud API key. Stored in this vault's plugin config.")
      .addText((t) => {
        t.setValue(this.plugin.settings.apiKey).onChange(async (v) => {
          this.plugin.settings.apiKey = v.trim();
          await this.plugin.saveSettings();
        });
        t.inputEl.type = "password";
      });

    new Setting(containerEl)
      .setName("Bank name")
      .setDesc(
        "Shared across all your vaults (default: obsidian). Vaults are kept separate by a vault: tag, so use the same bank in each vault."
      )
      .addText((t) =>
        t
          .setPlaceholder(this.plugin.getBankId())
          .setValue(this.plugin.settings.bankId)
          .onChange(async (v) => {
            this.plugin.settings.bankId = v.trim();
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Include folders")
      .setDesc("Comma-separated. Blank = whole vault.")
      .addText((t) =>
        t.setValue(this.plugin.settings.includeFolders.join(", ")).onChange(async (v) => {
          this.plugin.settings.includeFolders = parseFolders(v);
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Exclude folders")
      .setDesc("Comma-separated folders to skip.")
      .addText((t) =>
        t.setValue(this.plugin.settings.excludeFolders.join(", ")).onChange(async (v) => {
          this.plugin.settings.excludeFolders = parseFolders(v);
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Sync on edit")
      .setDesc("Re-ingest notes automatically as you edit. Off = manual 'Sync vault now' only.")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.syncOnEdit).onChange(async (v) => {
          this.plugin.settings.syncOnEdit = v;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Default chat depth")
      .setDesc("Reflect budget for chat answers.")
      .addDropdown((d) =>
        d
          .addOptions({ low: "Low (fast)", mid: "Medium", high: "High (thorough)" })
          .setValue(this.plugin.settings.defaultBudget)
          .onChange(async (v) => {
            this.plugin.settings.defaultBudget = v as Budget;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Remember conversations")
      .setDesc(
        "Off by default. When on, chat turns are stored in Hindsight — this creates memory that does NOT live in your vault."
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.rememberConversations).onChange(async (v) => {
          this.plugin.settings.rememberConversations = v;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Prefix document IDs with vault name")
      .setDesc(
        "On by default — required so vaults sharing a bank don't collide. Only turn off for a single-vault setup."
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.prefixDocId).onChange(async (v) => {
          this.plugin.settings.prefixDocId = v;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Debug logging")
      .setDesc(
        "Log each chat reflect request (incl. scope filter) and the citation sources to the console. Open devtools with Cmd+Opt+I to view."
      )
      .addToggle((t) =>
        t.setValue(this.plugin.settings.debugLogging).onChange(async (v) => {
          this.plugin.settings.debugLogging = v;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl).setName("Test connection").addButton((b) =>
      b.setButtonText("Test").onClick(async () => {
        const client = this.plugin.getClient();
        if (!client) {
          new Notice("Set an API URL first.");
          return;
        }
        new Notice(
          (await client.health()) ? "Hindsight: connected ✓" : "Hindsight: could not connect"
        );
      })
    );
  }
}
