import { type Debouncer, Notice, Plugin, TFile, addIcon, debounce, setIcon } from "obsidian";
import { HINDSIGHT_ICON_ID, HINDSIGHT_ICON_SVG, HINDSIGHT_MARK_DATA_URI } from "./branding";
import { ChatView, VIEW_TYPE_CHAT } from "./chat-view";
import { HindsightClient } from "./client";
import { DEFAULT_SETTINGS, type HindsightSettings, HindsightSettingTab } from "./settings";
import { SyncEngine, type SyncConfig, type SyncIndex, type SyncVault } from "./sync";
import { type SyncStatus, renderSyncStatus } from "./status-bar";

interface PluginData {
  settings: HindsightSettings;
  syncIndex: SyncIndex;
  lastSyncAt: number | null;
}

export default class HindsightPlugin extends Plugin {
  settings: HindsightSettings = { ...DEFAULT_SETTINGS };
  private syncIndex: SyncIndex = {};
  private client: HindsightClient | null = null;
  private engine: SyncEngine | null = null;
  private dirty = new Set<string>();
  private scheduleFlush!: Debouncer<[], void>;
  // Sync-status indicator state (surfaced in the status bar).
  private statusBarEl: HTMLElement | null = null;
  private statusTextEl: HTMLElement | null = null;
  private statusIconEl: HTMLElement | null = null;
  private syncing = 0;
  private lastSyncAt: number | null = null;
  private lastSyncError = false;

  async onload(): Promise<void> {
    await this.loadPluginData();
    this.rebuildClient();
    this.scheduleFlush = debounce(() => void this.flushDirty(), 4000, false);

    addIcon(HINDSIGHT_ICON_ID, HINDSIGHT_ICON_SVG);
    this.registerView(VIEW_TYPE_CHAT, (leaf) => new ChatView(leaf, this));
    // Replace the ribbon glyph with the real Hindsight logo (PNG can't go through addIcon).
    const ribbon = this.addRibbonIcon(
      HINDSIGHT_ICON_ID,
      "Hindsight chat",
      () => void this.activateChat()
    );
    ribbon.empty();
    ribbon.createEl("img", {
      cls: "hindsight-ribbon-icon",
      attr: { src: HINDSIGHT_MARK_DATA_URI, alt: "Hindsight" },
    });

    this.addCommand({
      id: "open-chat",
      name: "Open chat",
      callback: () => void this.activateChat(),
    });
    this.addCommand({
      id: "new-chat",
      name: "New chat",
      callback: () => {
        const view = this.app.workspace.getLeavesOfType(VIEW_TYPE_CHAT)[0]?.view;
        if (view instanceof ChatView) view.newChat();
        else void this.activateChat();
      },
    });
    this.addCommand({
      id: "sync-vault",
      name: "Sync vault now",
      callback: () => void this.syncVault(),
    });
    this.addCommand({
      id: "ingest-current-note",
      name: "Ingest current note",
      checkCallback: (checking) => {
        const file = this.app.workspace.getActiveFile();
        if (!file || file.extension !== "md" || !this.engine) return false;
        if (!checking) void this.ingestOne(file);
        return true;
      },
    });

    this.addSettingTab(new HindsightSettingTab(this.app, this));
    this.registerVaultWatchers();

    // Persistent sync indicator: text + a refresh button, click to sync now,
    // refreshed on a timer so "x ago" stays live.
    this.statusBarEl = this.addStatusBarItem();
    this.statusBarEl.addClass("mod-clickable", "hindsight-statusbar");
    this.statusTextEl = this.statusBarEl.createSpan();
    this.statusIconEl = this.statusBarEl.createSpan({ cls: "hindsight-statusbar__icon" });
    setIcon(this.statusIconEl, "refresh-cw");
    this.statusBarEl.addEventListener("click", () => void this.syncVault());
    this.registerInterval(window.setInterval(() => this.updateStatusBar(), 30_000));
    this.updateStatusBar();
  }

  /** Current sync state, consumed by the status bar and the chat header. */
  getSyncStatus(): SyncStatus {
    return {
      configured: this.engine !== null,
      syncing: this.syncing,
      pending: this.dirty.size,
      synced: Object.keys(this.syncIndex).length,
      lastSyncAt: this.lastSyncAt,
      error: this.lastSyncError,
    };
  }

  /** Recompute and paint the sync indicators (status bar + any open chat views). */
  private updateStatusBar(): void {
    const status = this.getSyncStatus();
    if (this.statusBarEl && this.statusTextEl) {
      const view = renderSyncStatus(status, Date.now());
      this.statusTextEl.setText(view.text);
      this.statusBarEl.setAttribute("aria-label", view.tooltip);
      this.statusBarEl.title = view.tooltip;
      this.statusIconEl?.toggleClass("is-syncing", status.syncing > 0);
    }
    for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE_CHAT)) {
      const view = leaf.view;
      if (view instanceof ChatView) view.refreshSyncStatus();
    }
  }

  /**
   * Run a sync operation while reflecting its progress in the status bar:
   * shows "syncing…", then a fresh "synced" timestamp or an error state.
   */
  private async withSync<T>(op: () => Promise<T>): Promise<T> {
    this.syncing += 1;
    this.lastSyncError = false;
    this.updateStatusBar();
    try {
      const result = await op();
      this.lastSyncAt = Date.now();
      return result;
    } catch (err) {
      this.lastSyncError = true;
      throw err;
    } finally {
      this.syncing -= 1;
      this.updateStatusBar();
    }
  }

  // ── Public accessors used by views ──────────────────────────────────────

  getClient(): HindsightClient | null {
    return this.client;
  }

  getBankId(): string {
    // One shared bank across all vaults by default; vaults are separated by a
    // `vault:` tag, not by separate banks (DESIGN.md §4.1).
    return this.settings.bankId.trim() || "obsidian";
  }

  /** Strip a `<vault>/` prefix from a document id so it resolves as a vault path. */
  stripDocPrefix(docId: string): string {
    const prefix = `${this.app.vault.getName()}/`;
    return this.settings.prefixDocId && docId.startsWith(prefix)
      ? docId.slice(prefix.length)
      : docId;
  }

  async saveSettings(): Promise<void> {
    this.rebuildClient();
    await this.savePluginData();
  }

  // ── Actions ─────────────────────────────────────────────────────────────

  private async activateChat(): Promise<void> {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(VIEW_TYPE_CHAT)[0];
    if (!leaf) {
      const right = workspace.getRightLeaf(false);
      if (!right) return;
      leaf = right;
      await leaf.setViewState({ type: VIEW_TYPE_CHAT, active: true });
    }
    await workspace.revealLeaf(leaf);
  }

  async syncVault(): Promise<void> {
    if (!this.engine) {
      new Notice("Hindsight: configure an API URL first.");
      return;
    }
    new Notice("Hindsight: syncing vault…");
    try {
      const engine = this.engine;
      const s = await this.withSync(() => engine.reconcile());
      await this.savePluginData();
      new Notice(
        `Hindsight: ${s.added} added, ${s.updated} updated, ${s.deleted} deleted, ${s.unchanged} unchanged.`
      );
    } catch (err) {
      this.reportError(err);
    }
  }

  private async ingestOne(file: TFile): Promise<void> {
    if (!this.engine) return;
    try {
      const engine = this.engine;
      const outcome = await this.withSync(() => engine.ingestFile(file, { force: true }));
      await this.savePluginData();
      new Notice(`Hindsight: note ${outcome}.`);
    } catch (err) {
      this.reportError(err);
    }
  }

  // ── Vault watchers ──────────────────────────────────────────────────────

  private registerVaultWatchers(): void {
    const isMd = (f: unknown): f is TFile => f instanceof TFile && f.extension === "md";

    const onUpsert = (file: unknown): void => {
      if (!this.settings.syncOnEdit || !isMd(file)) return;
      this.dirty.add(file.path);
      this.updateStatusBar(); // reflect the new pending count immediately
      this.scheduleFlush();
    };
    this.registerEvent(this.app.vault.on("create", onUpsert));
    this.registerEvent(this.app.vault.on("modify", onUpsert));

    this.registerEvent(
      this.app.vault.on("delete", (file) => {
        if (!this.settings.syncOnEdit || !isMd(file) || !this.engine) return;
        const engine = this.engine;
        this.withSync(() => engine.handleDelete(file.path)).catch((err) => this.reportError(err));
      })
    );

    this.registerEvent(
      this.app.vault.on("rename", (file, oldPath) => {
        if (!this.settings.syncOnEdit || !isMd(file) || !this.engine) return;
        const engine = this.engine;
        this.withSync(() => engine.handleRename(file, oldPath)).catch((err) =>
          this.reportError(err)
        );
      })
    );
  }

  private async flushDirty(): Promise<void> {
    if (!this.engine || this.dirty.size === 0) return;
    const engine = this.engine;
    const paths = [...this.dirty];
    this.dirty.clear();
    await this.withSync(async () => {
      for (const path of paths) {
        const file = this.app.vault.getAbstractFileByPath(path);
        if (file instanceof TFile) {
          try {
            await engine.ingestFile(file);
          } catch (err) {
            this.reportError(err);
          }
        }
      }
    });
  }

  // ── Wiring / persistence ────────────────────────────────────────────────

  private rebuildClient(): void {
    try {
      this.client = this.settings.apiUrl.trim()
        ? new HindsightClient(this.settings.apiUrl, this.settings.apiKey)
        : null;
    } catch {
      this.client = null;
    }
    this.engine = this.client
      ? new SyncEngine(
          this.client,
          this.app.vault as unknown as SyncVault,
          this.syncConfig(),
          this.syncIndex,
          (idx) => this.saveIndex(idx)
        )
      : null;
    this.updateStatusBar();
  }

  private syncConfig(): SyncConfig {
    return {
      bankId: this.getBankId(),
      includeFolders: this.settings.includeFolders,
      excludeFolders: this.settings.excludeFolders,
      vaultName: this.app.vault.getName(),
      prefixDocId: this.settings.prefixDocId,
    };
  }

  private async loadPluginData(): Promise<void> {
    const data = (await this.loadData()) as Partial<PluginData> | null;
    this.settings = { ...DEFAULT_SETTINGS, ...(data?.settings ?? {}) };
    this.syncIndex = data?.syncIndex ?? {};
    this.lastSyncAt = data?.lastSyncAt ?? null;
  }

  private async savePluginData(): Promise<void> {
    const data: PluginData = {
      settings: this.settings,
      syncIndex: this.syncIndex,
      lastSyncAt: this.lastSyncAt,
    };
    await this.saveData(data);
  }

  private async saveIndex(index: SyncIndex): Promise<void> {
    this.syncIndex = index;
    await this.savePluginData();
  }

  private reportError(err: unknown): void {
    const detail = err instanceof Error ? err.message : String(err);
    console.error("[hindsight]", err);
    new Notice(`Hindsight: ${detail}`);
  }
}
