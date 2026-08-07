/**
 * Project persistence. Everything lives in `<dir>/project.json` (camelCase, 1:1 with ProjectMeta).
 * No history dir, no facts/labels — `versions` is the durable record of what was applied.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";

import type {
  CheckResult,
  CurationEvent,
  StepEntry,
  GoldenMemory,
  MissionState,
  ProjectMeta,
  ProjectVersion,
} from "./types.js";

const PROJECT_FILE = "project.json";

function emptyMission(): MissionState {
  return { mission: null, feedback: [] };
}

export class Project {
  name: string;
  documents: string;
  apiUrl: string;
  apiKey: string | null;
  retain: MissionState;
  observe: MissionState;
  versions: ProjectVersion[];
  currentVersion: number | null;
  golden: GoldenMemory[];
  goldenAt: string | null;
  curations: CurationEvent[];
  lastCheck: CheckResult | null;
  steps: StepEntry[];
  createdAt: string;
  readonly dir: string;

  private constructor(dir: string, meta: ProjectMeta) {
    this.dir = dir;
    this.name = meta.name;
    this.documents = meta.documents;
    this.apiUrl = meta.apiUrl;
    this.apiKey = meta.apiKey;
    this.retain = meta.retain;
    this.observe = meta.observe;
    this.versions = meta.versions;
    this.currentVersion = meta.currentVersion;
    this.golden = meta.golden;
    this.goldenAt = meta.goldenAt;
    this.curations = meta.curations;
    this.lastCheck = meta.lastCheck;
    this.steps = meta.steps;
    this.createdAt = meta.createdAt;
  }

  static projectFile(dir: string): string {
    return path.join(dir, PROJECT_FILE);
  }

  static async exists(dir: string): Promise<boolean> {
    return existsSync(Project.projectFile(dir));
  }

  static async create(
    dir: string,
    init: { name: string; documents: string; apiUrl: string; apiKey?: string | null }
  ): Promise<Project> {
    await mkdir(dir, { recursive: true });
    const proj = new Project(dir, {
      name: init.name,
      documents: init.documents,
      apiUrl: init.apiUrl,
      apiKey: init.apiKey ?? null,
      retain: emptyMission(),
      observe: emptyMission(),
      versions: [],
      currentVersion: null,
      golden: [],
      goldenAt: null,
      curations: [],
      lastCheck: null,
      steps: [],
      createdAt: new Date().toISOString(),
    });
    await proj.save();
    return proj;
  }

  static async load(dir: string): Promise<Project> {
    const raw = JSON.parse(await readFile(Project.projectFile(dir), "utf8")) as ProjectMeta;
    return new Project(dir, {
      name: raw.name,
      documents: raw.documents,
      apiUrl: raw.apiUrl,
      apiKey: raw.apiKey ?? null,
      retain: raw.retain ?? emptyMission(),
      observe: raw.observe ?? emptyMission(),
      versions: (raw.versions ?? []).map((v) => ({
        ...v,
        notes: v.notes ?? "",
        feedback: v.feedback ?? [],
      })),
      currentVersion: raw.currentVersion ?? null,
      golden: (raw.golden ?? []).map((g) => ({
        ...g,
        curatedFrom: g.curatedFrom ?? null,
        curateReason: g.curateReason ?? null,
      })),
      goldenAt: raw.goldenAt ?? null,
      curations: raw.curations ?? [],
      lastCheck: raw.lastCheck ?? null,
      steps: raw.steps ?? [],
      createdAt: raw.createdAt,
    });
  }

  async save(): Promise<void> {
    const meta: ProjectMeta = {
      name: this.name,
      documents: this.documents,
      apiUrl: this.apiUrl,
      apiKey: this.apiKey,
      retain: this.retain,
      observe: this.observe,
      versions: this.versions,
      currentVersion: this.currentVersion,
      golden: this.golden,
      goldenAt: this.goldenAt,
      curations: this.curations,
      lastCheck: this.lastCheck,
      steps: this.steps,
      createdAt: this.createdAt,
    };
    await writeFile(Project.projectFile(this.dir), JSON.stringify(meta, null, 2));
  }

  /** Bank id of the current (latest applied) version, or null before the first `retain apply`. */
  currentBank(): string | null {
    if (this.currentVersion === null) return null;
    return `${this.name}-v${this.currentVersion}`;
  }

  /** Record a new retain version: allocates the next number + bank id and makes it current. */
  addVersion(missions: {
    retainMission: string | null;
    observeMission: string | null;
    feedback?: string[];
  }): ProjectVersion {
    const n = this.versions.reduce((max, v) => Math.max(max, v.n), 0) + 1;
    const version: ProjectVersion = {
      n,
      bank: `${this.name}-v${n}`,
      retainMission: missions.retainMission,
      observeMission: missions.observeMission,
      feedback: missions.feedback ?? [],
      notes: "",
      createdAt: new Date().toISOString(),
    };
    this.versions.push(version);
    this.currentVersion = n;
    return version;
  }

  /** Append an action to the activity log (caller persists via save()). */
  addStep(kind: string, summary: string, detail?: string | null): void {
    this.steps.push({
      id: `s${this.steps.length + 1}`,
      at: new Date().toISOString(),
      kind,
      summary,
      detail: detail ?? null,
    });
  }

  /** Set the free-text notes on a version (defaults to the current version). */
  setVersionNotes(notes: string, n?: number): ProjectVersion {
    const target = n ?? this.currentVersion;
    if (target === null) throw new Error("No version to annotate — run `retain apply` first.");
    const version = this.versions.find((v) => v.n === target);
    if (!version) throw new Error(`No version v${target}.`);
    version.notes = notes;
    return version;
  }
}
