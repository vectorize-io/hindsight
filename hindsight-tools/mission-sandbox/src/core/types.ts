/**
 * Domain types for mission-sandbox.
 *
 * A project binds a documents path + API config, and tracks two missions (retain + observe) that
 * you refine from feedback. Each `retain apply` ingests into a fresh versioned bank `<name>-vN`.
 * Task success is measured by an EXTERNAL validator — the tool stores no facts, labels, or scores.
 */

/** Progress sink so CLI (console) and any caller can render the same step output. */
export type ProgressFn = (message: string) => void;

/** Which mission a feedback round targets. */
export type MissionKind = "retain" | "observe";

/** A mission and the feedback that has shaped it, newest last. */
export interface MissionState {
  mission: string | null;
  feedback: string[];
}

/** One ingested bank version: the missions it was built with, plus free-text notes. */
export interface ProjectVersion {
  n: number;
  bank: string;
  retainMission: string | null;
  observeMission: string | null;
  /** The retain feedback entries that shaped this version (the delta since the previous one). */
  feedback: string[];
  /** Free-text notes for this version — e.g. validator results ("LOCOMO 4/5, missed X"). */
  notes: string;
  createdAt: string;
}

/** A frozen "golden" memory — the target Phase 2 optimizes the mission to reproduce. */
export interface GoldenMemory {
  docId: string;
  text: string;
  factType: string;
  /** If this memory was edited in Phase 1: the text before the edit + the reason (provenance). */
  curatedFrom: string | null;
  curateReason: string | null;
}

export type CurationKind = "edit" | "invalidate" | "revert";

/** A curation action applied to a memory in Phase 1 (in-place, no re-ingest). */
export interface CurationEvent {
  id: string;
  memoryId: string;
  kind: CurationKind;
  before: string;
  after: string | null;
  reason: string | null;
  at: string;
}

/** One recorded action in the loop — shown as a timeline in the UI. */
export interface StepEntry {
  id: string;
  at: string;
  /** Short command label, e.g. "trace", "curate", "retain check", "eval". */
  kind: string;
  summary: string;
  detail: string | null;
}

export interface ProjectMeta {
  /** Bank-id prefix; version banks are `${name}-v${n}`. */
  name: string;
  /** Absolute path to the documents directory bound at init. */
  documents: string;
  apiUrl: string;
  apiKey: string | null;
  retain: MissionState;
  observe: MissionState;
  versions: ProjectVersion[];
  /** Version number of the current (latest applied) bank, or null before the first apply. */
  currentVersion: number | null;
  /** Phase 1 output: the frozen target memory set + the curations that produced it. */
  golden: GoldenMemory[];
  goldenAt: string | null;
  curations: CurationEvent[];
  /** Phase 2: result of the most recent `retain check` (coverage of golden by the current mission). */
  lastCheck: CheckResult | null;
  /** Chronological log of every command run — the loop's story for the UI. */
  steps: StepEntry[];
  createdAt: string;
}

/** Coverage of the golden set by the current retain mission (per-doc re-extraction). */
export interface CheckResult {
  coverage: number;
  covered: number;
  total: number;
  docs: number;
  at: string;
}
