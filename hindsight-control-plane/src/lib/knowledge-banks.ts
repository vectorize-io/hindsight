/**
 * Knowledge banks: types for the `/v1/default/knowledge-banks` API and the small
 * helpers the screens share. The JSON calls live on `ControlPlaneClient` in
 * `lib/api.ts`; uploads live here because they need XHR for progress events.
 */

import { withBasePath } from "./base-path";

/** What a document or passage is. "document" covers PDF, Office and HTML files. */
export type MediaType = "document" | "image" | "audio" | "video" | "text";
export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export const MEDIA_TYPES: MediaType[] = ["document", "image", "audio", "video", "text"];
export const DOCUMENT_STATUSES: DocumentStatus[] = ["pending", "processing", "ready", "failed"];

export interface KnowledgeBank {
  bank_id: string;
  created_at: string | null;
  document_count: number;
  passage_count: number;
  ready_count: number;
  pending_count: number;
  failed_count: number;
  total_bytes: number;
  documents_by_media_type: Partial<Record<MediaType, number>>;
}

/** The settings a knowledge bank splits its files with: its own value, else the server's. */
export interface KnowledgeIngestSettings {
  text_passage_tokens: number;
  pdf_page_pixels: number;
  pdf_bands: boolean;
  image_tokens: number;
  audio_window_seconds: number;
  video_window_seconds: number;
  media_overlap_seconds: number;
  /** May only be lower than the server's. */
  max_media_seconds: number;
  /** May only be lower than the server's. */
  max_pdf_pages: number;
  /** May only add to the server's. */
  disabled_media_types: MediaType[];
}

export type KnowledgeSettingName = keyof KnowledgeIngestSettings;

/** `GET/PATCH/DELETE /{bank}/config`: every setting resolved, and only the ones the bank sets. */
export interface KnowledgeBankConfig {
  bank_id: string;
  config: KnowledgeIngestSettings;
  overrides: Partial<KnowledgeIngestSettings>;
}

/** A PATCH body's `updates`: a null clears that override. */
export type KnowledgeBankConfigUpdates = {
  [K in KnowledgeSettingName]?: KnowledgeIngestSettings[K] | null;
};

export interface KnowledgeDocument {
  document_id: string;
  filename: string | null;
  content_type: string;
  media_type: MediaType;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  passage_count: number;
  tags: string[];
  created_at: string | null;
  updated_at: string | null;
  document_url: string;
}

export interface KnowledgePassage {
  index: number;
  media_type: MediaType;
  text: string | null;
  page: number | null;
  start_ms: number | null;
  end_ms: number | null;
  passage_url: string | null;
}

export interface KnowledgeDocumentDetail extends KnowledgeDocument {
  passages: KnowledgePassage[];
}

export interface KnowledgeSearchResult extends KnowledgePassage {
  document_id: string;
  filename: string | null;
  tags: string[];
  document_url: string;
}

export type SearchBudget = "low" | "mid" | "high";
export const SEARCH_BUDGETS: SearchBudget[] = ["low", "mid", "high"];

export interface KnowledgeSearchRequest {
  query?: string;
  query_image?: string;
  /** Base64 MP4 clip; combined with the text and the image into one query. */
  query_video?: string;
  /** What the hits, and the text around them, may weigh in all, in tokens (server default 4096). */
  max_tokens?: number;
  /** How deep the search ranks before `max_tokens` picks from it (server default "mid"). */
  budget?: SearchBudget;
  tags?: string[];
  tags_match?: "any" | "all";
  document_ids?: string[];
  media_types?: MediaType[];
}

/** Results best first: the server sends ranks, not scores (they mean nothing across queries). */
export interface KnowledgeSearchResponse {
  results: KnowledgeSearchResult[];
}

export interface KnowledgeIngestResponse {
  document_ids: string[];
  operation_ids: string[];
  unchanged_document_ids: string[];
}

interface FileUploadOptions {
  document_id?: string;
  tags: string[];
}

const API_PREFIX = "/v1/default/knowledge-banks";

/** Fired by the header's "Add Document" on a knowledge bank page; the documents view opens its upload dialog. */
export const KNOWLEDGE_UPLOAD_EVENT = "hindsight:knowledge-upload";

/** Control-plane proxy path for a knowledge-bank endpoint; ids are percent-encoded. */
export function knowledgeApi(...segments: string[]): string {
  const suffix = segments.map(encodeURIComponent).join("/");
  return `/api/knowledge-banks${suffix ? `/${suffix}` : ""}`;
}

/**
 * The API hands out `passage_url`/`document_url` as dataplane paths
 * (`/v1/default/knowledge-banks/...`); the browser reaches them through the proxy.
 */
export function proxiedMediaUrl(apiPath: string): string {
  const path = apiPath.startsWith(API_PREFIX)
    ? `/api/knowledge-banks${apiPath.slice(API_PREFIX.length)}`
    : apiPath;
  return withBasePath(path);
}

/** Page route of a knowledge bank. */
export function knowledgeBankRoute(bankId: string, query = ""): string {
  return `/knowledge-banks/${encodeURIComponent(bankId)}${query}`;
}

/** A passage whose media is a picture: an image, or a document's rendered page. */
export function isPicturePassage(p: Pick<KnowledgePassage, "media_type" | "passage_url">): boolean {
  return (p.media_type === "image" || p.media_type === "document") && p.passage_url != null;
}

export function isInFlight(status: DocumentStatus): boolean {
  return status === "pending" || status === "processing";
}

/** 1536 -> "1.5 KB". Base 1024, one decimal under 10 units. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exp = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exp;
  const digits = exp === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[exp]}`;
}

/** 65_000 -> "1:05", 3_725_000 -> "1:02:05". */
export function formatMs(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

/** "0:00 – 1:00" for a timed passage, or null when it has no window. */
export function formatWindow(
  passage: Pick<KnowledgePassage, "start_ms" | "end_ms">
): string | null {
  if (passage.start_ms == null) return null;
  const end = passage.end_ms != null ? ` – ${formatMs(passage.end_ms)}` : "";
  return `${formatMs(passage.start_ms)}${end}`;
}

export function documentLabel(doc: Pick<KnowledgeDocument, "filename" | "document_id">) {
  return doc.filename || doc.document_id;
}

/**
 * A document's first passage media (a PDF's first page, the image itself, the first
 * audio/video window), for thumbnails. Same path the API hands out as `passage_url`.
 */
export function firstPassageUrl(doc: Pick<KnowledgeDocument, "document_url">): string {
  return doc.document_url.replace(/\/file$/, "/passages/0/media");
}

/** "a, b ,,c" -> ["a", "b", "c"]. */
export function parseTags(raw: string): string[] {
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
}

/** A File as bare base64 (no `data:` prefix), the shape `query_image` takes. */
export function fileToBase64(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result);
      resolve(url.slice(url.indexOf(",") + 1));
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** Builds the multipart body of `POST /{bank}/files/ingest`: `files` parts plus the aligned `request` JSON. */
/** Groups items, in order, into batches of at most `maxBytes`; an item bigger than that goes alone. */
export function packUploads<T>(items: T[], size: (item: T) => number, maxBytes: number): T[][] {
  const batches: T[][] = [];
  let bytes = 0;
  for (const item of items) {
    const last = batches[batches.length - 1];
    if (last && bytes + size(item) <= maxBytes) {
      last.push(item);
      bytes += size(item);
    } else {
      batches.push([item]);
      bytes = size(item);
    }
  }
  return batches;
}

export function buildUploadForm(files: File[], options: FileUploadOptions[]): FormData {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  const entries = options.map((o) => ({
    ...(o.document_id ? { document_id: o.document_id } : {}),
    tags: o.tags,
  }));
  form.append("request", JSON.stringify({ files: entries }));
  return form;
}

/**
 * Ingests files into a knowledge bank, reporting upload progress (0..1).
 * XHR rather than fetch: fetch exposes no upload progress events.
 */
export function uploadKnowledgeFiles(
  bankId: string,
  files: File[],
  options: FileUploadOptions[],
  onProgress: (fraction: number) => void
): Promise<KnowledgeIngestResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", withBasePath(knowledgeApi(bankId, "files", "ingest")));
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // non-JSON error page; fall through to the status text
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as KnowledgeIngestResponse);
      } else {
        reject(new Error(errorMessage(body) || `HTTP ${xhr.status} ${xhr.statusText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.send(buildUploadForm(files, options));
  });
}

/** The readable part of an API error body (`{detail}` from FastAPI, `{error}` from the proxy). */
export function errorMessage(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const { detail, error } = body as { detail?: unknown; error?: unknown };
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
      .join("; ");
  }
  return typeof error === "string" ? error : null;
}

export type ActivityPeriod = "1d" | "7d" | "30d" | "90d";
export const ACTIVITY_PERIODS: ActivityPeriod[] = ["1d", "7d", "30d", "90d"];

export interface ActivityBucket {
  /** Bucket start, ISO in UTC. */
  time: string;
  documents: number;
  passages: number;
}

/**
 * The server's timeseries only lists buckets that have documents; this lays them on
 * the full period (hourly for one day, daily otherwise) so the chart line is
 * continuous rather than a few isolated points.
 */
export function fillTimeseries(
  series: ActivityBucket[],
  period: ActivityPeriod,
  now: Date = new Date()
): { trunc: "hour" | "day"; buckets: ActivityBucket[] } {
  const trunc = period === "1d" ? "hour" : "day";
  const step = trunc === "hour" ? 3_600_000 : 86_400_000;
  const count = trunc === "hour" ? 24 : Number(period.slice(0, -1));
  const end = Math.floor(now.getTime() / step) * step; // start of the current bucket, UTC
  const start = end - (count - 1) * step;
  const buckets: ActivityBucket[] = Array.from({ length: count }, (_, i) => ({
    time: new Date(start + i * step).toISOString(),
    documents: 0,
    passages: 0,
  }));
  for (const b of series) {
    const i = Math.floor((new Date(b.time).getTime() - start) / step);
    if (i < 0 || i >= count) continue;
    buckets[i].documents += b.documents;
    buckets[i].passages += b.passages;
  }
  return { trunc, buckets };
}
