/**
 * Video Metadata Extractor
 * Uses ffprobe (via @ffprobe-installer) to extract video metadata server-side.
 * Falls back to safe defaults when ffprobe is unavailable.
 */

import type { VideoMetadata } from "@/types";
import path from "path";
import fs from "fs";

// Dynamically require ffprobe to avoid import errors when it isn't installed
function getFfprobe() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ffprobeInstaller = require("@ffprobe-installer/ffprobe");
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ffmpeg = require("fluent-ffmpeg");
    ffmpeg.setFfprobePath(ffprobeInstaller.path);
    return ffmpeg;
  } catch {
    return null;
  }
}

/**
 * Extract video metadata from a local file path.
 */
export async function extractVideoMetadata(
  filePath: string
): Promise<VideoMetadata> {
  const ffmpeg = getFfprobe();

  if (!ffmpeg) {
    console.warn("ffmpeg/ffprobe not available — returning estimated metadata");
    return estimateMetadata(filePath);
  }

  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err: Error, data: FfprobeData) => {
      if (err) {
        console.warn("ffprobe error:", err.message);
        resolve(estimateMetadata(filePath));
        return;
      }

      try {
        const videoStream = data.streams?.find(
          (s) => s.codec_type === "video"
        );
        const audioStream = data.streams?.find(
          (s) => s.codec_type === "audio"
        );
        const format = data.format;

        const width = videoStream?.width ?? 1080;
        const height = videoStream?.height ?? 1920;
        const duration = parseFloat(String(format?.duration ?? "0"));
        const fileSize = parseInt(String(format?.size ?? "0"), 10);
        const bitrate = parseInt(String(format?.bit_rate ?? "0"), 10);

        // Parse frame rate (often comes as "30/1" or "29.97")
        const fpsRaw = videoStream?.r_frame_rate ?? "30/1";
        const frameRate = parseFrameRate(fpsRaw);

        const aspectRatio = calcAspectRatio(width, height);
        const orientation =
          height > width ? "portrait" : width > height ? "landscape" : "square";

        resolve({
          duration,
          width,
          height,
          aspectRatio,
          frameRate,
          orientation,
          hasAudio: !!audioStream,
          fileSize,
          format: format?.format_name ?? "unknown",
          bitrate: bitrate > 0 ? bitrate : undefined,
        });
      } catch (parseErr) {
        resolve(estimateMetadata(filePath));
      }
    });
  });
}

// ─── Frame Extraction ─────────────────────────────────────────────────────────

export interface ExtractedFrames {
  frames: string[];       // base64-encoded JPEG data URIs
  timestamps: number[];   // corresponding timestamps in seconds
  count: number;
}

/**
 * Extract representative frames from a video file for vision analysis.
 * Extracts ~8-12 evenly spaced frames across the video duration.
 */
export async function extractFrames(
  filePath: string,
  metadata: VideoMetadata,
  outputDir: string,
  maxFrames = 10
): Promise<ExtractedFrames> {
  const ffmpeg = getFfprobe();

  if (!ffmpeg) {
    console.warn("ffmpeg not available — returning empty frame set");
    return { frames: [], timestamps: [], count: 0 };
  }

  // Ensure output dir exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const duration = metadata.duration;
  const interval = duration / (maxFrames + 1);
  const timestamps = Array.from(
    { length: maxFrames },
    (_, i) => parseFloat(((i + 1) * interval).toFixed(2))
  );

  const frameFiles: string[] = [];

  for (const ts of timestamps) {
    const outputFile = path.join(
      outputDir,
      `frame_${ts.toFixed(2).replace(".", "_")}.jpg`
    );

    await new Promise<void>((resolve, reject) => {
      ffmpeg(filePath)
        .seekInput(ts)
        .frames(1)
        .output(outputFile)
        .outputOptions(["-vf", "scale=720:-1"])
        .on("end", () => resolve())
        .on("error", (e: Error) => {
          console.warn(`Frame extraction failed at ${ts}s:`, e.message);
          resolve(); // don't reject — skip this frame
        })
        .run();
    });

    if (fs.existsSync(outputFile)) {
      frameFiles.push(outputFile);
    }
  }

  // Convert to base64 data URIs
  const frames = frameFiles.map((fp) => {
    const buffer = fs.readFileSync(fp);
    return `data:image/jpeg;base64,${buffer.toString("base64")}`;
  });

  // Clean up frame files
  frameFiles.forEach((fp) => {
    try { fs.unlinkSync(fp); } catch { /* ignore */ }
  });

  return {
    frames,
    timestamps: timestamps.slice(0, frames.length),
    count: frames.length,
  };
}

// ─── Scene Boundary Detection ─────────────────────────────────────────────────

/**
 * Rough scene boundary detection by dividing the video into segments.
 * For a proper scene detection we'd use ffmpeg's scene filter,
 * but for reliability we use a time-based split with overlapping context.
 */
export function estimateSceneBoundaries(
  duration: number,
  targetScenes = 8
): Array<{ start: number; end: number }> {
  const sceneDuration = duration / targetScenes;
  return Array.from({ length: targetScenes }, (_, i) => ({
    start: parseFloat((i * sceneDuration).toFixed(2)),
    end: parseFloat(Math.min((i + 1) * sceneDuration, duration).toFixed(2)),
  }));
}

// ─── URL Validation ───────────────────────────────────────────────────────────

const SUPPORTED_DOMAINS = [
  "instagram.com",
  "youtube.com",
  "youtu.be",
  "tiktok.com",
  "twitter.com",
  "x.com",
];

export function validateVideoUrl(url: string): {
  valid: boolean;
  reason?: string;
  domain?: string;
} {
  try {
    const parsed = new URL(url);
    const domain = parsed.hostname.replace("www.", "");

    if (!["http:", "https:"].includes(parsed.protocol)) {
      return { valid: false, reason: "URL must use HTTP or HTTPS" };
    }

    const supported = SUPPORTED_DOMAINS.find((d) => domain.includes(d));
    if (!supported) {
      return {
        valid: false,
        reason: `Unsupported domain. Supported: ${SUPPORTED_DOMAINS.join(", ")}. Please download the video and upload it instead.`,
      };
    }

    return { valid: true, domain };
  } catch {
    return { valid: false, reason: "Invalid URL format" };
  }
}

// ─── Upload Validation ────────────────────────────────────────────────────────

const SUPPORTED_MIME_TYPES = [
  "video/mp4",
  "video/quicktime",
  "video/x-msvideo",
  "video/webm",
  "video/mpeg",
  "video/3gpp",
  "video/x-matroska",
];

const SUPPORTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".webm", ".mpeg", ".3gp", ".mkv"];

export function validateUploadedFile(
  filename: string,
  mimetype: string,
  sizeBytes: number
): { valid: boolean; reason?: string } {
  const maxBytes =
    parseInt(process.env.MAX_UPLOAD_SIZE_MB ?? "200", 10) * 1024 * 1024;

  const ext = path.extname(filename).toLowerCase();

  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    return {
      valid: false,
      reason: `Unsupported file type: ${ext}. Supported: ${SUPPORTED_EXTENSIONS.join(", ")}`,
    };
  }

  if (!SUPPORTED_MIME_TYPES.includes(mimetype) && !mimetype.startsWith("video/")) {
    return {
      valid: false,
      reason: `Invalid file type. Please upload a video file.`,
    };
  }

  if (sizeBytes > maxBytes) {
    const sizeMb = Math.round(sizeBytes / 1024 / 1024);
    const maxMb = Math.round(maxBytes / 1024 / 1024);
    return {
      valid: false,
      reason: `File too large: ${sizeMb}MB. Maximum allowed: ${maxMb}MB`,
    };
  }

  return { valid: true };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function parseFrameRate(raw: string): number {
  if (raw.includes("/")) {
    const [num, den] = raw.split("/").map(Number);
    return den ? Math.round(num / den) : 30;
  }
  return parseFloat(raw) || 30;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

function calcAspectRatio(w: number, h: number): string {
  const d = gcd(w, h);
  return `${w / d}:${h / d}`;
}

function estimateMetadata(filePath: string): VideoMetadata {
  let fileSize = 0;
  try {
    fileSize = fs.statSync(filePath).size;
  } catch { /* ignore */ }

  return {
    duration: 30,
    width: 1080,
    height: 1920,
    aspectRatio: "9:16",
    frameRate: 30,
    orientation: "portrait",
    hasAudio: true,
    fileSize,
    format: "mp4",
  };
}

// ─── Types (local, minimal) ───────────────────────────────────────────────────

interface FfprobeStream {
  codec_type?: string;
  width?: number;
  height?: number;
  r_frame_rate?: string;
}

interface FfprobeFormat {
  duration?: string | number;
  size?: string | number;
  bit_rate?: string | number;
  format_name?: string;
}

interface FfprobeData {
  streams?: FfprobeStream[];
  format?: FfprobeFormat;
}
