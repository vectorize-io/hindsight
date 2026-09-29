/**
 * Project Store — lightweight JSON file database.
 * Stores project data (metadata, analysis, guides) locally.
 * Hindsight handles long-term MEMORY; this handles application STATE.
 *
 * In production, swap this for a real database (Postgres, MongoDB, etc.)
 */

import fs from "fs";
import path from "path";
import type { Project } from "@/types";

const DB_DIR = path.join(process.cwd(), "data");
const DB_FILE = path.join(DB_DIR, "projects.json");

// ─── Initialization ───────────────────────────────────────────────────────────

function ensureDbExists(): void {
  if (!fs.existsSync(DB_DIR)) {
    fs.mkdirSync(DB_DIR, { recursive: true });
  }
  if (!fs.existsSync(DB_FILE)) {
    fs.writeFileSync(DB_FILE, JSON.stringify({ projects: [] }, null, 2));
  }
}

function readDb(): { projects: Project[] } {
  ensureDbExists();
  try {
    const raw = fs.readFileSync(DB_FILE, "utf-8");
    return JSON.parse(raw);
  } catch {
    return { projects: [] };
  }
}

function writeDb(data: { projects: Project[] }): void {
  ensureDbExists();
  fs.writeFileSync(DB_FILE, JSON.stringify(data, null, 2));
}

// ─── CRUD ─────────────────────────────────────────────────────────────────────

export function getAllProjects(userId?: string): Project[] {
  const db = readDb();
  const uid = userId ?? process.env.DEFAULT_USER_ID ?? "reelcraft-user-default";
  return db.projects
    .filter((p) => p.userId === uid)
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
}

export function getProjectById(id: string): Project | null {
  const db = readDb();
  return db.projects.find((p) => p.id === id) ?? null;
}

export function createProject(project: Project): Project {
  const db = readDb();
  db.projects.push(project);
  writeDb(db);
  return project;
}

export function updateProject(id: string, updates: Partial<Project>): Project | null {
  const db = readDb();
  const idx = db.projects.findIndex((p) => p.id === id);
  if (idx === -1) return null;
  db.projects[idx] = {
    ...db.projects[idx],
    ...updates,
    updatedAt: new Date().toISOString(),
  };
  writeDb(db);
  return db.projects[idx];
}

export function deleteProject(id: string): boolean {
  const db = readDb();
  const before = db.projects.length;
  db.projects = db.projects.filter((p) => p.id !== id);
  writeDb(db);
  return db.projects.length < before;
}

// ─── Upload helpers ───────────────────────────────────────────────────────────

const UPLOADS_DIR = path.join(process.cwd(), "uploads");

export function ensureUploadsDir(): string {
  if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
  }
  return UPLOADS_DIR;
}

export function getTmpDir(): string {
  const tmpDir = path.join(process.cwd(), "tmp");
  if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
  }
  return tmpDir;
}

export function cleanupTmpFiles(dir: string): void {
  try {
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  } catch {
    /* ignore cleanup errors */
  }
}
