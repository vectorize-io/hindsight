import { promises as fs } from "node:fs";
import path from "node:path";

import {
  Project,
  readStatus,
  type ProjectStatus,
} from "@vectorize-io/hindsight-mission-sandbox/core";

export type { ProjectStatus };

export interface ProjectListItem {
  name: string;
  apiUrl: string;
  versions: number;
  currentBank: string | null;
  createdAt: string;
}

/** Root directory that holds all named projects, set by `mission-sandbox ui [dir]`. */
export function projectsRoot(): string {
  const env = process.env.MISSION_SANDBOX_PROJECTS_DIR;
  return env ? path.resolve(env) : process.cwd();
}

/** Resolve a project name to its directory under the root, rejecting path traversal. */
export function projectDir(name: string): string {
  const slug = name
    .trim()
    .replace(/[^a-zA-Z0-9 _-]/g, "")
    .replace(/\s+/g, "-");
  if (!slug) throw new Error(`Invalid project name: ${JSON.stringify(name)}`);
  const root = projectsRoot();
  const dir = path.resolve(root, slug);
  if (dir !== root && !dir.startsWith(root + path.sep)) throw new Error("Invalid project path");
  return dir;
}

/** List every initialized project under the root, newest first. */
export async function listProjects(): Promise<ProjectListItem[]> {
  const root = projectsRoot();
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const items: ProjectListItem[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(root, entry.name);
    if (!(await Project.exists(dir))) continue;
    const proj = await Project.load(dir);
    items.push({
      name: proj.name,
      apiUrl: proj.apiUrl,
      versions: proj.versions.length,
      currentBank: proj.currentBank(),
      createdAt: proj.createdAt,
    });
  }
  items.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return items;
}

/** Read one project's status, or null if it doesn't exist. */
export async function getStatus(name: string): Promise<ProjectStatus | null> {
  const dir = projectDir(name);
  if (!(await Project.exists(dir))) return null;
  return readStatus(dir);
}
