/**
 * Turn a knowledge-base snapshot (folder/page tree + page contents) into a
 * concrete on-disk mirror plan: which directories to create (folders) and which
 * `.md` files to write (pages), at their nested paths.
 */

import type { KnowledgeNode, KnowledgeSnapshot } from "./client.js";
import { stringifyFrontmatter, type Frontmatter } from "./frontmatter.js";

const PAGE_PLACEHOLDER = "_This page has not been generated yet._";

/** Map a folder/page name to a safe path segment (no slashes, lowercase). */
export function slug(name: string): string {
  const safe = name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/\.+/g, ".");
  return safe || "untitled";
}

export interface PageFile {
  /** Relative path under the mount, e.g. "engineering/runbooks/orders.md". */
  relPath: string;
  content: string;
}

export interface MirrorPlan {
  /** Folder directories to create, in tree order (parents before children). */
  dirs: string[];
  /** Page files to write at their nested paths. */
  files: PageFile[];
  folderCount: number;
  pageCount: number;
}

/** Ensure sibling nodes get distinct path segments even if their names collide. */
function uniqueSegment(base: string, id: string, used: Set<string>): string {
  if (!used.has(base)) {
    used.add(base);
    return base;
  }
  const suffix = id.replace(/[^a-z0-9]/gi, "").slice(-6) || "x";
  let seg = `${base}-${suffix}`;
  while (used.has(seg)) seg = `${seg}-x`;
  used.add(seg);
  return seg;
}

function pageContent(node: KnowledgeNode, snapshot: KnowledgeSnapshot): string {
  const doc = snapshot.content.get(node.id);
  if (doc && doc.trim().length > 0) return doc.endsWith("\n") ? doc : `${doc}\n`;
  // Fallback OKF-ish doc when the page body hasn't synthesized yet.
  const fm: Frontmatter = {
    id: node.id,
    type: "knowledge-page",
    title: node.name,
    tags: node.tags ?? [],
    timestamp: node.timestamp ?? null,
  };
  return `${stringifyFrontmatter(fm)}\n\n${PAGE_PLACEHOLDER}\n`;
}

/** Walk the tree into a flat list of directories + page files at nested paths. */
export function planMirror(snapshot: KnowledgeSnapshot): MirrorPlan {
  const dirs: string[] = [];
  const files: PageFile[] = [];
  let folderCount = 0;
  let pageCount = 0;

  function walk(nodes: KnowledgeNode[], parentDir: string): void {
    const used = new Set<string>();
    const ordered = [...nodes].sort((a, b) => a.name.localeCompare(b.name));
    for (const node of ordered) {
      const seg = uniqueSegment(slug(node.name), node.id, used);
      const rel = parentDir ? `${parentDir}/${seg}` : seg;
      if (node.kind === "folder") {
        folderCount++;
        dirs.push(rel);
        walk(node.children ?? [], rel);
      } else {
        pageCount++;
        files.push({ relPath: `${rel}.md`, content: pageContent(node, snapshot) });
      }
    }
  }

  walk(snapshot.roots, "");
  return { dirs, files, folderCount, pageCount };
}

/** Build the `index.md` overview written into the control directory. */
export function renderIndex(
  snapshot: KnowledgeSnapshot,
  config: { bankId: string; apiUrl: string }
): string {
  const plan = planMirror(snapshot);
  const frontmatter: Frontmatter = {
    bank: config.bankId,
    api_url: config.apiUrl,
    folders: plan.folderCount,
    pages: plan.pageCount,
  };

  const lines: string[] = [];
  function walk(nodes: KnowledgeNode[], depth: number, parentDir: string, used: Set<string>): void {
    for (const node of [...nodes].sort((a, b) => a.name.localeCompare(b.name))) {
      const seg = uniqueSegment(slug(node.name), node.id, used);
      const rel = parentDir ? `${parentDir}/${seg}` : seg;
      const indent = "  ".repeat(depth);
      if (node.kind === "folder") {
        const mission = node.mission ? ` — _${node.mission}_` : "";
        lines.push(`${indent}- **${node.name}/**${mission}`);
        walk(node.children ?? [], depth + 1, rel, new Set<string>());
      } else {
        const auto = node.managed ? " ·auto" : "";
        lines.push(`${indent}- [\`${node.name}\`](../${rel}.md)${auto}`);
      }
    }
  }
  walk(snapshot.roots, 0, "", new Set<string>());

  const body = lines.length ? lines.join("\n") : "_This bank has no knowledge base yet._";
  return `${stringifyFrontmatter(frontmatter)}\n\n# Knowledge base — \`${config.bankId}\`\n\n${body}\n`;
}
