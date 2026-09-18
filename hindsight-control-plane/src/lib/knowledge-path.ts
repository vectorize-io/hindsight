import type { KnowledgeNode } from "@/lib/api";

/**
 * Folder path of every node in a knowledge tree, bank root first
 * ("demo / Engineering / Retrieval").
 *
 * Search results carry ids and names only, so the tree is what says where a hit
 * actually lives — two pages called "Overview" in different folders are
 * otherwise indistinguishable in the result list.
 */
export function buildPathIndex(nodes: KnowledgeNode[], bankId: string): Map<string, string> {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const out = new Map<string, string>();
  for (const n of nodes) {
    const parts: string[] = [];
    let p = n.parent_id ? byId.get(n.parent_id) : undefined;
    while (p) {
      parts.unshift(p.name);
      p = p.parent_id ? byId.get(p.parent_id) : undefined;
    }
    out.set(n.id, [bankId, ...parts].join(" / "));
  }
  return out;
}
