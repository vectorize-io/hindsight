import type { KnowledgeNode } from "./api";

export function findRequestedKnowledgePageId(
  nodes: KnowledgeNode[],
  requestedId: string | null
): string | null {
  if (!requestedId) return null;

  for (const node of nodes) {
    if (node.id === requestedId) return node.kind === "page" ? node.id : null;
    const nested = findRequestedKnowledgePageId(node.children, requestedId);
    if (nested) return nested;
  }

  return null;
}
