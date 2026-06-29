/**
 * Render a mental model as a markdown document with YAML frontmatter, and pick
 * a safe filename for it.
 */

import type { MentalModel } from "./client.js";
import { stringifyFrontmatter, type Frontmatter } from "./frontmatter.js";

/** Map a mental-model id to a safe, stable `.md` filename. */
export function fileNameFor(id: string): string {
  const safe = id
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/\.+/g, ".");
  return `${safe || "mental-model"}.md`;
}

/**
 * Build the on-disk markdown for a mental model.
 *
 * Frontmatter carries only server-stable metadata (no per-poll timestamp) so an
 * unchanged model renders identical bytes every refresh — keeping file hashes
 * and mtimes stable. The last poll time is tracked in state.json / `status`.
 */
export function renderMentalModel(model: MentalModel): string {
  const frontmatter: Frontmatter = {
    id: model.id,
    name: model.name,
    bank: model.bank_id,
    tags: model.tags ?? [],
    source_query: model.source_query ?? null,
    last_refreshed_at: model.last_refreshed_at ?? null,
    created_at: model.created_at ?? null,
  };
  if (typeof model.is_stale === "boolean") {
    frontmatter.is_stale = model.is_stale;
  }

  const content = (model.content ?? "").trim();
  const body = content.length > 0 ? content : "_This mental model has not been generated yet._";

  return `${stringifyFrontmatter(frontmatter)}\n\n${body}\n`;
}

/** Build the `index.md` overview written into the control directory. */
export function renderIndex(
  models: MentalModel[],
  config: { bankId: string; apiUrl: string }
): string {
  const frontmatter: Frontmatter = {
    bank: config.bankId,
    api_url: config.apiUrl,
    count: models.length,
  };
  const rows = models
    .slice()
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((m) => {
      const stale = m.is_stale === true ? " ⟳ stale" : "";
      const tags = (m.tags ?? []).length ? ` _(${(m.tags ?? []).join(", ")})_` : "";
      return `- [\`${fileNameFor(m.id)}\`](../${fileNameFor(m.id)}) — **${m.name}**${tags}${stale}`;
    })
    .join("\n");

  const list = rows || "_No mental models in this bank yet._";
  return `${stringifyFrontmatter(frontmatter)}\n\n# Mental models — \`${config.bankId}\`\n\n${list}\n`;
}
