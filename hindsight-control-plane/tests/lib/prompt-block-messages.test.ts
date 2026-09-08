import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

/**
 * The prompt preview ships no display copy: a block is identified by its config
 * `field` or by a `section` slug, and the dialog looks the name and the switched-off
 * explanation up in these catalogues. A block whose key is missing renders an
 * `IntlError` at runtime and nothing else — which is exactly what happened when
 * `entities_allow_free_form` became a block and its strings were forgotten.
 *
 * These lists mirror what `hindsight_api/engine/prompt_preview.py` can emit. Adding a
 * block there means adding it here, and this test says so before a user finds out.
 */
const FIELDS = [
  "entities_allow_free_form",
  // Run settings: named in the same catalogue, though they produce no block.
  "retain_chunk_size",
  "retain_structured_chunk_size",
  "entity_labels",
  "llm_output_language",
  "observations_mission",
  "reflect_mission",
  "retain_custom_instructions",
  "retain_extract_causal_links",
  "retain_extraction_mode",
  "retain_mission",
];

const SECTIONS = ["bank_identity", "directives", "disposition"];

// Only the consolidation mission always occupies its slot — unset just means the
// built-in default fills it. Everything else can come back switched off, the
// extraction mode included: `chunks` sends no prompt, and the mode block is then
// reported inactive so the control that got you there stays reachable. Exempting it
// here is what let that off-state ship with no wording.
// The consolidation mission's slot is always filled — unset just means the built-in
// default. Run settings never appear as blocks at all, so they have no off-state.
const ALWAYS_ACTIVE = new Set([
  "observations_mission",
  "retain_chunk_size",
  "retain_structured_chunk_size",
]);

const MESSAGES_DIR = join(__dirname, "..", "..", "src", "messages");
const locales = readdirSync(MESSAGES_DIR).filter((f) => f.endsWith(".json"));

describe("prompt block translations", () => {
  it.each(locales)("%s names every block the API can emit", (file) => {
    const messages = JSON.parse(readFileSync(join(MESSAGES_DIR, file), "utf8"));
    const block = messages.bankConfig?.promptBlock;

    expect(block, `${file} is missing bankConfig.promptBlock`).toBeDefined();
    for (const field of FIELDS) {
      expect(block.field[field], `${file} has no name for field ${field}`).toBeTruthy();
      if (!ALWAYS_ACTIVE.has(field)) {
        expect(block.off[field], `${file} has no off-state note for field ${field}`).toBeTruthy();
      }
    }
    for (const section of SECTIONS) {
      expect(block.section[section], `${file} has no name for section ${section}`).toBeTruthy();
      expect(block.off[section], `${file} has no off-state note for section ${section}`).toBeTruthy();
    }
    expect(block.instructions, `${file} has no fallback name for unnamed blocks`).toBeTruthy();
  });
});
