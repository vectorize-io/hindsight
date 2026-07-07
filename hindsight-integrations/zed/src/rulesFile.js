/**
 * Manage Hindsight's recall/retain rule inside Zed's global instructions file.
 *
 * Zed includes a global instructions file (`~/.config/zed/AGENTS.md` on macOS and
 * Linux) in *every* agent conversation. We write a static rule there telling the
 * agent to use the Hindsight MCP tools — recall relevant memory at the start of a
 * task, and retain durable facts as it learns them.
 *
 * The rule lives inside a fenced `<!-- HINDSIGHT:BEGIN -->` … `<!-- HINDSIGHT:END -->`
 * block so we can update or remove it without touching the user's own rules in the
 * same file.
 */

import { readFileSync, writeFileSync, mkdirSync, unlinkSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";

import { isFile } from "./fsutil.js";

export const BEGIN_MARKER = "<!-- HINDSIGHT:BEGIN -->";
export const END_MARKER = "<!-- HINDSIGHT:END -->";

// The recall/retain instruction injected into Zed's global rules.
export const RULE_TEXT =
  "You have persistent long-term memory through the Hindsight MCP server " +
  "(`recall`, `retain`, and `reflect` tools).\n\n" +
  "- At the start of each task, call `recall` with the user's request to load " +
  "relevant decisions, preferences, and project context before you answer. " +
  "Use what's relevant and ignore the rest.\n" +
  "- When you learn a durable fact — an architectural decision, a user " +
  "preference, a convention, or anything worth remembering across sessions — " +
  "call `retain` to store it.\n" +
  "- Do not mention these memory operations unless the user asks about them.";

/** Zed's global instructions file (`~/.config/zed/AGENTS.md`). */
export function defaultRulesPath() {
  return join(homedir(), ".config", "zed", "AGENTS.md");
}

/** Remove an existing HINDSIGHT block (and its surrounding blank lines). */
function stripBlock(text) {
  const start = text.indexOf(BEGIN_MARKER);
  if (start === -1) return text;
  let end = text.indexOf(END_MARKER, start);
  if (end === -1) {
    // Malformed (begin without end) — drop from the marker onward.
    return text.slice(0, start).replace(/\s+$/, "") + "\n";
  }
  end += END_MARKER.length;
  const before = text.slice(0, start).replace(/\s+$/, ""); // rstrip
  const after = text.slice(end).replace(/^\s+/, ""); // lstrip
  if (before && after) return `${before}\n\n${after}`;
  const remainder = before || after;
  return remainder.replace(/\s+$/, "") + (remainder ? "\n" : "");
}

/** Render the fenced HINDSIGHT rule block (no trailing newline). */
export function renderBlock(ruleText = RULE_TEXT) {
  return `${BEGIN_MARKER}\n${ruleText.trim()}\n${END_MARKER}`;
}

/**
 * Write/replace Hindsight's rule block in the instructions file at `path`.
 *
 * Preserves any user-authored content and only rewrites our fenced block,
 * placing it at the top so the memory rule leads the instructions.
 */
export function writeRule(path, ruleText = RULE_TEXT) {
  const existing = isFile(path) ? readFileSync(path, "utf-8") : "";
  const base = stripBlock(existing).replace(/\s+$/, "");
  const block = renderBlock(ruleText);
  const newText = base ? `${block}\n\n${base}\n` : `${block}\n`;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, newText, "utf-8");
  return path;
}

/**
 * Remove Hindsight's rule block from the instructions file, if present.
 *
 * Leaves the rest of the file intact. If removing the block empties a file that
 * held nothing else, the file is deleted.
 */
export function clearRule(path) {
  if (!isFile(path)) return path;
  const existing = readFileSync(path, "utf-8");
  if (!existing.includes(BEGIN_MARKER)) return path;
  const stripped = stripBlock(existing).trim();
  if (!stripped) {
    unlinkSync(path);
    return path;
  }
  writeFileSync(path, stripped + "\n", "utf-8");
  return path;
}

/** Whether our rule block is present in the instructions file at `path`. */
export function isInstalled(path) {
  return isFile(path) && readFileSync(path, "utf-8").includes(BEGIN_MARKER);
}
