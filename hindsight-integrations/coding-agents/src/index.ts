/**
 * hindsight-coding-agents — long-term memory for coding agents (recall + INJECT), harness-pluggable.
 *
 * This file is the opencode entrypoint: opencode loads the default export as a persistent Plugin.
 * opencode runs the SAME v2 surface as the Claude Code / Codex hook harnesses, just delivered
 * through opencode's plugin hooks instead of a fresh process per event:
 *   READ   — each user turn, recall on the prompt and PUSH a `<hindsight_memories>` block (with the
 *            attribution + user-feedback framing) into the system prompt.
 *   SEED   — on load, cold-check the bank and (if cold) start a background git-log seed + codebase
 *            survey. (The knowledge-page preamble is built per session, on the first prompt.)
 *   TOOLS  — register the hindsight_* knowledge/recall suite natively (no MCP server needed).
 *   WRITE  — on by default: upsert the rich transcript (text + tool calls/outputs) on the turn
 *            cadence, and again on session.idle — only the idle pass can see the agent's reply.
 *
 * The recall/inject/seed/write-back logic is a harness-agnostic RuntimeCore; the opencode adapter
 * binds it to opencode's plugin API. All configuration comes from ~/.hindsight/coding-agent.json
 * (no environment variables) — see core/config.ts for the shape and defaults.
 *
 * The body lives in harness/plugin-entry.ts, shared with the Kilo entry (src/kilo.ts) — Kilo CLI is
 * an opencode fork running the identical plugin contract.
 */
import { createPluginEntry } from "./harness/plugin-entry";

const HindsightCodingAgentsPlugin = createPluginEntry("opencode");

export default HindsightCodingAgentsPlugin;
export { HindsightCodingAgentsPlugin };
