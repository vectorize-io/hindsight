/**
 * opencode harness adapter — full v2 parity with the hook harnesses (Claude Code / Codex).
 *
 * Maps opencode's persistent-plugin hooks onto the shared RuntimeCore, and reads opencode sessions
 * for backfill. opencode is the cleanest platform of the lot: a real per-turn event, a working
 * system-prompt injection channel, transcript access, and NATIVE tool registration — so the whole
 * v2 surface (per-turn recall + attribution/user-feedback injection, the hindsight_* knowledge
 * tools, cold-check auto-seed, rich write-back) rides these five hooks with no MCP server needed.
 * This is the only opencode-specific file; everything it uses is in ../core.
 */
import { tool } from "@opencode-ai/plugin";
import type { Config, Hooks } from "@opencode-ai/plugin";
import type { RuntimeCore } from "../core/runtime";
import type { HarnessAdapter } from "../core/types";
import { diag } from "../core/diag";
import type { ToolSpec } from "../core/knowledge-tools";
import {
  readOpencodeMessages,
  opencodeSessionId,
  type OcMessage,
} from "../core/transcript-opencode";
import { jsonChatReader } from "./registry";
import { SURVEY_AGENT, SURVEY_AGENT_CONFIG } from "../core/survey";

// opencode message part shape for the per-turn prompt (structurally typed).
type Part = { type?: string; text?: string };

/**
 * Teach the host about the survey agent (core/survey.ts spawns `opencode run --agent
 * hindsight-survey`), so the recipe needs nothing in the user's opencode.json.
 *
 * Declared as `Pick<Hooks, "config">` rather than inlined into the returned object, and that IS the
 * point: plugin-entry.ts casts the whole runtime object to the host's Hooks type (the other hooks
 * take deliberately narrower params than the SDK declares, so they cannot be checked), which means
 * nothing would catch this hook being misnamed or its signature changing under us — it would just
 * silently never fire, and the survey would die on an agent the host never heard of. Pinning this
 * one hook to the SDK's own type restores that check where it matters.
 */
const surveyAgentHook: Pick<Hooks, "config"> = {
  config: async (cfg: Config) => {
    cfg.agent ??= {};
    // Never overwrite an existing entry: a user who defined `hindsight-survey` themselves outranks
    // us.
    //
    // The cast covers one field the published type under-describes: `permission` is declared with a
    // fixed key set (edit/bash/webfetch/…), while the runtime takes arbitrary action names —
    // opencode's own built-in `explore` agent is defined with `"*": "deny"` plus per-tool allows,
    // and a live 1.18.9 session under this agent was offered exactly glob/grep/read/
    // hindsight_ingest_document. Casting the entry beats widening it to `unknown`, which would drop
    // the checking on `description`/`mode` too.
    cfg.agent[SURVEY_AGENT] ??= SURVEY_AGENT_CONFIG as NonNullable<Config["agent"]>[string];
  },
};

const textOf = (parts: Part[]) =>
  (parts || [])
    .filter((p) => p?.type === "text" && p.text)
    .map((p) => p!.text)
    .join("\n")
    .trim();

// ── backfill: read opencode's past sessions ─────────────────────────────────────
// Same normalized JSON export every harness uses — kept here only so a real opencodeAdapter (used
// by index.ts) is a complete HarnessAdapter; the registry never routes through this file to get it
// (see harness/registry.ts's getHarness("opencode")), so this line pulling in "./registry" never
// drags @opencode-ai/plugin along for backfill's sake.
const chatReader = jsonChatReader("opencode");

/**
 * Adapt a harness-agnostic ToolSpec (the MCP-shaped spec every harness shares) to an opencode native
 * tool(). The spec's Zod raw shape IS opencode's `args` shape (both zod v4); its handler returns an
 * MCP `{content:[{text}]}` result which never throws, so we surface the joined text to opencode.
 */
function toOpencodeTool(spec: ToolSpec) {
  return tool({
    description: spec.description,
    // @opencode-ai/plugin bundles its own zod (v4.1.x); the project uses zod v4.4.x. The two are
    // runtime-compatible (same major), but their $ZodType brands differ, so TS rejects the
    // cross-instance assignment. Cast the raw shape to the plugin's expected `args` type — opencode
    // validates the args with its own zod at call time regardless.
    args: spec.inputSchema as unknown as Parameters<typeof tool>[0]["args"],
    async execute(args: Record<string, unknown>) {
      const r = await spec.handler(args);
      return r.content?.map((c) => c.text).join("\n") || "";
    },
  });
}

// ── runtime: opencode plugin hooks wired to the RuntimeCore ──────────────────────
function createRuntime(core: RuntimeCore) {
  // Register the full hindsight_* knowledge + recall suite natively (no MCP server needed).
  const tools: Record<string, ReturnType<typeof tool>> = {};
  for (const spec of core.toolSpecs()) tools[spec.name] = toOpencodeTool(spec);

  return {
    ...surveyAgentHook,
    tool: tools,
    // Each user turn: recall on the prompt; the injection it builds is pushed by system.transform.
    "chat.message": async (input: { sessionID?: string }, output: { parts: Part[] }) => {
      await core.onPrompt(input.sessionID, textOf(output.parts));
    },
    // Push this turn's injection (recalled memories + attribution/user-feedback framing, plus the
    // knowledge preamble on turn 1 and the roster refresh on cadence) into the system prompt every
    // turn. opencode fires this hook with NO sessionId (input is just `{model}`), so getInjection
    // falls back to the most recent turn's block (see RuntimeCore.getInjection).
    "experimental.chat.system.transform": async (
      input: { sessionID?: string },
      output: { system: string[] }
    ) => {
      const inj = core.getInjection(input.sessionID);
      if (inj) output.system.push(inj);
      diag(core.harness, inj ? "inject_ok" : "inject_empty", {
        chars: inj?.length ?? 0,
        hasSession: !!input.sessionID,
      });
    },
    // Write-back (on by default): normalize the live transcript to rich turns (text + tool calls +
    // their inline output) and hand it to core, which upserts every N user turns.
    //
    // NOTE this fires while the host BUILDS a request, so `messages` never includes the reply that
    // request is about to produce — on its own it always lags a turn. `session.idle` below closes
    // that gap; this stays as the mid-session cadence path.
    "experimental.chat.messages.transform": async (
      _input: unknown,
      output: { messages: OcMessage[] }
    ) => {
      if (!core.writeBackEnabled) return;
      const msgs = output.messages || [];
      const sid = opencodeSessionId(msgs);
      if (!sid) return;
      await core.onTranscript(sid, readOpencodeMessages(msgs));
    },
    // The Stop-equivalent these hosts otherwise lack: `session.idle` fires once the assistant has
    // finished, which is the only moment the completed exchange is readable. Without it a session's
    // last turn — usually its conclusion — was never retained at all.
    event: async (input: { event?: { type?: string; properties?: { sessionID?: string } } }) => {
      if (input?.event?.type !== "session.idle") return;
      const sid = input.event.properties?.sessionID;
      if (!sid) return;
      await core.onSessionIdle(sid);
    },
  };
}

export const opencodeAdapter: HarnessAdapter = { name: "opencode", chatReader, createRuntime };
