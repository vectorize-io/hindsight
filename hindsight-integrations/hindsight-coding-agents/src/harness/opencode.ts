/**
 * opencode harness adapter.
 *
 * Maps opencode's plugin hooks onto the shared RuntimeCore, and reads opencode sessions for backfill.
 * This is the only opencode-specific file; everything it uses is in ../core.
 */
import { readFileSync } from "node:fs";
import { tool } from "@opencode-ai/plugin";
import type { RuntimeCore } from "../core/runtime";
import type { HarnessAdapter, ChatReader, ChatSession } from "../core/types";
import type { TransportTurn } from "../core/chat";

// opencode part/message shapes (structurally typed — avoids a hard dep on the plugin types here).
type Part = { type?: string; text?: string };
type OcMessage = {
  info?: { role?: string; sessionID?: string; time?: { created?: number } };
  parts: Part[];
};

const textOf = (parts: Part[]) =>
  (parts || [])
    .filter((p) => p?.type === "text" && p.text)
    .map((p) => p!.text)
    .join("\n")
    .trim();

// ── backfill: read opencode's past sessions ─────────────────────────────────────
const chatReader: ChatReader = {
  describe:
    "opencode sessions via a normalized JSON export " +
    "(--conversations file: [{ id, turns:[{role,text,timestamp?}] }])",
  async read(opts): Promise<ChatSession[]> {
    // Sessions are consumed as a normalized JSON export (the interchange format any exporter can emit).
    if (!opts.conversations) return [];
    return JSON.parse(readFileSync(opts.conversations, "utf8")) as ChatSession[];
  },
};

// ── runtime: opencode plugin hooks wired to the RuntimeCore ──────────────────────
function createRuntime(core: RuntimeCore) {
  return {
    // On-demand memory: the agent can query project memory itself, mid-task, for any symptom/question —
    // the same synthesized reflect that's auto-injected on the first message, but callable at will.
    tool: {
      memory_reflect: tool({
        description:
          "Query this project's long-term memory about why the code is the way it is. Given a bug " +
          "symptom, error, or design question, returns a synthesized root-cause answer drawn from THIS " +
          "repository's git history and past developer conversations, including exact rules/values and " +
          "REF-ID citations. Use it whenever you need the project's own rationale or a precise past " +
          "decision — not general knowledge. Phrase the query as the concrete problem you are facing.",
        args: {
          query: tool.schema
            .string()
            .describe(
              "The symptom, error, or question to reflect on, phrased as the problem you face."
            ),
        },
        async execute(args: { query: string }) {
          const ans = await core.reflectNow(args.query);
          return ans || "No relevant project memory found for that query.";
        },
      }),
    },
    // First task message: reflect on its symptom once; the surfaced decision is injected every turn.
    "chat.message": async (input: { sessionID?: string }, output: { parts: Part[] }) => {
      if (input.sessionID) await core.onTask(input.sessionID, textOf(output.parts));
    },
    // Push the surfaced decision into the system prompt (every turn, so it survives interventions).
    "experimental.chat.system.transform": async (
      input: { sessionID?: string },
      output: { system: string[] }
    ) => {
      const inj = core.getInjection(input.sessionID);
      if (inj) output.system.push(inj);
    },
    // Opt-in write-back: normalize the transcript to user/assistant text turns and hand it to core.
    "experimental.chat.messages.transform": async (
      _input: unknown,
      output: { messages: OcMessage[] }
    ) => {
      if (!core.writeBackEnabled) return;
      const msgs = output.messages || [];
      const sid = msgs.find((m) => m.info?.sessionID)?.info?.sessionID;
      if (!sid) return;
      const turns: TransportTurn[] = [];
      for (const m of msgs) {
        const role = m.info?.role;
        if (role !== "user" && role !== "assistant") continue; // drop non-conversational roles
        const text = textOf(m.parts); // text parts only => drops tool calls/comments
        if (!text) continue;
        const created = m.info?.time?.created; // per-turn timestamp (Unix ms) -> ISO
        turns.push({
          role,
          content: text,
          ...(created ? { timestamp: new Date(created).toISOString() } : {}),
        });
      }
      await core.onTranscript(sid, turns);
    },
  };
}

export const opencodeAdapter: HarnessAdapter = { name: "opencode", chatReader, createRuntime };
