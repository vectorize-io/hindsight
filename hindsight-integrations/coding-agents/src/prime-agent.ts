/**
 * Prime Agent (PrimeIntellect) 0.10+, the Rust rewrite.
 *
 * 0.10 removed the JS extension host the 0.9 integration ran in (prime-agent#3189), and it has no
 * lifecycle hooks either. What it still loads is a user-declared MCP server, which the session's
 * Python kernel starts on the first `mcp.call_tool("hindsight", …)` and keeps until the session
 * ends, in the session's own working directory. So the MCP server (src/mcp-server.ts) is the whole
 * runtime here: it serves the tools, and while it runs it writes the session back from the
 * transcript Prime keeps on disk. The global AGENTS.md block the installer writes is what asks the
 * agent to make that first call at the start of a task.
 *
 * Prime hands an MCP server only the env vars its registration names, by REFERENCE to the kernel's
 * environment. The kernel always sets `RLM_SESSION_ID` and `PRIME_AGENT_CODING_AGENT_DIR`, so the
 * installer forwards those two (as the names below); `RLM_SESSION_DIR` would be simpler but is
 * unset for an unsaved session, and an unresolvable reference stops the server from starting.
 */
import { statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { TransportTurn } from "./core/chat";
import type { RuntimeCore } from "./core/runtime";
import { readPiTranscript } from "./core/transcript-pi";

const HARNESS = "prime-agent";
export const PRIME_AGENT_HARNESS = HARNESS;

/** The env names the installer's `mcpServers.hindsight.env` maps the kernel's variables onto. */
export const PRIME_SESSION_ID_ENV = "HINDSIGHT_PRIME_SESSION_ID";
export const PRIME_AGENT_DIR_ENV = "HINDSIGHT_PRIME_AGENT_DIR";

export interface PrimeSession {
  id: string;
  /** The session's transcript: `<agentDir>/sessions/<id>.jsonl`, the 0.9 layout 0.10 kept. */
  file: string;
}

/** The session this server was started for; undefined for an unsaved session (empty id). */
export function primeSession(
  env: NodeJS.ProcessEnv = process.env,
  home = homedir()
): PrimeSession | undefined {
  const id = env[PRIME_SESSION_ID_ENV]?.trim();
  if (!id) return undefined;
  const agentDir = env[PRIME_AGENT_DIR_ENV]?.trim() || join(home, ".prime", "agent");
  // ponytail: ignores the `sessionDir` setting; read it from settings.json if anyone moves theirs.
  return { id, file: join(agentDir, "sessions", `${id}.jsonl`) };
}

/**
 * Write the session back while this server runs. Prime gives no turn-end signal, so the transcript
 * file is polled: whenever it changed and its last turn is the agent's reply, the runtime's idle
 * path re-reads it and retains it if it grew. Returns the stop function, which does one final write
 * (the last reply may land between two polls) and waits for the in-flight retains.
 */
export function startPrimeWriteBack(
  core: Pick<RuntimeCore, "setTranscriptSource" | "onSessionIdle" | "flush">,
  session: PrimeSession,
  intervalMs = 5_000
): () => Promise<void> {
  let turns: TransportTurn[] = [];
  core.setTranscriptSource(async () => turns);
  let seen = 0;
  const sync = async (final: boolean): Promise<void> => {
    let mtime: number;
    try {
      mtime = statSync(session.file).mtimeMs;
    } catch {
      return; // not written yet
    }
    if (mtime === seen && !final) return;
    seen = mtime;
    turns = readPiTranscript(session.file);
    // Mid-turn the last turn is the prompt or a tool call: recording it now would log a turn
    // before its reply. The next poll (or the final sync) catches it once the reply is in.
    // ponytail: a reply sent through a `keepToolInputs` tool is an assistant turn too, so with such
    // a rule a turn that keeps working after it reads as finished one poll early, and the usage of
    // its later tool calls can be missed. Mark kept turns in toolTurn if that ever matters.
    if (!final && turns.at(-1)?.role !== "assistant") return;
    await core.onSessionIdle(session.id);
  };
  let busy = Promise.resolve();
  const tick = (final: boolean) => (busy = busy.then(() => sync(final)));
  const timer = setInterval(() => void tick(false), intervalMs);
  return async () => {
    clearInterval(timer);
    await tick(true);
    await core.flush();
  };
}
