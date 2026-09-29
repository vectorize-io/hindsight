/**
 * A local stub model server for harnesses whose subscription credentials cannot be handed to a
 * container (Cursor, Copilot, Claude Code, Kilo keep them in a keyring or an account-bound session).
 *
 * It ECHOES back the text it was sent. That is the whole trick: the E2E seeds a decision that only
 * reaches the model through Hindsight's injection, so if the echoed reply contains the seeded
 * statuses, the memory demonstrably arrived in the model request. The existing assertion therefore
 * works unchanged — and for a sharper reason than with a real model, since a live model could in
 * principle guess "429" from general knowledge, whereas an echo can only repeat what was injected.
 *
 * What this does NOT prove is that a model reasons over the memory; the real-subscription runs
 * (codex, opencode, cline) still cover that. Use this to test OUR wiring, not model behaviour.
 */
import { createServer, type IncomingMessage, type Server } from "node:http";

export interface StubModel {
  /** Base URL reachable FROM THE CONTAINER (host.docker.internal), without a trailing slash. */
  containerUrl: string;
  /** Number of completion requests served — 0 means the CLI never reached the stub. */
  requests(): number;
  close(): Promise<void>;
}

/** Pull every piece of text out of an OpenAI- or Anthropic-shaped request body. */
function collectText(body: unknown): string {
  const out: string[] = [];
  const visit = (v: unknown): void => {
    if (typeof v === "string") {
      out.push(v);
    } else if (Array.isArray(v)) {
      for (const item of v) visit(item);
    } else if (v && typeof v === "object") {
      for (const [key, value] of Object.entries(v as Record<string, unknown>)) {
        // Skip fields that are configuration rather than conversation, so the echo stays the
        // conversation itself (model ids and tool schemas would swamp it).
        if (["model", "tools", "tool_choice", "metadata"].includes(key)) continue;
        visit(value);
      }
    }
  };
  visit(body);
  return out.join("\n");
}

const STUB_MODEL = "hindsight-e2e-stub";

/** OpenAI Responses-API body — the shape Cursor expects even when posting to /chat/completions. */
function responsesBody(text: string): Record<string, unknown> {
  return {
    id: "resp_stub",
    object: "response",
    status: "completed",
    model: STUB_MODEL,
    output: [
      {
        id: "msg_stub",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text, annotations: [] }],
      },
    ],
    output_text: text,
    usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
  };
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString("utf8");
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

/**
 * Start the stub on an ephemeral port. Speaks the two request shapes these CLIs use: OpenAI
 * chat-completions and Anthropic messages, plus the model-list endpoints they probe at startup.
 */
export async function startStubModel(): Promise<StubModel> {
  let served = 0;
  const server: Server = createServer((req, res) => {
    void (async () => {
      const url = req.url || "";
      const json = (status: number, payload: unknown): void => {
        const text = JSON.stringify(payload);
        res.writeHead(status, {
          "content-type": "application/json",
          "content-length": Buffer.byteLength(text),
        });
        res.end(text);
      };

      if (req.method === "GET" && url.includes("/models")) {
        json(200, {
          object: "list",
          data: [{ id: STUB_MODEL, object: "model", owned_by: "hindsight" }],
        });
        return;
      }

      const body = await readBody(req);
      const echo = collectText(body).slice(0, 20_000);
      served++;

      const asRecord = (body ?? {}) as Record<string, unknown>;
      // Cursor posts Responses-API payloads even to /chat/completions, so detect by SHAPE (an
      // `input` field instead of `messages`) rather than by path alone.
      const wantsResponses = url.includes("/responses") || "input" in asRecord;
      const wantsAnthropic = url.includes("/messages") && !wantsResponses;

      if (asRecord.stream === true) {
        // These CLIs stream by default and simply hang on a single JSON body — Cursor sat until the
        // run timed out before this existed.
        res.writeHead(200, {
          "content-type": "text/event-stream",
          "cache-control": "no-cache",
          connection: "keep-alive",
        });
        const send = (event: string | null, data: unknown): void => {
          if (event) res.write(`event: ${event}\n`);
          res.write(`data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`);
        };

        if (wantsAnthropic) {
          const message = {
            id: "msg_stub",
            type: "message",
            role: "assistant",
            model: STUB_MODEL,
            content: [],
            stop_reason: null,
            usage: { input_tokens: 1, output_tokens: 1 },
          };
          send("message_start", { type: "message_start", message });
          send("content_block_start", {
            type: "content_block_start",
            index: 0,
            content_block: { type: "text", text: "" },
          });
          send("content_block_delta", {
            type: "content_block_delta",
            index: 0,
            delta: { type: "text_delta", text: echo },
          });
          send("content_block_stop", { type: "content_block_stop", index: 0 });
          send("message_delta", {
            type: "message_delta",
            delta: { stop_reason: "end_turn" },
            usage: { output_tokens: 1 },
          });
          send("message_stop", { type: "message_stop" });
        } else if (wantsResponses) {
          send("response.created", {
            type: "response.created",
            response: { id: "resp_stub", status: "in_progress" },
          });
          send("response.output_text.delta", { type: "response.output_text.delta", delta: echo });
          send("response.completed", { type: "response.completed", response: responsesBody(echo) });
        } else {
          send(null, {
            id: "chatcmpl-stub",
            object: "chat.completion.chunk",
            model: STUB_MODEL,
            choices: [
              { index: 0, delta: { role: "assistant", content: echo }, finish_reason: null },
            ],
          });
          send(null, {
            id: "chatcmpl-stub",
            object: "chat.completion.chunk",
            model: STUB_MODEL,
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          });
          send(null, "[DONE]");
        }
        res.end();
        return;
      }

      if (wantsResponses) {
        json(200, responsesBody(echo));
        return;
      }

      if (wantsAnthropic) {
        // Anthropic shape (Claude Code).
        json(200, {
          id: "msg_hindsight_e2e_stub",
          type: "message",
          role: "assistant",
          model: STUB_MODEL,
          content: [{ type: "text", text: echo }],
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        });
        return;
      }

      // OpenAI chat-completions shape (everything else).
      json(200, {
        id: "chatcmpl-hindsight-e2e-stub",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: STUB_MODEL,
        choices: [
          { index: 0, message: { role: "assistant", content: echo }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      });
    })();
  });

  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;

  return {
    // The container reaches the host through this alias, already added to every docker run.
    containerUrl: `http://host.docker.internal:${port}`,
    requests: () => served,
    close: () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
      }),
  };
}
