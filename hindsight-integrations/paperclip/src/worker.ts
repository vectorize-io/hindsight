/**
 * paperclip-plugin-hindsight — worker entrypoint.
 *
 * Gives Paperclip agents persistent long-term memory via Hindsight.
 *
 * Lifecycle:
 *   agent.run.started      → fetch the run's issue, recall relevant memories,
 *                            cache them in plugin state for the run.
 *   issue.comment.created  → retain the full comment body to Hindsight (this is
 *                            the durable signal — it covers both user comments
 *                            and agent-authored comments).
 *   agent.run.finished     → no-op (Paperclip's lifecycle payload does not
 *                            carry run output; subscription kept for future use).
 *
 * Agent tools (callable mid-run):
 *   hindsight_recall(query)   → search memory, returns relevant context
 *   hindsight_retain(content) → store content in memory immediately
 */

import { definePlugin, runWorker } from "@paperclipai/plugin-sdk";
import type { ToolRunContext } from "@paperclipai/plugin-sdk";
import { HindsightClient, formatMemories } from "./client.js";
import { deriveBankId } from "./bank.js";

interface PluginConfig {
  hindsightApiUrl: string;
  hindsightApiKeyRef?: string;
  bankGranularity?: Array<"company" | "agent">;
  recallBudget?: "low" | "mid" | "high";
  autoRetain?: boolean;
}

interface RunStartedPayload {
  agentId: string;
  runId: string;
  issueId?: string | null;
}

interface RunFinishedPayload {
  agentId: string;
  runId: string;
}

interface CommentCreatedPayload {
  commentId?: string;
  bodySnippet?: string;
  agentId?: string | null;
  runId?: string | null;
}

async function getConfig(ctx: {
  config: { get(): Promise<Record<string, unknown>> };
}): Promise<PluginConfig> {
  return (await ctx.config.get()) as unknown as PluginConfig;
}

async function resolveApiKey(
  ctx: { secrets: { resolve(ref: string): Promise<string | null> } },
  config: PluginConfig
): Promise<string | undefined> {
  if (!config.hindsightApiKeyRef) return undefined;
  const resolved = await ctx.secrets.resolve(config.hindsightApiKeyRef);
  return resolved ?? undefined;
}

const plugin = definePlugin({
  async setup(ctx) {
    ctx.logger.info("Hindsight memory plugin starting");

    // ---------------------------------------------------------------------------
    // agent.run.started — recall memories using the issue's title + description
    //
    // Paperclip's lifecycle payload only includes runId/agentId/issueId/etc.,
    // so we fetch the issue via ctx.issues.get to obtain the title/description
    // we need for a meaningful recall query.
    // ---------------------------------------------------------------------------
    ctx.events.on("agent.run.started", async (event) => {
      const payload = event.payload as RunStartedPayload;
      const config = await getConfig(ctx);
      const { agentId, runId, issueId } = payload;
      const companyId = event.companyId;
      if (!issueId || !companyId) return;

      let issue;
      try {
        issue = await ctx.issues.get(issueId, companyId);
      } catch (err) {
        ctx.logger.warn("Failed to fetch issue for recall", {
          runId,
          issueId,
          error: String(err),
        });
        return;
      }
      if (!issue) return;

      const query = [issue.title, issue.description].filter(Boolean).join("\n");
      if (!query.trim()) return;

      try {
        const apiKey = await resolveApiKey(ctx, config);
        const client = new HindsightClient(config.hindsightApiUrl, apiKey);
        const bankId = deriveBankId({ companyId, agentId }, config);

        const response = await client.recall(bankId, query, config.recallBudget ?? "mid");

        const memories = formatMemories(response.results);
        if (memories) {
          await ctx.state.set(
            { scopeKind: "run", scopeId: runId, stateKey: "recalled-memories" },
            memories
          );
          ctx.logger.info("Recalled memories for run", {
            runId,
            bankId,
            count: response.results.length,
          });
        }
      } catch (err) {
        // Non-fatal: agent runs without memory context.
        ctx.logger.warn("Failed to recall memories on run start", {
          runId,
          error: String(err),
        });
      }
    });

    // ---------------------------------------------------------------------------
    // issue.comment.created — retain the full comment body
    //
    // Paperclip's comment-created event payload only carries a 120-char snippet,
    // so we fetch the full comment via ctx.issues.listComments. Comments are the
    // primary durable record of agent + user output, so this captures both.
    // ---------------------------------------------------------------------------
    ctx.events.on("issue.comment.created", async (event) => {
      const config = await getConfig(ctx);
      if (config.autoRetain === false) return;

      const companyId = event.companyId;
      const issueId = event.entityId;
      const payload = (event.payload ?? {}) as CommentCreatedPayload;
      const commentId = payload.commentId;
      const payloadAgentId = payload.agentId ?? null;
      if (!issueId || !companyId || !commentId) return;

      let body = "";
      try {
        const comments = await ctx.issues.listComments(issueId, companyId);
        const match = comments.find((c) => c.id === commentId);
        if (match && typeof match.body === "string") body = match.body;
      } catch (err) {
        // Fall back to the truncated snippet if listComments isn't available.
        if (typeof payload.bodySnippet === "string") {
          body = payload.bodySnippet;
        } else {
          ctx.logger.warn("Failed to fetch comment body", {
            commentId,
            error: String(err),
          });
          return;
        }
      }
      if (!body.trim()) return;

      // Bank attribution: comments authored by an agent belong in that agent's
      // bank; user/system comments fall back to the issue's assignee.
      let bankAgentId: string | null = payloadAgentId;
      if (!bankAgentId) {
        try {
          const issue = await ctx.issues.get(issueId, companyId);
          bankAgentId = issue?.assigneeAgentId ?? null;
        } catch {
          /* ignore */
        }
      }

      if (!bankAgentId) {
        ctx.logger.info("Skipping retain — no agent attribution available", {
          commentId,
          issueId,
        });
        return;
      }

      try {
        const apiKey = await resolveApiKey(ctx, config);
        const client = new HindsightClient(config.hindsightApiUrl, apiKey);
        const bankId = deriveBankId({ companyId, agentId: bankAgentId }, config);
        await client.retain(bankId, body, commentId, {
          agentId: bankAgentId,
          companyId,
          issueId,
          commentId,
        });
        ctx.logger.info("Retained comment to memory", { commentId, bankId });
      } catch (err) {
        ctx.logger.warn("Failed to retain comment", {
          commentId,
          error: String(err),
        });
      }
    });

    // ---------------------------------------------------------------------------
    // agent.run.finished — no-op
    //
    // Paperclip's run-lifecycle payload only includes status/timing fields and
    // does not carry the agent's output. Retention now flows through the
    // issue.comment.created handler above. The subscription is kept so the
    // plugin remains visible in the event subscription list and so future
    // payload additions (e.g. an output reference) can be picked up here.
    // ---------------------------------------------------------------------------
    ctx.events.on("agent.run.finished", async (event) => {
      const payload = event.payload as RunFinishedPayload;
      ctx.logger.debug(
        "agent.run.finished received (no-op; retention handled by issue.comment.created)",
        { runId: payload?.runId }
      );
    });

    // ---------------------------------------------------------------------------
    // Tool: hindsight_recall
    // ---------------------------------------------------------------------------
    ctx.tools.register(
      "hindsight_recall",
      {
        displayName: "Recall from Memory",
        description: "Search Hindsight long-term memory for context relevant to a query.",
        parametersSchema: {
          type: "object",
          required: ["query"],
          properties: {
            query: { type: "string", description: "What to search for" },
          },
        },
      },
      async (params: unknown, runCtx: ToolRunContext) => {
        const { query } = params as { query: string };
        const config = await getConfig(ctx);
        const bankId = deriveBankId(
          { companyId: runCtx.companyId, agentId: runCtx.agentId },
          config
        );

        // Return cached memories from run start if available
        const cached = await ctx.state.get({
          scopeKind: "run",
          scopeId: runCtx.runId,
          stateKey: "recalled-memories",
        });
        if (cached && typeof cached === "string") {
          return { content: cached };
        }

        // Live recall fallback
        try {
          const apiKey = await resolveApiKey(ctx, config);
          const client = new HindsightClient(config.hindsightApiUrl, apiKey);
          const response = await client.recall(bankId, query, config.recallBudget ?? "mid");
          const memories = formatMemories(response.results);
          return { content: memories || "No relevant memories found." };
        } catch (err) {
          return { content: `Memory recall failed: ${String(err)}` };
        }
      }
    );

    // ---------------------------------------------------------------------------
    // Tool: hindsight_retain
    // ---------------------------------------------------------------------------
    ctx.tools.register(
      "hindsight_retain",
      {
        displayName: "Save to Memory",
        description:
          "Store important facts, decisions, or outcomes in Hindsight long-term memory for future runs.",
        parametersSchema: {
          type: "object",
          required: ["content"],
          properties: {
            content: {
              type: "string",
              description: "The content to store in memory",
            },
          },
        },
      },
      async (params: unknown, runCtx: ToolRunContext) => {
        const { content } = params as { content: string };
        const config = await getConfig(ctx);
        const bankId = deriveBankId(
          { companyId: runCtx.companyId, agentId: runCtx.agentId },
          config
        );

        try {
          const apiKey = await resolveApiKey(ctx, config);
          const client = new HindsightClient(config.hindsightApiUrl, apiKey);
          await client.retain(bankId, content, undefined, {
            agentId: runCtx.agentId,
            companyId: runCtx.companyId,
            runId: runCtx.runId,
          });
          return { content: "Memory saved." };
        } catch (err) {
          return { content: `Failed to save memory: ${String(err)}` };
        }
      }
    );

    ctx.logger.info("Hindsight memory plugin ready");
  },

  async onHealth() {
    return { status: "ok" };
  },

  async onValidateConfig(config) {
    const c = config as Partial<PluginConfig>;
    if (!c.hindsightApiUrl?.trim()) {
      return { ok: false, errors: ["hindsightApiUrl is required"] };
    }

    try {
      const client = new HindsightClient(c.hindsightApiUrl);
      const healthy = await client.health();
      if (!healthy) {
        return {
          ok: false,
          errors: [`Cannot reach Hindsight at ${c.hindsightApiUrl}`],
        };
      }
    } catch (err) {
      return { ok: false, errors: [`Connection failed: ${String(err)}`] };
    }

    return { ok: true };
  },
});

export default plugin;
runWorker(plugin, import.meta.url);
