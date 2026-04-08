/**
 * Express middleware for Paperclip HTTP adapter agents.
 *
 * Automatically injects recalled memories into each request and
 * retains the agent's output after each response.
 *
 * Paperclip HTTP adapter request shape:
 * {
 *   runId: string,
 *   agentId: string,
 *   companyId: string,
 *   context: { taskId: string, taskDescription?: string, ... }
 * }
 */

import type { Request, Response, NextFunction } from 'express';
import type { PaperclipMemoryConfig } from './config.js';
import { recall } from './recall.js';
import { retain } from './retain.js';

/** Augmented request with Hindsight memory context. */
export interface HindsightRequest extends Request {
  hindsight: {
    memories: string;
    companyId: string;
    agentId: string;
    runId: string;
  };
}

/**
 * Create Express middleware that auto-recalls before each heartbeat
 * and auto-retains after each response.
 *
 * @example
 * ```typescript
 * import express from 'express'
 * import { createMemoryMiddleware, loadConfig } from '@vectorize-io/hindsight-paperclip'
 *
 * const app = express()
 * app.use(express.json())
 * app.use(createMemoryMiddleware(loadConfig()))
 *
 * app.post('/heartbeat', (req, res) => {
 *   const { memories, runId } = (req as HindsightRequest).hindsight
 *   const { context } = req.body
 *
 *   const prompt = memories
 *     ? `Past context:\n${memories}\n\nCurrent task: ${context.taskDescription}`
 *     : `Task: ${context.taskDescription}`
 *
 *   // ... run agent ...
 *   res.json({ output: agentOutput })  // auto-retained by middleware
 * })
 * ```
 */
export function createMemoryMiddleware(config: PaperclipMemoryConfig) {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    const { runId, agentId, companyId, context } = req.body ?? {};

    if (!agentId || !companyId) {
      next();
      return;
    }

    const query: string = context?.taskDescription ?? context?.taskTitle ?? '';

    // Pre-recall: inject memories into request
    const memories = await recall({ companyId, agentId, query }, config);
    (req as HindsightRequest).hindsight = {
      memories,
      companyId,
      agentId,
      runId: runId ?? '',
    };

    // Post-retain: wrap res.json to capture agent output
    const originalJson = res.json.bind(res) as (body: unknown) => Response;
    (res as Response).json = function (body: unknown): Response {
      // Fire-and-forget retain (don't block response)
      if (body && typeof body === 'object' && 'output' in body && runId) {
        const output = (body as { output: unknown }).output;
        if (typeof output === 'string' && output.trim()) {
          retain(
            { companyId, agentId, content: output, documentId: runId },
            config,
          ).catch((err) => {
            console.warn('[hindsight-paperclip] retain failed:', (err as Error).message);
          });
        }
      }
      return originalJson(body);
    };

    next();
  };
}
