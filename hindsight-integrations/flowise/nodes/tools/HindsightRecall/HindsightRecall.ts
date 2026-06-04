import { DynamicStructuredTool } from "@langchain/core/tools";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { z } from "zod";

import { ICommonObject, INode, INodeData, INodeParams } from "../../../src/Interface";
import { getBaseClasses, getCredentialData, getCredentialParam } from "../../../src/utils";

const RecallSchema = z.object({
  bankId: z.string().describe("The Hindsight memory bank to search."),
  query: z.string().describe("Natural-language query describing what to recall."),
  budget: z
    .enum(["low", "mid", "high"])
    .optional()
    .describe("Recall budget: low/mid/high. Higher values search more memories at higher cost."),
  maxTokens: z
    .number()
    .int()
    .positive()
    .optional()
    .describe("Maximum tokens to return in the recall response."),
  tags: z.array(z.string()).optional().describe("Optional list of tags to filter the search by."),
});

class HindsightRecall_Tools implements INode {
  label: string;
  name: string;
  version: number;
  description: string;
  type: string;
  icon: string;
  category: string;
  baseClasses: string[];
  credential: INodeParams;
  inputs: INodeParams[];

  constructor() {
    this.label = "Hindsight Recall";
    this.name = "hindsightRecall";
    this.version = 1.0;
    this.type = "HindsightRecall";
    this.icon = "hindsight.png";
    this.category = "Tools";
    this.description =
      "Search a Hindsight memory bank for memories relevant to a query. Returns ranked results.";
    this.credential = {
      label: "Connect Credential",
      name: "credential",
      type: "credential",
      credentialNames: ["hindsightApi"],
    };
    this.inputs = [
      {
        label: "Default Bank ID",
        name: "bankId",
        type: "string",
        description:
          "Default memory bank to recall from when the agent does not pass one. Banks are created on first use.",
        placeholder: "user-123",
        optional: true,
      },
      {
        label: "Default Budget",
        name: "budget",
        type: "options",
        options: [
          { label: "Low", name: "low" },
          { label: "Mid", name: "mid" },
          { label: "High", name: "high" },
        ],
        default: "mid",
        optional: true,
        description: "Default recall budget when the agent does not pass one.",
      },
    ];
    this.baseClasses = [this.type, ...getBaseClasses(DynamicStructuredTool)];
  }

  async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<unknown> {
    const credentialData = await getCredentialData(nodeData.credential ?? "", options);
    const apiUrl =
      getCredentialParam("apiUrl", credentialData, nodeData) ||
      "https://api.hindsight.vectorize.io";
    const apiKey = getCredentialParam("apiKey", credentialData, nodeData);

    const defaultBankId = (nodeData.inputs?.bankId as string) || "";
    const defaultBudget = ((nodeData.inputs?.budget as string) || "mid") as "low" | "mid" | "high";

    const client = new HindsightClient({
      baseUrl: apiUrl,
      ...(apiKey ? { apiKey } : {}),
    });

    return new DynamicStructuredTool({
      name: "hindsight_recall",
      description:
        "Search a Hindsight memory bank with a natural-language query and return the most relevant memories. Use this before answering the user to ground your response in past context.",
      schema: RecallSchema as any,
      func: async ({
        bankId,
        query,
        budget,
        maxTokens,
        tags,
      }: {
        bankId: string;
        query: string;
        budget?: "low" | "mid" | "high";
        maxTokens?: number;
        tags?: string[];
      }) => {
        const targetBank = bankId || defaultBankId;
        if (!targetBank) {
          return "Error: bankId is required (no default bank configured on the node).";
        }
        const opts: Record<string, unknown> = {
          budget: budget || defaultBudget,
        };
        if (typeof maxTokens === "number") opts.maxTokens = maxTokens;
        if (tags && tags.length) opts.tags = tags;
        const response = await client.recall(targetBank, query, opts as any);
        return JSON.stringify(response);
      },
    });
  }
}

module.exports = { nodeClass: HindsightRecall_Tools };
