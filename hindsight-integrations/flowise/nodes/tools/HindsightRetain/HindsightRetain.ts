import { DynamicStructuredTool } from "@langchain/core/tools";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { z } from "zod";

import { ICommonObject, INode, INodeData, INodeParams } from "../../../src/Interface";
import { getBaseClasses, getCredentialData, getCredentialParam } from "../../../src/utils";

const RetainSchema = z.object({
  bankId: z.string().describe("The Hindsight memory bank to retain content into."),
  content: z
    .string()
    .describe("Free-text content to store. Hindsight extracts facts asynchronously."),
  tags: z
    .array(z.string())
    .optional()
    .describe("Optional list of tags to apply to the retained memory."),
});

class HindsightRetain_Tools implements INode {
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
    this.label = "Hindsight Retain";
    this.name = "hindsightRetain";
    this.version = 1.0;
    this.type = "HindsightRetain";
    this.icon = "hindsight.png";
    this.category = "Tools";
    this.description =
      "Store free-text content as long-term memory in a Hindsight bank. Hindsight extracts facts asynchronously after the call returns.";
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
          "Default memory bank to retain into when the agent does not pass one. Banks are created on first use.",
        placeholder: "user-123",
        optional: true,
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

    const client = new HindsightClient({
      baseUrl: apiUrl,
      ...(apiKey ? { apiKey } : {}),
    });

    return new DynamicStructuredTool({
      name: "hindsight_retain",
      description:
        "Store free-text content into a Hindsight memory bank. Use this to retain facts about a user, decisions made, or anything else worth remembering across conversations. Hindsight extracts structured facts asynchronously.",
      schema: RetainSchema as any,
      func: async ({
        bankId,
        content,
        tags,
      }: {
        bankId: string;
        content: string;
        tags?: string[];
      }) => {
        const targetBank = bankId || defaultBankId;
        if (!targetBank) {
          return "Error: bankId is required (no default bank configured on the node).";
        }
        const opts = tags && tags.length ? { tags } : undefined;
        const response = await client.retain(targetBank, content, opts);
        return JSON.stringify({ bankId: targetBank, success: true, response });
      },
    });
  }
}

module.exports = { nodeClass: HindsightRetain_Tools };
