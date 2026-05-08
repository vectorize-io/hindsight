import type {
  IDataObject,
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from "n8n-workflow";
import { NodeConnectionTypes, NodeOperationError } from "n8n-workflow";

type Budget = "low" | "mid" | "high";

interface HindsightCredentials {
  apiUrl: string;
  apiKey?: string;
}

/**
 * Hindsight n8n node — retain, recall, and reflect on long-term memory
 * directly from any n8n workflow.
 *
 * Three operations:
 *  - Retain: Store a piece of content in a memory bank
 *  - Recall: Search a bank for memories relevant to a query
 *  - Reflect: Get an LLM-synthesized answer using the bank's memories
 *
 * HTTP calls go through n8n's built-in `httpRequestWithAuthentication` helper
 * so the package ships with zero runtime dependencies (required for n8n
 * Cloud verified-node distribution). The Bearer header is applied
 * automatically from the configured Hindsight API credential.
 */
export class Hindsight implements INodeType {
  description: INodeTypeDescription = {
    displayName: "Hindsight",
    name: "hindsight",
    icon: "file:hindsight.svg",
    group: ["transform"],
    version: 1,
    subtitle: '={{$parameter["operation"]}}',
    description: "Retain, recall, and reflect on long-term memory",
    defaults: {
      name: "Hindsight",
    },
    inputs: [NodeConnectionTypes.Main],
    outputs: [NodeConnectionTypes.Main],
    credentials: [
      {
        name: "hindsightApi",
        required: true,
      },
    ],
    properties: [
      {
        displayName: "Operation",
        name: "operation",
        type: "options",
        noDataExpression: true,
        options: [
          {
            name: "Retain",
            value: "retain",
            description: "Store content in a memory bank",
            action: "Retain content to a memory bank",
          },
          {
            name: "Recall",
            value: "recall",
            description: "Search a memory bank for relevant memories",
            action: "Recall memories from a bank",
          },
          {
            name: "Reflect",
            value: "reflect",
            description: "Get an LLM-synthesized answer using the bank",
            action: "Reflect on memories in a bank",
          },
        ],
        default: "retain",
      },

      // Bank ID — common to all operations
      {
        displayName: "Bank ID",
        name: "bankId",
        type: "string",
        default: "",
        required: true,
        description:
          "The Hindsight memory bank to operate on. A bank is created on first use if it does not exist.",
        placeholder: "user-123",
      },

      // === RETAIN ===
      {
        displayName: "Content",
        name: "content",
        type: "string",
        typeOptions: {
          rows: 4,
        },
        default: "",
        required: true,
        description:
          "The text to store in memory. Hindsight extracts facts asynchronously after retain.",
        displayOptions: {
          show: {
            operation: ["retain"],
          },
        },
      },
      {
        displayName: "Tags",
        name: "retainTags",
        type: "string",
        default: "",
        description:
          'Comma-separated tags applied to the stored memory (e.g. "user:alex,scope:profile")',
        displayOptions: {
          show: {
            operation: ["retain"],
          },
        },
      },

      // === RECALL ===
      {
        displayName: "Query",
        name: "recallQuery",
        type: "string",
        default: "",
        required: true,
        description: "Natural-language query to search memories with",
        displayOptions: {
          show: {
            operation: ["recall"],
          },
        },
      },
      {
        displayName: "Budget",
        name: "recallBudget",
        type: "options",
        options: [
          { name: "Low", value: "low" },
          { name: "Mid", value: "mid" },
          { name: "High", value: "high" },
        ],
        default: "mid",
        description: "Recall budget level — controls how exhaustive the retrieval is",
        displayOptions: {
          show: {
            operation: ["recall"],
          },
        },
      },
      {
        displayName: "Max Tokens",
        name: "recallMaxTokens",
        type: "number",
        default: 4096,
        description: "Maximum tokens of memory to return",
        displayOptions: {
          show: {
            operation: ["recall"],
          },
        },
      },
      {
        displayName: "Tags Filter",
        name: "recallTags",
        type: "string",
        default: "",
        description: "Comma-separated tags to filter recall (leave blank for no filter)",
        displayOptions: {
          show: {
            operation: ["recall"],
          },
        },
      },

      // === REFLECT ===
      {
        displayName: "Query",
        name: "reflectQuery",
        type: "string",
        default: "",
        required: true,
        description: "Question to answer using the bank's memories",
        displayOptions: {
          show: {
            operation: ["reflect"],
          },
        },
      },
      {
        displayName: "Budget",
        name: "reflectBudget",
        type: "options",
        options: [
          { name: "Low", value: "low" },
          { name: "Mid", value: "mid" },
          { name: "High", value: "high" },
        ],
        default: "mid",
        description: "Reflect budget level",
        displayOptions: {
          show: {
            operation: ["reflect"],
          },
        },
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    const credentials = (await this.getCredentials(
      "hindsightApi"
    )) as unknown as HindsightCredentials;
    const baseUrl = (credentials.apiUrl || "").replace(/\/$/, "");

    for (let i = 0; i < items.length; i++) {
      try {
        const operation = this.getNodeParameter("operation", i) as string;
        const bankId = this.getNodeParameter("bankId", i) as string;

        if (!bankId) {
          throw new NodeOperationError(this.getNode(), "bankId is required", { itemIndex: i });
        }

        let result: IDataObject;

        if (operation === "retain") {
          const content = this.getNodeParameter("content", i) as string;
          const tagsRaw = this.getNodeParameter("retainTags", i, "") as string;
          const tags = parseTags(tagsRaw);

          // Retain: POST /v1/default/banks/{bank_id}/memories
          // Body matches HindsightClient.retain() shape: { items: [{ content, tags? }] }
          const item: IDataObject = { content };
          if (tags.length) {
            item.tags = tags;
          }

          const response = (await this.helpers.httpRequestWithAuthentication.call(
            this,
            "hindsightApi",
            {
              method: "POST",
              url: `${baseUrl}/v1/default/banks/${encodeURIComponent(bankId)}/memories`,
              body: { items: [item] },
              json: true,
            }
          )) as IDataObject;
          result = response;
        } else if (operation === "recall") {
          const query = this.getNodeParameter("recallQuery", i) as string;
          const budget = this.getNodeParameter("recallBudget", i, "mid") as Budget;
          const maxTokens = this.getNodeParameter("recallMaxTokens", i, 4096) as number;
          const tagsRaw = this.getNodeParameter("recallTags", i, "") as string;
          const tags = parseTags(tagsRaw);

          // Recall: POST /v1/default/banks/{bank_id}/memories/recall
          // Body matches HindsightClient.recall() shape
          const body: IDataObject = {
            query,
            max_tokens: maxTokens,
            budget,
          };
          if (tags.length) {
            body.tags = tags;
          }

          const response = (await this.helpers.httpRequestWithAuthentication.call(
            this,
            "hindsightApi",
            {
              method: "POST",
              url: `${baseUrl}/v1/default/banks/${encodeURIComponent(bankId)}/memories/recall`,
              body,
              json: true,
            }
          )) as IDataObject;
          result = response;
        } else if (operation === "reflect") {
          const query = this.getNodeParameter("reflectQuery", i) as string;
          const budget = this.getNodeParameter("reflectBudget", i, "mid") as Budget;

          // Reflect: POST /v1/default/banks/{bank_id}/reflect
          // Body matches HindsightClient.reflect() shape
          const body: IDataObject = {
            query,
            budget,
          };

          const response = (await this.helpers.httpRequestWithAuthentication.call(
            this,
            "hindsightApi",
            {
              method: "POST",
              url: `${baseUrl}/v1/default/banks/${encodeURIComponent(bankId)}/reflect`,
              body,
              json: true,
            }
          )) as IDataObject;
          result = response;
        } else {
          throw new NodeOperationError(this.getNode(), `Unknown operation: ${operation}`, {
            itemIndex: i,
          });
        }

        returnData.push({ json: result, pairedItem: { item: i } });
      } catch (error) {
        if (this.continueOnFail()) {
          returnData.push({
            json: { error: (error as Error).message },
            pairedItem: { item: i },
          });
          continue;
        }
        throw error;
      }
    }

    return [returnData];
  }
}

function parseTags(raw: string): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
}
