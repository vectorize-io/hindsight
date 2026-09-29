import { INodeParams, INodeCredential } from "../src/Interface";

class HindsightApi implements INodeCredential {
  label: string;
  name: string;
  version: number;
  description: string;
  inputs: INodeParams[];

  constructor() {
    this.label = "Hindsight API";
    this.name = "hindsightApi";
    this.version = 1.0;
    this.description =
      "Credentials for the Hindsight memory API. Defaults to Hindsight Cloud; change the URL for self-hosted instances.";
    this.inputs = [
      {
        label: "API URL",
        name: "apiUrl",
        type: "string",
        default: "https://api.hindsight.vectorize.io",
        description:
          "Base URL of the Hindsight API. Defaults to Hindsight Cloud; change for self-hosted instances (e.g. http://localhost:8888).",
      },
      {
        label: "API Key",
        name: "apiKey",
        type: "password",
        description:
          'API key for Hindsight Cloud (begins with "hsk_"). Leave blank for unauthenticated self-hosted instances.',
        optional: true,
      },
    ];
  }
}

module.exports = { credClass: HindsightApi };
