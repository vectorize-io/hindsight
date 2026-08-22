import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  client: "@hey-api/client-fetch",
  input: process.env.HINDSIGHT_OPENAPI_SPEC ?? "../../hindsight-docs/static/openapi.json",
  output: {
    path: "./generated",
    format: "prettier",
  },
  plugins: ["@hey-api/typescript", "@hey-api/sdk"],
});
