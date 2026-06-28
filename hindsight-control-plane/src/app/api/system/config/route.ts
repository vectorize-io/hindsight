import { NextResponse } from "next/server";

export async function GET() {
  const config = {
    storage: {
      type: process.env.HINDSIGHT_API_FILE_STORAGE_TYPE || "native",
      s3_bucket: process.env.HINDSIGHT_API_FILE_STORAGE_S3_BUCKET || null,
      s3_region: process.env.HINDSIGHT_API_FILE_STORAGE_S3_REGION || null,
    },
    llm: {
      provider: process.env.HINDSIGHT_API_LLM_PROVIDER || "openai",
      model: process.env.HINDSIGHT_API_LLM_MODEL || "gpt-5-mini",
      prompt_cache_enabled: process.env.HINDSIGHT_API_LLM_PROMPT_CACHE_ENABLED === "true",
    },
    embeddings: {
      provider: process.env.HINDSIGHT_API_EMBEDDINGS_PROVIDER || "local",
      model: process.env.HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL || "BAAI/bge-small-en-v1.5",
      dimension: parseInt(process.env.HINDSIGHT_API_EMBEDDINGS_DIMENSION || "384"),
    },
    reranker: {
      provider: process.env.HINDSIGHT_API_RERANKER_PROVIDER || "local",
    },
    database: {
      type: process.env.HINDSIGHT_API_DATABASE_URL?.startsWith("pg0") ? "pg0 (embedded)" : "external",
    },
  };

  return NextResponse.json(config);
}
