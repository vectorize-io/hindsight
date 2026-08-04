import { NextRequest } from "next/server";
import { sdk, getLowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ chunkId: string }> }
) {
  const { chunkId } = await params;
  const response = await sdk.getChunk({
    client: await getLowLevelClient(),
    path: { chunk_id: chunkId },
  });
  return respondWithSdk(response, "Failed to fetch chunk", { request });
}
