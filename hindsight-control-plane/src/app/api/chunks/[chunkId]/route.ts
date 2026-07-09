import { NextRequest } from "next/server";
import { sdk, createDataplaneClientForRequest } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ chunkId: string }> }
) {
  const { chunkId } = await params;
  const response = await sdk.getChunk({
    client: createDataplaneClientForRequest(request),
    path: { chunk_id: chunkId },
  });
  return respondWithSdk(response, "Failed to fetch chunk", { request });
}
