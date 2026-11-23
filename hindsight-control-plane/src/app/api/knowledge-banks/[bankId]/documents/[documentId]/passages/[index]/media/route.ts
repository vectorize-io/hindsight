import { streamBinary } from "@/lib/knowledge-stream";

/** Not sent through the SDK: the clip is served with byte ranges, so <video> can seek. */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; documentId: string; index: string }> }
) {
  const { bankId, documentId, index } = await params;
  return streamBinary(
    request,
    [bankId, "documents", documentId, "passages", index, "media"],
    "media"
  );
}
