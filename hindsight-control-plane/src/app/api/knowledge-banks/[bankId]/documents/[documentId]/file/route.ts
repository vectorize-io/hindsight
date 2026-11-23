import { streamBinary } from "@/lib/knowledge-stream";

/** Streamed, not sent through the SDK: an original can be hundreds of MB. */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; documentId: string }> }
) {
  const { bankId, documentId } = await params;
  return streamBinary(request, [bankId, "documents", documentId, "file"], "file");
}
