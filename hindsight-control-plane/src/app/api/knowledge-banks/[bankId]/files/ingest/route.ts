import { streamUpload } from "@/lib/knowledge-stream";

/** Streamed, not sent through the SDK: the SDK would buffer the whole multipart body. */
export async function POST(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  return streamUpload(request, [bankId, "files", "ingest"]);
}
