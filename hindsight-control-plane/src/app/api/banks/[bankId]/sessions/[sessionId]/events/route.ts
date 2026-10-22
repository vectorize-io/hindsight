import { proxySessionRequest } from "@/lib/sessions-proxy";

type Params = { params: Promise<{ bankId: string; sessionId: string }> };

/** The log after `?after=<seq>`, oldest first: the UI polls this to follow a session live. */
export async function GET(request: Request, { params }: Params) {
  const { bankId, sessionId } = await params;
  return proxySessionRequest(
    request,
    bankId,
    `/sessions/${encodeURIComponent(sessionId)}/events`,
    "GET",
    "events"
  );
}
