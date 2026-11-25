import { proxySessionRequest } from "@/lib/sessions-proxy";

type Params = { params: Promise<{ bankId: string; sessionId: string }> };

export async function GET(request: Request, { params }: Params) {
  const { bankId, sessionId } = await params;
  return proxySessionRequest(
    request,
    bankId,
    `/sessions/${encodeURIComponent(sessionId)}`,
    "GET",
    "get"
  );
}
