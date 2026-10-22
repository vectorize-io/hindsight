import { proxySessionRequest } from "@/lib/sessions-proxy";

export async function GET(request: Request, { params }: { params: Promise<{ bankId: string }> }) {
  const { bankId } = await params;
  return proxySessionRequest(request, bankId, "/sessions", "GET", "list");
}
