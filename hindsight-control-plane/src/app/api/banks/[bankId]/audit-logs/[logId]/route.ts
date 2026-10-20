import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ bankId: string; logId: string }> }
) {
  const { bankId, logId } = await params;
  const response = await sdk.getAuditLog({
    client: lowLevelClient,
    path: { bank_id: bankId, log_id: logId },
  });
  return respondWithSdk(response, "Failed to get audit log entry", {
    request,
    errorKey: "api.errors.auditLogs.get",
  });
}
