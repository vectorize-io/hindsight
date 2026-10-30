import { NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

/**
 * Forwards one sessions request to the dataplane. The four session routes differ
 * only in path, method and error message, so they share this instead of each
 * repeating the audit-logs route's body.
 */
export async function proxySessionRequest(
  request: Request,
  bankId: string,
  suffix: string,
  method: "GET" | "POST",
  errorKey: "list" | "get" | "events"
): Promise<NextResponse> {
  const fallback = {
    list: "Failed to list sessions",
    get: "Failed to get session",
    events: "Failed to get session events",
  }[errorKey];
  try {
    const query = new URL(request.url).searchParams.toString();
    const response = await fetch(dataplaneBankUrl(bankId, `${suffix}${query ? `?${query}` : ""}`), {
      method,
      headers: getDataplaneHeaders(),
    });
    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        {
          ...localizeApiErrorPayload(request, {
            error: fallback,
            errorKey: `api.errors.sessions.${errorKey}`,
          }),
          detail: data.detail,
        },
        { status: response.status }
      );
    }
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error(`Error proxying session request (${errorKey}):`, error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: fallback,
        errorKey: `api.errors.sessions.${errorKey}`,
      }),
      { status: 500 }
    );
  }
}
