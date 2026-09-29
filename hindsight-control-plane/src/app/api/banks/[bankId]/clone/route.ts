import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

/**
 * Clone a bank into a new one.
 *
 * Proxies to POST /v1/default/banks/{bank_id}/clone, which runs the export and
 * the import back to back server-side and returns an operation to poll. The
 * scope flags are only forwarded when the caller set them, so the dataplane's
 * own defaults stay the single source of truth for what a clone carries.
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;
    const body = await request.json().catch(() => ({}));
    const targetBankId = body?.target_bank_id;
    if (!targetBankId || typeof targetBankId !== "string") {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "target_bank_id is required",
          errorKey: "api.errors.validation.targetBankIdRequired",
        }),
        { status: 400 }
      );
    }

    const qs = new URLSearchParams({ target_bank_id: targetBankId });
    for (const flag of ["include_data", "include_bank_config", "include_history"] as const) {
      if (typeof body?.[flag] === "boolean") {
        qs.set(flag, String(body[flag]));
      }
    }

    const response = await fetch(dataplaneBankUrl(bankId, `/clone?${qs.toString()}`), {
      method: "POST",
      headers: getDataplaneHeaders(),
    });

    const data = await response.json().catch(() => ({ detail: response.statusText }));
    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data, { status: 202 });
  } catch (error) {
    console.error("Error cloning bank:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to clone bank",
        errorKey: "api.errors.banks.clone",
      }),
      { status: 500 }
    );
  }
}
