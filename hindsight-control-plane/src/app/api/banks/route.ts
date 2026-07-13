import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";
import { getSessionPrefix } from "@/lib/auth/session";
import { bankAllowed } from "@/lib/auth/tokens";
import { assertBankAllowed } from "@/lib/auth/bank-guard";

const HTTP_CREATED = 201;

type BankListEntry = { bank_id?: string };
type BankListData = { banks?: BankListEntry[] };

export async function GET(request: NextRequest) {
  const response = await sdk.listBanks({ client: lowLevelClient });

  const prefix = await getSessionPrefix(request);
  if (prefix && response.data) {
    const data = response.data as BankListData;
    if (Array.isArray(data.banks)) {
      response.data = {
        ...data,
        banks: data.banks.filter((bank) => bankAllowed(prefix, bank.bank_id ?? "")),
      } as typeof response.data;
    }
  }

  return respondWithSdk(response, "Failed to fetch banks", { request });
}

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Invalid JSON body",
        errorKey: "api.errors.auth.invalidRequestBody",
      }),
      { status: 400 }
    );
  }
  const { bank_id } = body;

  if (!bank_id) {
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "bank_id is required",
        errorKey: "api.errors.validation.bankIdRequired",
      }),
      { status: 400 }
    );
  }

  const forbidden = await assertBankAllowed(request, bank_id);
  if (forbidden) return forbidden;

  const response = await sdk.createOrUpdateBank({
    client: lowLevelClient,
    path: { bank_id },
    body: {},
  });
  return respondWithSdk(response, "Failed to create bank", HTTP_CREATED, { request });
}
