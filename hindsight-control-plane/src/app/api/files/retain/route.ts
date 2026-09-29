import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    // Clone the form data to read bank_id without consuming it
    const formData = await request.formData();

    // Extract bank_id from request JSON
    const requestJson = formData.get("request");
    if (!requestJson || typeof requestJson !== "string") {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Missing request data",
          errorKey: "api.errors.validation.missingRequestData",
        }),
        { status: 400 }
      );
    }

    let requestData: { bank_id?: string };
    try {
      requestData = JSON.parse(requestJson);
    } catch {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Invalid request body",
          errorKey: "api.errors.auth.invalidRequestBody",
        }),
        { status: 400 }
      );
    }

    const bankId = requestData.bank_id;

    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Missing bank_id",
          errorKey: "api.errors.validation.missingBankId",
        }),
        { status: 400 }
      );
    }

    // Use the shared dataplane URL configuration
    const url = dataplaneBankUrl(bankId, "/files/retain");

    // Forward the form data to the dataplane
    const response = await fetch(url, {
      method: "POST",
      headers: getDataplaneHeaders(),
      body: formData,
      // Don't set Content-Type - let fetch handle multipart boundary
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch {
        errorData = { error: errorText };
      }
      return NextResponse.json(errorData, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error("Error uploading files:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: error instanceof Error ? error.message : "Failed to upload files",
        errorKey: "api.errors.files.upload",
      }),
      { status: 500 }
    );
  }
}
