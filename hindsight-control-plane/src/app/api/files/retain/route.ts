import { NextRequest, NextResponse } from "next/server";
import { DATAPLANE_URL } from "@/lib/hindsight-client";

export async function POST(request: NextRequest) {
  try {
    // Clone the form data to read bank_id without consuming it
    const formData = await request.formData();

    // Extract bank_id from request JSON
    const requestJson = formData.get("request");
    if (!requestJson || typeof requestJson !== "string") {
      return NextResponse.json({ error: "Missing request data" }, { status: 400 });
    }

    const requestData = JSON.parse(requestJson);
    const bankId = requestData.bank_id;

    if (!bankId) {
      return NextResponse.json({ error: "Missing bank_id" }, { status: 400 });
    }

    // Use the shared dataplane URL configuration
    const url = `${DATAPLANE_URL}/v1/default/banks/${bankId}/files/retain`;

    // Forward the form data to the dataplane
    const response = await fetch(url, {
      method: "POST",
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
      { error: error instanceof Error ? error.message : "Failed to upload files" },
      { status: 500 }
    );
  }
}
