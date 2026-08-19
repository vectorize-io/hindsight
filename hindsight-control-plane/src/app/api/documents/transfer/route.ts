import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { DATAPLANE_URL, dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

// The dataplane export is asynchronous (a large whole-bank export can pin the API,
// so the synchronous endpoint was removed). We hide that from the browser: submit
// the export, poll the operation to completion, then stream the finished archive
// back so the caller still gets a single zip download.
const EXPORT_POLL_INTERVAL_MS = 1500;
const EXPORT_TIMEOUT_MS = 5 * 60 * 1000;

/**
 * Export documents as a transfer ZIP archive.
 * Orchestrates the async dataplane export (submit -> poll -> download) and streams
 * the binary zip back to the browser.
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");
    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    const qs = new URLSearchParams();
    for (const id of searchParams.getAll("document_id")) {
      qs.append("document_id", id);
    }
    if (searchParams.get("include_observations") === "true") {
      qs.set("include_observations", "true");
    }

    // 1. Submit the async export operation.
    const submitSuffix = `/document-transfer/export${qs.toString() ? `?${qs.toString()}` : ""}`;
    const submitResponse = await fetch(dataplaneBankUrl(bankId, submitSuffix), {
      method: "POST",
      headers: getDataplaneHeaders(),
    });
    if (!submitResponse.ok) {
      const error = await submitResponse
        .json()
        .catch(() => ({ detail: submitResponse.statusText }));
      return NextResponse.json(error, { status: submitResponse.status });
    }
    const { operation_id: operationId } = await submitResponse.json();

    // 2. Poll the operation until it completes.
    const deadline = Date.now() + EXPORT_TIMEOUT_MS;
    let downloadUrl: string | undefined;
    for (;;) {
      const statusResponse = await fetch(dataplaneBankUrl(bankId, `/operations/${operationId}`), {
        headers: getDataplaneHeaders(),
      });
      if (!statusResponse.ok) {
        const error = await statusResponse
          .json()
          .catch(() => ({ detail: statusResponse.statusText }));
        return NextResponse.json(error, { status: statusResponse.status });
      }
      const status = await statusResponse.json();
      if (status.status === "completed") {
        downloadUrl = status.result_metadata?.download_url;
        break;
      }
      if (status.status === "failed" || status.status === "cancelled") {
        return NextResponse.json(
          localizeApiErrorPayload(request, {
            error: status.error_message || `Export ${status.status}`,
            errorKey: "api.errors.documents.export",
          }),
          { status: 500 }
        );
      }
      if (Date.now() >= deadline) {
        return NextResponse.json(
          localizeApiErrorPayload(request, {
            error: "Export timed out",
            errorKey: "api.errors.documents.export",
          }),
          { status: 504 }
        );
      }
      await new Promise((resolve) => setTimeout(resolve, EXPORT_POLL_INTERVAL_MS));
    }

    if (!downloadUrl) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "Export completed without a download URL",
          errorKey: "api.errors.documents.export",
        }),
        { status: 500 }
      );
    }

    // 3. Download the finished archive and stream it back to the browser.
    const response = await fetch(`${DATAPLANE_URL}${downloadUrl}`, {
      headers: getDataplaneHeaders(),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    const body = await response.arrayBuffer();
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": "application/zip",
        "Content-Disposition":
          response.headers.get("content-disposition") ||
          `attachment; filename="${bankId}-documents.zip"`,
      },
    });
  } catch (error) {
    console.error("Error exporting documents:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to export documents",
        errorKey: "api.errors.documents.export",
      }),
      { status: 500 }
    );
  }
}

/**
 * Import a transfer ZIP archive into a bank.
 * Proxies the multipart upload to POST /v1/default/banks/{bank_id}/document-transfer.
 */
export async function POST(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const bankId = searchParams.get("bank_id");
    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }
    const onConflict = searchParams.get("on_conflict") || "skip";

    const inForm = await request.formData();
    const file = inForm.get("file");
    if (!(file instanceof Blob)) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "file is required",
          errorKey: "api.errors.validation.fileRequired",
        }),
        { status: 400 }
      );
    }

    const outForm = new FormData();
    const filename = file instanceof File ? file.name : "transfer.zip";
    outForm.append("file", file, filename);

    const suffix = `/document-transfer?on_conflict=${encodeURIComponent(onConflict)}`;
    const response = await fetch(dataplaneBankUrl(bankId, suffix), {
      method: "POST",
      headers: getDataplaneHeaders(),
      body: outForm,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return NextResponse.json(error, { status: response.status });
    }

    return NextResponse.json(await response.json(), { status: 200 });
  } catch (error) {
    console.error("Error importing documents:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to import documents",
        errorKey: "api.errors.documents.import",
      }),
      { status: 500 }
    );
  }
}
