import { NextResponse } from "next/server";
import { localizeApiErrorPayload, type RequestLike } from "@/lib/i18n/api-errors";

const DEFAULT_UPSTREAM_STATUS = 502 as const;
const SUCCESS_STATUS = 200 as const;

type SdkErrorOptions = {
  request?: RequestLike;
  errorKey?: string;
};

const failureErrorKeys: Record<string, string> = {
  "Failed to get entity": "api.errors.entities.fetch",
  "Failed to regenerate entity observations": "api.errors.entities.regenerateObservations",
  "Failed to list entities": "api.errors.entities.list",
  "Failed to fetch entity graph": "api.errors.entities.graph",
  "Failed to fetch stats": "api.errors.stats.fetch",
  "Failed to batch retain async": "api.errors.memories.retainAsync",
  "Failed to fetch banks": "api.errors.banks.fetch",
  "Failed to create bank": "api.errors.banks.create",
  "Failed to recover consolidation": "api.errors.consolidation.recover",
  "Failed to update bank": "api.errors.banks.update",
  "Failed to delete bank": "api.errors.banks.delete",
  "Failed to trigger consolidation": "api.errors.consolidation.trigger",
  "Failed to fetch bank config": "api.errors.bankConfig.fetch",
  "Failed to update bank config": "api.errors.bankConfig.update",
  "Failed to reset bank config": "api.errors.bankConfig.reset",
  "Failed to fetch documents": "api.errors.documents.fetchList",
  "Failed to get operation status": "api.errors.operations.status",
  "Failed to reflect": "api.errors.reflect.failed",
  "Failed to fetch document": "api.errors.documents.fetch",
  "Failed to delete document": "api.errors.documents.delete",
  "Failed to fetch chunk": "api.errors.chunks.fetch",
  "Failed to fetch operations": "api.errors.operations.fetch",
  "Failed to cancel operation": "api.errors.operations.cancel",
  "Failed to fetch bank profile": "api.errors.bankProfile.fetch",
  "Failed to update bank profile": "api.errors.bankProfile.update",
};

/**
 * Minimal structural type for the @hey-api/client-fetch RequestResult shape
 * (success: `{data, error: undefined}`, failure: `{data: undefined, error}`,
 * both with `request`/`response`). Kept local so the helper has no compile-time
 * dependency on the generated SDK package and can be unit-tested without it.
 */
export type SdkResult<T> = {
  data?: T;
  error?: unknown;
  request?: Request;
  response?: Response;
};

/**
 * Serialize the result of an SDK call into a NextResponse.
 *
 * Why this exists: `NextResponse.json(result.data, {status: 200})` throws
 * `TypeError: Value is not JSON serializable` when `result.data` is `undefined`
 * — which is exactly what the @hey-api/client-fetch SDK returns on non-2xx
 * upstream responses (since it doesn't throw). The resulting TypeError gets
 * caught and logged as the failure, hiding the real upstream error and forcing
 * the response status to a hard-coded 500.
 *
 * This helper checks `result.error` / `result.data` first, surfaces the upstream
 * status and error detail in the response body, and only serializes `data` on
 * the success path.
 *
 * @param result        The SDK call return value (`await sdk.someMethod(...)`).
 * @param failureLabel  Short human-readable label for the operation, used in
 *                      both the log line and the response body's `error` field
 *                      (e.g. `"Failed to fetch stats"`).
 * @param successStatus HTTP status to use on the success path. Defaults to 200.
 *                      Pass `201` for create endpoints.
 */
export function respondWithSdk<T>(
  result: SdkResult<T>,
  failureLabel: string,
  successStatusOrOptions: number | SdkErrorOptions = SUCCESS_STATUS,
  options?: SdkErrorOptions
): NextResponse {
  const successStatus =
    typeof successStatusOrOptions === "number" ? successStatusOrOptions : SUCCESS_STATUS;
  const errorOptions =
    typeof successStatusOrOptions === "number" ? options : successStatusOrOptions;

  if (result.error !== undefined || result.data === undefined) {
    const upstreamStatus = result.response?.status ?? DEFAULT_UPSTREAM_STATUS;
    const errorKey = errorOptions?.errorKey ?? failureErrorKeys[failureLabel];
    console.error(`${failureLabel}:`, {
      upstreamStatus,
      upstreamError: result.error,
    });
    const payload = {
      error: failureLabel,
      ...(errorKey ? { errorKey } : {}),
      upstream: {
        status: upstreamStatus,
        detail: result.error ?? null,
      },
    };

    return NextResponse.json(
      errorKey ? localizeApiErrorPayload(errorOptions?.request, { ...payload, errorKey }) : payload,
      { status: upstreamStatus }
    );
  }

  return NextResponse.json(result.data, { status: successStatus });
}
