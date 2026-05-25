import { NextResponse } from "next/server";

const DEFAULT_UPSTREAM_STATUS = 502 as const;
const SUCCESS_STATUS = 200 as const;

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
  successStatus: number = SUCCESS_STATUS
): NextResponse {
  if (result.error !== undefined || result.data === undefined) {
    const upstreamStatus = result.response?.status ?? DEFAULT_UPSTREAM_STATUS;
    console.error(`${failureLabel}:`, {
      upstreamStatus,
      upstreamError: result.error,
    });
    return NextResponse.json(
      {
        error: failureLabel,
        upstream: {
          status: upstreamStatus,
          detail: result.error ?? null,
        },
      },
      { status: upstreamStatus }
    );
  }

  return NextResponse.json(result.data, { status: successStatus });
}
