const ABSOLUTE_URL_PATTERN = /^[a-z][a-z\d+\-.]*:\/\//i;

function ensureLeadingSlash(path: string): string {
  if (!path) return "/";
  return path.startsWith("/") ? path : `/${path}`;
}

export function normalizeBasePath(value = process.env.NEXT_PUBLIC_BASE_PATH): string {
  if (!value || value === "/") return "";
  const withSlash = ensureLeadingSlash(value.trim());
  return withSlash.replace(/\/+$/, "");
}

export function withBasePath(path: string): string {
  if (ABSOLUTE_URL_PATTERN.test(path) || path.startsWith("//")) {
    return path;
  }

  const basePath = normalizeBasePath();
  const normalizedPath = ensureLeadingSlash(path);

  if (!basePath) {
    return normalizedPath;
  }
  if (
    normalizedPath === basePath ||
    normalizedPath.startsWith(`${basePath}/`) ||
    normalizedPath.startsWith(`${basePath}?`)
  ) {
    return normalizedPath;
  }

  return `${basePath}${normalizedPath}`;
}

// Validates a `returnTo` query parameter to prevent open-redirect attacks
// (e.g. ?returnTo=//evil.com, ?returnTo=javascript:...). Returns the fallback
// when the value isn't a safe same-origin app path. Leading control chars are
// stripped because browsers ignore them when resolving URLs.
export function sanitizeReturnTo(
  rawValue: string | null | undefined,
  fallback = "/dashboard"
): string {
  if (!rawValue) return fallback;
  // Strip leading C0 controls and space — the WHATWG URL parser ignores these,
  // so a value like " //evil.com" would still resolve off-origin if we didn't.
  // eslint-disable-next-line no-control-regex
  const trimmed = rawValue.replace(/^[\x00-\x20]+/, "");
  if (!trimmed.startsWith("/")) return fallback;
  if (trimmed.startsWith("//") || trimmed.startsWith("/\\") || ABSOLUTE_URL_PATTERN.test(trimmed)) {
    return fallback;
  }
  return stripBasePath(trimmed);
}

export function stripBasePath(path: string): string {
  if (ABSOLUTE_URL_PATTERN.test(path) || path.startsWith("//")) {
    return path;
  }

  const basePath = normalizeBasePath();
  const normalizedPath = ensureLeadingSlash(path);

  if (!basePath) {
    return normalizedPath;
  }

  const match = normalizedPath.match(/^([^?#]*)(.*)$/);
  const pathname = match?.[1] || "/";
  const suffix = match?.[2] || "";

  if (pathname === basePath) {
    return `/${suffix}`;
  }
  if (pathname.startsWith(`${basePath}/`)) {
    return `${pathname.slice(basePath.length)}${suffix}`;
  }

  return normalizedPath;
}
