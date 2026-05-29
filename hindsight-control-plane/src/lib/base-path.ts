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
