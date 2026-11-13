import type { NextConfig } from "next";
import path from "path";
import createNextIntlPlugin from "next-intl/plugin";
import { maxUploadBytes } from "./src/lib/upload-limit";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

// Use a port-scoped distDir so multiple dev instances don't collide on the lock file
const distDir = process.env.PORT && process.env.PORT !== '9999'
  ? `.next-${process.env.PORT}`
  : '.next';

// Maximum request body size (see src/lib/upload-limit.ts): Next.js truncates a longer body
// that passes through middleware (default 10MB), so the middleware refuses one with a 413.
const maxUploadBodySize = maxUploadBytes();

const nextConfig: NextConfig = {
  output: 'standalone',
  distDir,
  basePath: basePath,
  assetPrefix: basePath,
  // Disable request logging in production
  logging: false,
  experimental: {
    proxyClientMaxBodySize: maxUploadBodySize,
  },
  // Set the monorepo root explicitly to avoid detecting wrong lockfiles in parent directories
  turbopack: {
    root: path.resolve(__dirname, '..'),
  },
};

export default withNextIntl(nextConfig);
