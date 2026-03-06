import type { NextConfig } from "next";
import path from "path";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

// Use a port-scoped distDir so multiple dev instances don't collide on the lock file
const distDir = process.env.PORT && process.env.PORT !== '9999'
  ? `.next-${process.env.PORT}`
  : '.next';

const nextConfig: NextConfig = {
  output: 'standalone',
  distDir,
  basePath: basePath,
  assetPrefix: basePath,
  // Disable request logging in production
  logging: false,
  // Set the monorepo root explicitly to avoid detecting wrong lockfiles in parent directories
  turbopack: {
    root: path.resolve(__dirname, '..'),
  },
};

export default nextConfig;
