import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // fluent-ffmpeg and sharp need to run in Node.js, not the Edge runtime
  serverExternalPackages: ["fluent-ffmpeg", "sharp", "@ffprobe-installer/ffprobe"],
};

export default nextConfig;
