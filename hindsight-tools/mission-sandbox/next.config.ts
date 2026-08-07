import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // core is consumed as a built package (dist); its heavy runtime deps stay external.
  serverExternalPackages: ["@google/genai", "@vectorize-io/hindsight-client"],
};

export default nextConfig;
