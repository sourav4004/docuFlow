import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output is required by the production Dockerfile
  // (COPY --from=builder /app/.next/standalone ./) and keeps images small.
  output: "standalone",
};

export default nextConfig;
