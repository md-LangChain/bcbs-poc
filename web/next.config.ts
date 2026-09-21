import path from "node:path";
import type { NextConfig } from "next";

// The repo root .env is the single source of env for the whole POC (agents and
// UI both). Next only reads .env files inside web/, so pull the root one in.
try {
  process.loadEnvFile(path.join(process.cwd(), "..", ".env"));
} catch {
  // No root .env yet — fall back to web/.env.local or the ambient environment.
}

const nextConfig: NextConfig = {
  devIndicators: false,
};

export default nextConfig;
