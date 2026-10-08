/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep the dev server output separate from production build artifacts.
  distDir: process.env.NODE_ENV === "development" ? ".next-dev" : ".next",
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
  }
};

module.exports = nextConfig;
