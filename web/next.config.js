/** @type {import('next').NextConfig} */
const nextConfig = {
  // The web viewer reads eval/data/schema JSON from web/evals, web/data,
  // web/shared (a committed snapshot of the repo-root artifacts). Vercel's
  // Root Directory is set to web/, so these are bundled with the app.
  // fs.readFileSync with a dynamic path is not statically traceable by nft,
  // so we declare the globs explicitly per route.
  outputFileTracingIncludes: {
    "/": ["./evals/**/*", "./data/**/*", "./shared/**/*"],
    "/documents/[id]": ["./evals/**/*", "./data/**/*", "./shared/**/*"],
    "/api/run": ["./evals/**/*", "./data/**/*", "./shared/**/*"],
    "/api/retrieval": ["./evals/**/*", "./data/**/*", "./shared/**/*"],
  },
};

module.exports = nextConfig;
