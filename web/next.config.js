/** @type {import('next').NextConfig} */
const nextConfig = {
  // The web viewer lives in web/ but reads eval JSON from the repo root.
  // In dev, getRepoRoot() walks up to the repo root. In production, a prebuild
  // step copies evals/, data/, shared/ into web/ so the serverless bundle can
  // read them from process.cwd(). externalDir lets dev read the parent dir.
  experimental: {
    externalDir: true,
  },
  // Force-include the copied data artifacts in the serverless function bundle.
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
