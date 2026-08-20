// Copies the eval/data/schema artifacts from the repo root into web/ at build
// time so Vercel's serverless file-tracing bundles them. The web viewer reads
// these via fs at runtime; on Vercel process.cwd() is the web/ dir, so the
// files must live inside web/. In local dev, getRepoRoot() walks up to the
// repo root instead, so this copy is only consumed by the production build.
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = join(__dirname, "..");
const repoRoot = join(webRoot, "..");

for (const dir of ["evals", "data", "shared"]) {
  const src = join(repoRoot, dir);
  const dst = join(webRoot, dir);
  if (!existsSync(src)) {
    console.warn(`[copy-data] skipping ${dir}: source missing`);
    continue;
  }
  rmSync(dst, { recursive: true, force: true });
  cpSync(src, dst, {
    recursive: true,
    filter: (s) => !s.endsWith("embeddings.jsonl"), // not needed by the viewer
  });
  console.log(`[copy-data] copied ${dir} -> web/${dir}`);
}
