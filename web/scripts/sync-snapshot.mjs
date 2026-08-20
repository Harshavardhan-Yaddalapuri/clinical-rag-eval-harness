// Syncs the repo-root eval/data/schema artifacts into web/ as a committed
// snapshot. Vercel's Root Directory is web/, so the viewer reads these from
// web/evals, web/data, web/shared. Run this after re-running evals so the
// deployed scorecard reflects the latest committed scores.
//
//   node web/scripts/sync-snapshot.mjs
//
// The embeddings cache (evals/embeddings.jsonl) is intentionally excluded:
// the viewer never reads it.
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
    console.warn(`[sync-snapshot] skipping ${dir}: source missing`);
    continue;
  }
  rmSync(dst, { recursive: true, force: true });
  cpSync(src, dst, {
    recursive: true,
    filter: (s) => !s.endsWith("embeddings.jsonl"),
  });
  console.log(`[sync-snapshot] copied ${dir} -> web/${dir}`);
}
