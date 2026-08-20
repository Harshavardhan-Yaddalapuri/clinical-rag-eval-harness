import { NextResponse } from "next/server";
import { readJson, type RetrievalEval, type RetrievalQuery } from "@/lib/data";
import fs from "fs";
import path from "path";
import { getRepoRoot } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const retrieval = readJson<RetrievalEval>("evals/retrieval.json");

  if (!retrieval) {
    return NextResponse.json(
      { error: "No retrieval evaluation found. Run `python -m harness.cli retrieval-eval` first." },
      { status: 404 }
    );
  }

  // Enrich per_query with expected spans from golden retrieval.json
  const root = getRepoRoot();
  const goldenDir = path.join(root, "data", "golden");

  const enriched = retrieval.strategies.map((strategy) => ({
    ...strategy,
    per_query: strategy.per_query.map((pq) => {
      const docId = pq.query_id.split("-")[0];
      const retrievalPath = path.join(goldenDir, docId, "retrieval.json");
      let expectedSpans: RetrievalQuery["expected_spans"] = [];
      let rationale = "";
      if (fs.existsSync(retrievalPath)) {
        try {
          const queries: RetrievalQuery[] = JSON.parse(fs.readFileSync(retrievalPath, "utf-8"));
          const match = queries.find((q) => q.id === pq.query_id);
          if (match) {
            expectedSpans = match.expected_spans;
            rationale = match.rationale;
          }
        } catch {
          // ignore parse errors
        }
      }
      return {
        ...pq,
        expected_spans: expectedSpans,
        rationale,
      };
    }),
  }));

  return NextResponse.json({
    config: retrieval.config,
    strategies: enriched,
  });
}