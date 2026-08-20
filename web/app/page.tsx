import { readJson, listDir, fmtScore, scoreColor, type Summary, type Results, type RetrievalEval, type Schema } from "@/lib/data";
import Link from "next/link";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const summary = readJson<Summary>("evals/summary.json");
  const results = readJson<Results>("evals/results.json");
  const retrieval = readJson<RetrievalEval>("evals/retrieval.json");
  const schema = readJson<Schema>("shared/schema.json");
  const docIds = listDir("data/golden").filter((d) => {
    // Only show dirs that have source.md + golden.json
    return readJson(`data/golden/${d}/golden.json`) !== null;
  });

  const hasData = summary && Object.keys(summary).length > 0;
  const hasRetrieval = retrieval && retrieval.strategies.length > 0;

  return (
    <div className="container">
      <h1>Clinical RAG Eval Harness</h1>
      <p className="subtitle">
        Retrieval and extraction evaluation on real clinical trial protocols.
        {schema && ` ${schema.documents.length} documents, `}
        {hasData && ` ${Object.keys(summary!).length} models scored.`}
      </p>

      {!hasData && !hasRetrieval && (
        <div className="empty">
          <p>No evaluation results found.</p>
          <p style={{ marginTop: 8, fontSize: "0.8rem" }}>
            Run <code>python -m harness.cli eval</code> and{" "}
            <code>python -m harness.cli retrieval-eval</code> to generate scores.
          </p>
        </div>
      )}

      {hasData && (
        <>
          <h2>Extraction Scorecard</h2>
          <p className="subtitle">
            LLM field extraction vs golden answers. Per-field precision / recall / F1
            across {Object.keys(summary!).length} models and {docIds.length} clinical trial protocols.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="numeric">Precision</th>
                  <th className="numeric">Recall</th>
                  <th className="numeric">F1</th>
                  <th className="numeric">Docs</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary!)
                  .sort(([, a], [, b]) => b.f1 - a.f1)
                  .map(([model, agg]) => {
                    const barWidth = `${Math.round(agg.f1 * 100)}px`;
                    return (
                      <tr key={model}>
                        <td>
                          <code>{model}</code>
                        </td>
                        <td className="numeric">{fmtScore(agg.precision)}</td>
                        <td className="numeric">{fmtScore(agg.recall)}</td>
                        <td className="numeric">
                          <span className={`score-bar ${scoreColor(agg.f1)}`} style={{ width: barWidth, marginRight: 6 }} />
                          {fmtScore(agg.f1)}
                        </td>
                        <td className="numeric">{agg.n_documents}</td>
                        <td>
                          {results && results[model] && (
                            <Link className="btn btn-sm" href={`/documents/${model.replace(/[:/]/g, "-")}`}>
                              Fields
                            </Link>
                          )}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {hasRetrieval && (
        <>
          <h2>Retrieval Scorecard</h2>
          <p className="subtitle">
            BM25 vs dense vs hybrid (RRF) retrieval. Hit@k, recall@k, MRR with k={retrieval!.config.k}.
            {" "}
            <Link href="/api/retrieval" style={{ fontSize: "0.85rem" }}>Debug view</Link>
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th className="numeric">Hit@k</th>
                  <th className="numeric">Recall@k</th>
                  <th className="numeric">MRR</th>
                  <th className="numeric">Queries</th>
                </tr>
              </thead>
              <tbody>
                {retrieval!.strategies.map((s) => (
                  <tr key={s.name}>
                    <td><code>{s.name}</code></td>
                    <td className="numeric">{fmtScore(s.metrics.hit_at_k)}</td>
                    <td className="numeric">{fmtScore(s.metrics.recall_at_k)}</td>
                    <td className="numeric">{fmtScore(s.metrics.mrr)}</td>
                    <td className="numeric">{s.metrics.n_queries}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>Per-document retrieval</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Doc</th>
                  <th className="numeric">Hit@k</th>
                  <th className="numeric">Recall@k</th>
                  <th className="numeric">MRR</th>
                </tr>
              </thead>
              <tbody>
                {retrieval!.strategies.map((s) =>
                  Object.entries(s.per_doc).map(([docId, m]) => (
                    <tr key={`${s.name}-${docId}`}>
                      <td><code>{s.name}</code></td>
                      <td>
                        <Link href={`/documents/${docId}`}>{docId}</Link>
                      </td>
                      <td className="numeric">{fmtScore(m.hit_at_k)}</td>
                      <td className="numeric">{fmtScore(m.recall_at_k)}</td>
                      <td className="numeric">{fmtScore(m.mrr)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h2>Documents</h2>
      <p className="subtitle">Golden corpus: real clinical trial protocols from ClinicalTrials.gov</p>
      {docIds.map((docId) => {
        const docSchema = schema?.documents.find((d) => d.doc_id === docId);
        return (
          <div key={docId} className="card">
            <Link href={`/documents/${docId}`}>
              <strong>{docId}</strong>
              {docSchema && <span className="subtitle"> - {docSchema.title}</span>}
            </Link>
            {docSchema && (
              <div className="subtitle">
                {docSchema.disease_area} | NCT: {docSchema.nct_id} |{" "}
                {Object.keys(docSchema.fields).length} fields
              </div>
            )}
          </div>
        );
      })}

      <h2>Live Extraction</h2>
      <p className="subtitle">
        Run a model against any golden document in real time. Uses server-side API key.
      </p>
      <div className="card">
        <LiveRunForm models={hasData ? Object.keys(summary!) : ["glm-5.2"]} docIds={docIds} />
      </div>
    </div>
  );
}

function LiveRunForm({ models, docIds }: { models: string[]; docIds: string[] }) {
  return (
    <div>
      <form action="/api/run" method="POST" className="run-form">
        <select name="model" defaultValue={models[0]}>
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <select name="doc_id" defaultValue={docIds[0]}>
          {docIds.map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
        <button type="submit" className="btn btn-primary btn-sm">Run Extraction</button>
      </form>
      <p className="subtitle" style={{ marginTop: 8 }}>
        Submitting runs a live LLM call. Results appear below.
      </p>
    </div>
  );
}