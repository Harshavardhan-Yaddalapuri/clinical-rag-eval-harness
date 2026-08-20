import { readJson, readText, listDir, fmtVal, fmtScore, type Results, type Schema, type SchemaDoc, type RetrievalQuery, type RetrievalEval } from "@/lib/data";
import Link from "next/link";

export const dynamic = "force-dynamic";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function DocumentPage({ params }: PageProps) {
  const { id } = await params;

  // Two cases: id is a doc_id (actt1) or a model name (glm-5.2 -> glm-5.2)
  // If it's a model name (has results), show field comparison for that model
  // If it's a doc_id, show source + schema + retrieval queries

  const schema = readJson<Schema>("shared/schema.json");
  const results = readJson<Results>("evals/results.json");
  const retrieval = readJson<RetrievalEval>("evals/retrieval.json");

  // Decode model name (dashes back to colons/slashes for lookup)
  // Model files are stored as e.g. "glm-5.2", "deepseek-v4-pro-0813"
  const possibleModelKeys = [id, id.replace(/-/g, ":"), id.replace(/-/g, "/")];

  // Check if this is a model-based field view (linked from scorecard)
  let modelResult = null;
  let modelKey = "";
  for (const key of possibleModelKeys) {
    if (results && results[key]) {
      modelResult = results[key];
      modelKey = key;
      break;
    }
  }

  if (modelResult) {
    return <ModelFieldsView modelKey={modelKey} modelResult={modelResult} schema={schema} />;
  }

  // Otherwise, treat as a doc_id
  const docSchema = schema?.documents.find((d) => d.doc_id === id);
  const sourceMd = readText(`data/golden/${id}/source.md`);
  const golden = readJson<Record<string, unknown>>(`data/golden/${id}/golden.json`);
  const retrievalQueries = readJson<RetrievalQuery[]>(`data/golden/${id}/retrieval.json`);

  if (!sourceMd && !docSchema && !golden) {
    return (
      <div className="container">
        <div className="back-link"><Link href="/">&larr; Back</Link></div>
        <div className="empty">
          <p>Document or model &quot;{id}&quot; not found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="back-link"><Link href="/">&larr; Back to scorecard</Link></div>

      <h1>{id}</h1>
      {docSchema && (
        <p className="subtitle">
          {docSchema.title} | {docSchema.disease_area} |{" "}
          <a href={docSchema.source_url} target="_blank" rel="noopener noreferrer">
            {docSchema.nct_id}
          </a>
        </p>
      )}

      {golden && (
        <>
          <h2>Golden Extraction ({Object.keys(golden).length} fields)</h2>
          <div className="field-grid">
            {Object.entries(golden).map(([field, value]) => (
              <div key={field} className="field-row">
                <div className="field-name">{field}</div>
                <div className="field-values">
                  <div className="field-value field-gold">{fmtVal(value)}</div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {docSchema && (
        <>
          <h2>Schema ({Object.keys(docSchema.fields).length} fields)</h2>
          <div className="schema-fields">
            {Object.entries(docSchema.fields).map(([name, cfg]) => (
              <div key={name} className="schema-field">
                <code>{name}</code>{" "}
                <span className="schema-field-type">({cfg.type})</span>
                {" - "}{cfg.description}
                {cfg.valid_values && (
                  <span className="schema-field-type"> [{cfg.valid_values.join(", ")}]</span>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {retrievalQueries && retrievalQueries.length > 0 && (
        <>
          <h2>Retrieval Queries ({retrievalQueries.length})</h2>
          <p className="subtitle">
            Hard queries requiring cross-section synthesis. Each has ground-truth spans for retrieval eval.
          </p>
          {retrievalQueries.map((q) => (
            <div key={q.id} className="query-card">
              <div className="query-id">{q.id}</div>
              <div className="query-question">{q.question}</div>
              <details>
                <summary className="subtitle">Expected spans ({q.expected_spans.length})</summary>
                {q.expected_spans.map((span, i) => (
                  <div key={i} style={{ marginTop: 4, fontSize: "0.82rem" }}>
                    <strong className="subtitle">{span.section}:</strong>{" "}
                    <span>{span.quote}</span>
                  </div>
                ))}
                <div style={{ marginTop: 6, fontSize: "0.82rem", color: "var(--muted)" }}>
                  <strong>Rationale:</strong> {q.rationale}
                </div>
              </details>
              {retrieval && (
                <div style={{ marginTop: 8 }}>
                  <div className="strategy-results">
                    {retrieval.strategies.map((s) => {
                      const qr = s.per_query.find((pq) => pq.query_id === q.id);
                      if (!qr) return null;
                      return (
                        <span key={s.name} className={`strategy-badge ${qr.hit ? "badge-hit" : "badge-miss"}`}>
                          {s.name}: {qr.hit ? "HIT" : "miss"} MRR={fmtScore(qr.mrr)}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ))}
        </>
      )}

      {sourceMd && (
        <>
          <h2>Source Protocol</h2>
          <div className="source-md">{sourceMd}</div>
        </>
      )}
    </div>
  );
}

function ModelFieldsView({
  modelKey,
  modelResult,
  schema,
}: {
  modelKey: string;
  modelResult: import("@/lib/data").ModelResult;
  schema: Schema | null;
}) {
  return (
    <div className="container">
      <div className="back-link"><Link href="/">&larr; Back to scorecard</Link></div>

      <h1>Field Results: {modelKey}</h1>
      <p className="subtitle">
        Overall: P={fmtScore(modelResult.aggregates.precision)} |{" "}
        R={fmtScore(modelResult.aggregates.recall)} |{" "}
        F1={fmtScore(modelResult.aggregates.f1)} |{" "}
        {modelResult.aggregates.n_documents} docs
      </p>

      {Object.entries(modelResult.per_document).map(([docId, doc]) => {
        const docSchema = schema?.documents.find((d) => d.doc_id === docId);
        return (
          <div key={docId}>
            <h2>
              <Link href={`/documents/${docId}`}>{docId}</Link>
              {" - "}
              <span className="subtitle" style={{ fontSize: "0.95rem" }}>
                P={fmtScore(doc.aggregates.precision)} R={fmtScore(doc.aggregates.recall)} F1={fmtScore(doc.aggregates.f1)}
              </span>
            </h2>
            <div className="field-grid">
              {Object.entries(doc.per_field).map(([fieldName, fr]) => (
                <div key={fieldName} className="field-row">
                  <div className="field-name">
                    {fieldName} ({fr.category}){" "}
                    <span className={`tag ${fr.correct ? "tag-pass" : "tag-fail"}`}>
                      {fr.correct ? "correct" : "wrong"}
                    </span>
                  </div>
                  <div className="field-values">
                    <div className="field-value field-gold">
                      <strong>gold:</strong> {fmtVal(fr.gold)}
                    </div>
                    <div className={`field-value ${fr.correct ? "field-pred" : "field-miss"}`}>
                      <strong>pred:</strong> {fmtVal(fr.predicted)}
                    </div>
                    {fr.reason && (
                      <div className="subtitle" style={{ fontSize: "0.78rem" }}>{fr.reason}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}