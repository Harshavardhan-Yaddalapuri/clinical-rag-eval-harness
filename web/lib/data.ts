import fs from "fs";
import path from "path";

/**
 * Resolve the repo root. In dev (running from web/), the eval/data/schema
 * artifacts live two levels up at the repo root. In production, a prebuild
 * step copies evals/, data/, shared/ INTO web/, so they live at process.cwd().
 * We prefer the in-cwd copy when present, and fall back to the parent dir.
 */
export function getRepoRoot(): string {
  const cwd = process.cwd();
  // If the copied artifacts are present in cwd (production), use cwd.
  if (fs.existsSync(path.join(cwd, "evals")) || fs.existsSync(path.join(cwd, "shared"))) {
    return cwd;
  }
  // Otherwise (dev), walk up to the repo root.
  if (path.basename(cwd) === "web") {
    return path.resolve(cwd, "..");
  }
  return cwd;
}

export function readJson<T>(relativePath: string): T | null {
  const root = getRepoRoot();
  const full = path.join(root, relativePath);
  try {
    const raw = fs.readFileSync(full, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function readText(relativePath: string): string | null {
  const root = getRepoRoot();
  const full = path.join(root, relativePath);
  try {
    return fs.readFileSync(full, "utf-8");
  } catch {
    return null;
  }
}

export function listDir(relativePath: string): string[] {
  const root = getRepoRoot();
  const full = path.join(root, relativePath);
  try {
    return fs.readdirSync(full).sort();
  } catch {
    return [];
  }
}

// --- Types ---

export interface SummaryEntry {
  precision: number;
  recall: number;
  f1: number;
  n_documents: number;
}

export type Summary = Record<string, SummaryEntry>;

export interface FieldResult {
  correct: boolean;
  precision: number;
  recall: number;
  f1: number;
  reason: string;
  field: string;
  category: string;
  gold: unknown;
  predicted: unknown;
}

export interface DocAggregates {
  precision: number;
  recall: number;
  f1: number;
  n_fields: number;
}

export interface DocResult {
  doc_id: string;
  per_field: Record<string, FieldResult>;
  aggregates: DocAggregates;
}

export interface ModelAggregates {
  precision: number;
  recall: number;
  f1: number;
  n_documents: number;
}

export interface ModelResult {
  model: string;
  per_document: Record<string, DocResult>;
  aggregates: ModelAggregates;
}

export type Results = Record<string, ModelResult>;

export interface RetrievalConfig {
  k: number;
  docs: string[];
}

export interface PerQueryResult {
  hit: boolean;
  recall: number;
  mrr: number;
  matched_ranks: number[];
  n_expected: number;
  n_retrieved: number;
  query_id: string;
  question: string;
  strategy: string;
}

export interface StrategyMetrics {
  hit_at_k: number;
  recall_at_k: number;
  mrr: number;
  n_queries: number;
}

export interface StrategyResult {
  name: string;
  metrics: StrategyMetrics;
  per_doc: Record<string, StrategyMetrics>;
  per_query: PerQueryResult[];
}

export interface RetrievalEval {
  strategies: StrategyResult[];
  config: RetrievalConfig;
}

export interface SchemaDoc {
  doc_id: string;
  nct_id: string;
  title: string;
  disease_area: string;
  source_url: string;
  fields: Record<string, {
    type: string;
    description: string;
    valid_values?: string[];
    tolerance?: { abs: number; pct: number };
    item_type?: string;
    fields?: Record<string, string>;
  }>;
}

export interface Schema {
  version: string;
  description: string;
  documents: SchemaDoc[];
}

export interface RetrievalQuery {
  id: string;
  question: string;
  expected_spans: { section: string; quote: string }[];
  rationale: string;
}

export function scoreColor(score: number): string {
  if (score >= 0.9) return "score-bar-high";
  if (score >= 0.7) return "score-bar-mid";
  return "score-bar-low";
}

export function scoreTag(score: number): string {
  if (score >= 0.9) return "tag-pass";
  if (score >= 0.7) return "tag-mid";
  return "tag-fail";
}

export function fmtScore(n: number): string {
  return n.toFixed(4);
}

export function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "(null)";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return v.length === 0 ? "[]" : JSON.stringify(v);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}