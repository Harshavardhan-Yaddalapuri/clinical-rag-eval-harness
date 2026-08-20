import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { getRepoRoot, readJson, type Schema, type FieldResult } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface RunRequest {
  model?: string;
  doc_id?: string;
}

interface RunResponse {
  model: string;
  doc_id: string;
  fields: Record<string, unknown>;
  gold: Record<string, unknown> | null;
  comparison: Record<string, { predicted: unknown; gold: unknown; match: boolean }> | null;
  error?: string;
}

export async function POST(request: NextRequest) {
  let body: RunRequest;
  try {
    body = await request.json();
  } catch {
    // Try form-encoded
    const formData = await request.formData();
    body = {
      model: String(formData.get("model") || ""),
      doc_id: String(formData.get("doc_id") || ""),
    };
  }

  const model = body.model || "glm-5.2";
  const docId = body.doc_id || "actt1";

  if (!model || !docId) {
    return NextResponse.json(
      { error: "Both model and doc_id are required" },
      { status: 400 }
    );
  }

  const apiKey = process.env.OLLAMA_API_KEY;
  const baseUrl = process.env.OLLAMA_BASE_URL || "https://ollama.com/v1";

  if (!apiKey) {
    return NextResponse.json(
      { error: "OLLAMA_API_KEY not configured on the server. Live extraction is unavailable." },
      { status: 503 }
    );
  }

  const root = getRepoRoot();

  // Read source text
  const sourcePath = path.join(root, "data", "golden", docId, "source.md");
  if (!fs.existsSync(sourcePath)) {
    return NextResponse.json(
      { error: `Document ${docId} not found` },
      { status: 404 }
    );
  }
  const sourceText = fs.readFileSync(sourcePath, "utf-8");

  // Read schema and build field spec
  const schema = readJson<Schema>("shared/schema.json");
  if (!schema) {
    return NextResponse.json(
      { error: "Schema not found" },
      { status: 500 }
    );
  }
  const docSchema = schema.documents.find((d) => d.doc_id === docId);
  if (!docSchema) {
    return NextResponse.json(
      { error: `Schema for ${docId} not found` },
      { status: 404 }
    );
  }

  // Read extraction prompt
  const promptPath = path.join(root, "shared", "prompts", "extract.txt");
  const systemPrompt = fs.existsSync(promptPath)
    ? fs.readFileSync(promptPath, "utf-8")
    : "You are a clinical trial protocol extraction engine. Extract the requested fields into strict JSON. Return ONLY valid JSON.";

  // Build field spec
  const fieldLines: string[] = [];
  for (const [key, cfg] of Object.entries(docSchema.fields)) {
    let spec = `- ${key} (${cfg.type}): ${cfg.description}`;
    if (cfg.type === "categorical" && cfg.valid_values) {
      spec += ` [one of: ${cfg.valid_values.join(", ")}]`;
    }
    if (cfg.type === "list") {
      spec += ` [list of ${cfg.item_type || "string"}]`;
    }
    fieldLines.push(spec);
  }

  const userPrompt = `Extract the following fields from the protocol text below.\n\nFIELDS:\n${fieldLines.join("\n")}\n\nPROTOCOL TEXT:\n${sourceText}\n\nReturn a single JSON object with exactly these keys. Use null for any value not stated in the text.`;

  // Call the LLM
  try {
    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        temperature: 0,
      }),
      signal: AbortSignal.timeout(120000),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      return NextResponse.json(
        { error: `LLM API error: ${resp.status} ${errText.slice(0, 200)}` },
        { status: 502 }
      );
    }

    const data = await resp.json();
    const content = data?.choices?.[0]?.message?.content || "";

    // Parse JSON from content (tolerate code fences)
    let parsed: Record<string, unknown>;
    let text = content.trim();
    if (text.startsWith("```")) {
      text = text.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "").trim();
    }
    try {
      parsed = JSON.parse(text);
    } catch {
      return NextResponse.json(
        { error: "LLM returned invalid JSON", raw: content.slice(0, 500) },
        { status: 502 }
      );
    }

    // Normalize: ensure all schema fields exist
    const fields: Record<string, unknown> = {};
    for (const key of Object.keys(docSchema.fields)) {
      fields[key] = parsed[key] ?? null;
    }

    // Load gold for comparison
    const goldPath = path.join(root, "data", "golden", docId, "golden.json");
    let gold: Record<string, unknown> | null = null;
    let comparison: Record<string, { predicted: unknown; gold: unknown; match: boolean }> | null = null;
    if (fs.existsSync(goldPath)) {
      gold = JSON.parse(fs.readFileSync(goldPath, "utf-8"));
      comparison = {};
      for (const key of Object.keys(docSchema.fields)) {
        const goldVal = gold?.[key];
        const predVal = fields[key];
        const match = JSON.stringify(goldVal) === JSON.stringify(predVal);
        comparison[key] = { predicted: predVal, gold: goldVal, match };
      }
    }

    const result: RunResponse = {
      model,
      doc_id: docId,
      fields,
      gold,
      comparison,
    };

    return NextResponse.json(result);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: `Extraction failed: ${msg}` },
      { status: 500 }
    );
  }
}