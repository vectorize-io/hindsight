import { NextRequest, NextResponse } from "next/server";

import { dataplaneBankUrl, getDataplaneHeaders } from "@/lib/hindsight-client";

/**
 * Proxy for the dataplane dry-run extraction endpoint: extract facts from text with a candidate
 * retain mission, WITHOUT persisting (no resolution/links/embeddings). Preview-only.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const bankId = body.bank_id || "default";
    const {
      content,
      retain_mission,
      retain_extraction_mode,
      retain_custom_instructions,
      retain_chunk_size,
      entity_labels,
      entities_allow_free_form,
      llm_output_language,
      agent_name,
    } = body;

    const res = await fetch(dataplaneBankUrl(bankId, "/memories/dry-run-extract"), {
      method: "POST",
      headers: getDataplaneHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        content,
        retain_mission,
        retain_extraction_mode,
        retain_custom_instructions,
        retain_chunk_size,
        entity_labels,
        entities_allow_free_form,
        llm_output_language,
        agent_name,
      }),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.ok ? 200 : res.status });
  } catch (e) {
    return NextResponse.json(
      { detail: e instanceof Error ? e.message : String(e) },
      { status: 500 }
    );
  }
}
