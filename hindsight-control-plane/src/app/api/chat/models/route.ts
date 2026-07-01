/**
 * Chat Models API Route
 *
 * Proxies model listing to the CollabMind AI proxy.
 * GET /api/chat/models → GET {PROXY_URL}/models
 */

import { NextResponse } from "next/server";

const PROXY_URL = process.env.COLLABMIND_AI_PROXY_URL || "http://0.0.0.0:3001/v1";

export async function GET() {
  try {
    const apiKey = process.env.COLLABMIND_AI_PROXY_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "AI proxy key not configured. Set COLLABMIND_AI_PROXY_KEY in .env.local" },
        { status: 500 }
      );
    }

    const res = await fetch(`${PROXY_URL}/models`, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      cache: "no-store",
    });

    if (!res.ok) {
      const errorText = await res.text();
      return NextResponse.json(
        { error: `AI proxy error (${res.status}): ${errorText}` },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error("Chat models error:", error);
    return NextResponse.json({ error: error?.message || "Internal server error" }, { status: 500 });
  }
}
