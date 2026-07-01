/**
 * Chat Proxy API Route
 *
 * Proxies chat completion requests to the CollabMind AI proxy.
 * Uses environment variables for configuration:
 *   - COLLABMIND_AI_PROXY_URL  (default: http://0.0.0.0:3001/v1)
 *   - COLLABMIND_AI_PROXY_KEY  (required)
 */

import { NextRequest, NextResponse } from "next/server";

const PROXY_URL = process.env.COLLABMIND_AI_PROXY_URL || "http://0.0.0.0:3001/v1";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { messages, model = "auto", stream = false, ...rest } = body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return NextResponse.json({ error: "Messages array is required" }, { status: 400 });
    }

    const apiKey = process.env.COLLABMIND_AI_PROXY_KEY;
    if (!apiKey) {
      return NextResponse.json(
        {
          error: "AI proxy key not configured. Set COLLABMIND_AI_PROXY_KEY in .env.local",
        },
        { status: 500 }
      );
    }

    const payload: Record<string, unknown> = {
      model,
      messages,
      stream,
      ...rest,
    };

    // Non-streaming response
    if (!stream) {
      const res = await fetch(`${PROXY_URL}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(payload),
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
    }

    // Streaming response
    const aiRes = await fetch(`${PROXY_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(payload),
    });

    if (!aiRes.ok) {
      const errorText = await aiRes.text();
      return NextResponse.json(
        { error: `AI proxy error (${aiRes.status}): ${errorText}` },
        { status: aiRes.status }
      );
    }

    // Return the stream directly
    return new Response(aiRes.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error: any) {
    console.error("Chat proxy error:", error);
    return NextResponse.json({ error: error?.message || "Internal server error" }, { status: 500 });
  }
}
