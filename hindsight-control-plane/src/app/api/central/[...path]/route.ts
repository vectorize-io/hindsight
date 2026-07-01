import { NextRequest, NextResponse } from "next/server";

const CENTRAL_API_BASE = process.env.CENTRAL_API_BASE_URL || "http://localhost:8000";

export async function GET(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(request, params.path);
}

export async function POST(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(request, params.path);
}

export async function PUT(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(request, params.path);
}

export async function PATCH(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(request, params.path);
}

export async function DELETE(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxyRequest(request, params.path);
}

async function proxyRequest(request: NextRequest, pathSegments: string[]) {
  const targetPath = pathSegments.join("/");
  const searchParams = request.nextUrl.searchParams.toString();
  const queryString = searchParams ? `?${searchParams}` : "";

  const targetUrl = `${CENTRAL_API_BASE}/api/${targetPath}${queryString}`;

  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    // Forward auth header if present
    const authHeader = request.headers.get("authorization");
    if (authHeader) {
      headers["Authorization"] = authHeader;
    }

    const fetchOptions: RequestInit = {
      method: request.method,
      headers,
      signal: AbortSignal.timeout(10000),
    };

    // Forward body for mutating methods
    if (["POST", "PUT", "PATCH"].includes(request.method)) {
      try {
        const body = await request.json();
        fetchOptions.body = JSON.stringify(body);
      } catch {
        // No body or non-JSON body — proceed without
      }
    }

    const response = await fetch(targetUrl, fetchOptions);

    const responseHeaders = new Headers();
    responseHeaders.set("Content-Type", response.headers.get("Content-Type") || "application/json");

    const responseBody = await response.text();

    return new NextResponse(responseBody, {
      status: response.status,
      headers: responseHeaders,
    });
  } catch (error: any) {
    console.error(`[CENTRAL_API_PROXY] Failed to reach ${targetUrl}:`, error.message);
    return NextResponse.json(
      {
        error: "Central API unreachable",
        detail: error.message,
        target: targetUrl,
      },
      { status: 502 }
    );
  }
}
