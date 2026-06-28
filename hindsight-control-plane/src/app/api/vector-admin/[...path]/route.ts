import { NextRequest, NextResponse } from "next/server";

const VECTOR_ADMIN_BACKEND =
  process.env.VECTOR_ADMIN_BACKEND_URL || "http://localhost:3001";

export const dynamic = "force-dynamic";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const pathStr = path.join("/");
  const searchParams = _request.nextUrl.searchParams.toString();
  const url = `${VECTOR_ADMIN_BACKEND}/api/${pathStr}${searchParams ? `?${searchParams}` : ""}`;

  try {
    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...(process.env.VECTOR_ADMIN_API_KEY
          ? { "X-API-Key": process.env.VECTOR_ADMIN_API_KEY }
          : {}),
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `VectorAdmin backend error: ${response.statusText}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown error";
    const statusCode = 502;
    return NextResponse.json(
      { error: `Failed to connect to VectorAdmin backend: ${message}`, status: "disconnected" },
      { status: statusCode }
    );
  }
}

export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const pathStr = path.join("/");
  const body = await _request.json().catch(() => undefined);
  const url = `${VECTOR_ADMIN_BACKEND}/api/${pathStr}`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.VECTOR_ADMIN_API_KEY
          ? { "X-API-Key": process.env.VECTOR_ADMIN_API_KEY }
          : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `VectorAdmin backend error: ${response.statusText}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { error: `Failed to connect to VectorAdmin backend: ${message}` },
      { status: 502 }
    );
  }
}
