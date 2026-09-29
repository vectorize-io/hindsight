/**
 * GET  /api/projects         — list all projects for the user
 * DELETE /api/projects?id=X  — delete a project
 */

import { NextRequest, NextResponse } from "next/server";
import { getAllProjects, deleteProject } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const userId = req.headers.get("x-user-id") ?? process.env.DEFAULT_USER_ID ?? "reelcraft-user-default";
  try {
    const projects = getAllProjects(userId);
    return NextResponse.json({ success: true, data: { projects } });
  } catch (error: unknown) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Failed to fetch projects" },
      { status: 500 }
    );
  }
}

export async function DELETE(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const id = searchParams.get("id");

  if (!id) {
    return NextResponse.json({ success: false, error: "Project ID required" }, { status: 400 });
  }

  try {
    const deleted = deleteProject(id);
    if (!deleted) {
      return NextResponse.json({ success: false, error: "Project not found" }, { status: 404 });
    }
    return NextResponse.json({ success: true, message: "Project deleted" });
  } catch (error: unknown) {
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : "Failed to delete" },
      { status: 500 }
    );
  }
}
