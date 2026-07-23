import { runExtractPreview } from "@vectorize-io/hindsight-mission-sandbox/core";

import { projectDir } from "@/app/lib/project-context";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Dry-run extraction preview: what does this mission extract from the given text? (no ingest) */
export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as {
    project?: string;
    content?: string;
    retainMission?: string | null;
  };
  if (!body.project || !body.content) {
    return Response.json({ error: "project and content are required" }, { status: 400 });
  }
  try {
    const facts = await runExtractPreview({
      projectDir: projectDir(body.project),
      content: body.content,
      retainMission: body.retainMission,
    });
    return Response.json({ facts });
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
