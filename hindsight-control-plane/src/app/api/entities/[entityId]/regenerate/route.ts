import { NextRequest, NextResponse } from 'next/server';

const DATAPLANE_URL = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || 'http://localhost:8888';

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ entityId: string }> }
) {
  try {
    const { entityId } = await params;
    const searchParams = request.nextUrl.searchParams;
    const agentId = searchParams.get('agent_id');

    if (!agentId) {
      return NextResponse.json(
        { error: 'agent_id is required' },
        { status: 400 }
      );
    }

    const decodedEntityId = decodeURIComponent(entityId);
    const url = `${DATAPLANE_URL}/api/v1/agents/${agentId}/entities/${decodedEntityId}/regenerate`;
    const response = await fetch(url, { method: 'POST' });
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('Error regenerating entity observations:', error);
    return NextResponse.json(
      { error: 'Failed to regenerate entity observations' },
      { status: 500 }
    );
  }
}
