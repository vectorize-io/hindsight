#!/bin/bash
# Start Control Plane for LOCAL development

set -e

cd /Users/oliververmeulen/hindsight/hindsight-control-plane

# Load LOCAL environment
export HINDSIGHT_CP_DATAPLANE_API_URL=http://localhost:8888
export HINDSIGHT_CP_ACCESS_KEY=your-collabminds-access-key
export PORT=9999

echo "🚀 Starting Control Plane (LOCAL MODE)"
echo "   API: http://localhost:8888"
echo "   UI:  http://localhost:9998"

exec npm run dev -- --turbopack -p 9999
