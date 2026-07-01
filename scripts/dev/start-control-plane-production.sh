#!/bin/bash
# Start Control Plane for PRODUCTION (public domain)

set -e

cd /Users/oliververmeulen/hindsight/hindsight-control-plane

# Load PRODUCTION environment
export HINDSIGHT_CP_DATAPLANE_API_URL=https://hindsight-api.collabmind.dev
export HINDSIGHT_CP_ACCESS_KEY=your-collabminds-access-key
export PORT=9999
export NODE_ENV=production

echo "🚀 Starting Control Plane (PRODUCTION MODE)"
echo "   API: https://hindsight-api.collabmind.dev"
echo "   UI:  http://localhost:9998"
echo "   Public: https://neuron-ai-controller.collabmind.dev"

# Use standalone production build
exec node standalone/server.js
