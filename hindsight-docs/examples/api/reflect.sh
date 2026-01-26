#!/bin/bash
# Reflect API examples for Hindsight CLI
# Run: bash examples/api/reflect.sh

set -e

HINDSIGHT_URL="${HINDSIGHT_API_URL:-http://localhost:8888}"

# =============================================================================
# Setup (not shown in docs)
# =============================================================================
hindsight memory retain my-bank "Alice works at Google as a software engineer"
hindsight memory retain my-bank "Alice has been working there for 5 years"

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:reflect-basic]
hindsight memory reflect my-bank "What do you know about Alice?"
# [/docs:reflect-basic]


# [docs:reflect-with-context]
hindsight memory reflect my-bank "Should I learn Python?" --context "career advice"
# [/docs:reflect-with-context]


# [docs:reflect-high-budget]
hindsight memory reflect my-bank "Summarize my week" --budget high
# [/docs:reflect-high-budget]


# [docs:reflect-structured-output]
# First, create a JSON schema file schema.json:
cat > schema.json << 'EOF'
{
  "type": "object",
  "properties": {
    "recommendation": {"type": "string"},
    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    "key_factors": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["recommendation", "confidence", "key_factors"]
}
EOF

# Then use the --schema flag:
hindsight memory reflect hiring-team \
  "Should we hire Alice for the ML team lead position?" \
  --schema schema.json

# Cleanup the temporary schema file
rm -f schema.json
# [/docs:reflect-structured-output]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
curl -s -X DELETE "${HINDSIGHT_URL}/v1/default/banks/my-bank" > /dev/null

echo "reflect.sh: All examples passed"
