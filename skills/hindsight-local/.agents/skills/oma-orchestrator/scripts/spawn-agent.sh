#!/bin/bash
# spawn-agent.sh - Wrapper for oma agent:spawn
# Usage: ./spawn-agent.sh <agent-id> <prompt> <session-id> [-w workspace] [-m model]

exec oma agent:spawn "$@"
