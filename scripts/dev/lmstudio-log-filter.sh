#!/bin/bash
# Filter LM Studio logs to show only important info

tail -f "$1" 2>/dev/null | grep -v "Unexpected endpoint" | grep -E "INFO|Accumulated|progress|ERROR.*Client disconnected"
