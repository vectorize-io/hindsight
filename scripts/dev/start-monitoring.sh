#!/bin/bash
# Convenience wrapper to start the monitoring stack
exec "$(dirname "${BASH_SOURCE[0]}")/monitoring/start.sh" "$@"
