#!/bin/bash
# parallel-run.sh - Wrapper for oma agent:parallel
# Usage: ./parallel-run.sh <tasks-file.yaml> [-m <model>]
#        ./parallel-run.sh --inline "backend:task1" "frontend:task2" ...

exec oma agent:parallel "$@"
