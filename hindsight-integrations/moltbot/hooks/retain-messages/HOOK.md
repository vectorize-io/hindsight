---
name: hindsight-retain-messages
description: Automatically retains messages to Hindsight long-term memory
events:
  - agent_end
metadata:
  moltbot:
    emoji: 🧠
---

# Hindsight Message Retention

This hook automatically retains conversation messages to Hindsight's long-term memory.

## When It Runs

- On `agent_end`: After each agent turn completes

## What It Does

1. Captures the current session messages
2. Formats them into a conversation transcript
3. Calls Hindsight's retain API with the session_id as document_id
4. Queues for background processing (async)
5. Extracts facts, entities, and relationships from the conversation
