---
sidebar_position: 3
---

# Go Memory-Augmented API


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/go-memory-service)
:::


A Go HTTP microservice demonstrating per-user memory isolation with Hindsight. Remembers each user's tech stack, problems solved, and preferences to provide personalized assistance.

## Features

- 🔐 **Per-User Isolation**: Each user gets their own memory bank
- 🧠 **Context-Aware Responses**: Uses recall + reflect for personalized answers
- 🏃 **Fire-and-Forget Memory**: Background goroutines store interactions without blocking responses
- 🏷️ **Tag-Based Partitioning**: Organize memories by type (projects, debugging, preferences)

## Setup

### 1. Start Hindsight

```bash
export OPENAI_API_KEY=your-key

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=o3-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

### 2. Run the service

```bash
go run main.go
```

### 3. Try it out

```bash
# Store memories
curl -s localhost:8080/learn -d '{
  "user_id": "alice",
  "content": "I am building a Go microservice with gRPC and PostgreSQL",
  "tags": ["project"]
}'

curl -s localhost:8080/learn -d '{
  "user_id": "alice",
  "content": "I prefer structured logging with slog over zerolog",
  "tags": ["preferences"]
}'

# Ask questions (uses recall + reflect)
curl -s localhost:8080/ask -d '{
  "user_id": "alice",
  "query": "What tech stack am I using?"
}' | jq .

# Raw memory recall
curl -s "localhost:8080/recall/alice?q=database" | jq .
```

## API Endpoints

- `POST /learn` - Store new information for a user
- `POST /ask` - Ask a question using the user's memories
- `GET /recall/{userID}?q=query` - Direct memory recall
- `GET /health` - Health check

## Key Patterns

**Per-User Banks**: Each user gets an isolated memory bank (`user-alice`, `user-bob`)

**Async Memory Storage**: Interactions are stored in background goroutines:

```go
go func() {
    bgCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    retainReq := hindsight.RetainRequest{
        Items: []hindsight.MemoryItem{{
            Content: interaction,
            Context: *hindsight.NewNullableString(hindsight.PtrString("Q&A interaction")),
        }},
    }
    client.MemoryAPI.RetainMemories(bgCtx, bankID).RetainRequest(retainReq).Execute()
}()
```

**Tag-Based Filtering**: Partition memories within a bank by type for scoped retrieval

## Learn More

- [Go SDK Documentation](https://hindsight.vectorize.io/sdks/go)
- [Hindsight Documentation](https://hindsight.vectorize.io)
