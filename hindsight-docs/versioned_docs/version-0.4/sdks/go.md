---
sidebar_position: 3
---

# Go Client

Official Go client for the Hindsight API, built on [ogen](https://github.com/ogen-go/ogen) for strongly-typed code generation from the OpenAPI 3.1 spec.

## Installation

```bash
go get github.com/vectorize-io/hindsight/hindsight-clients/go
```

Requires Go 1.25+.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    hindsight "github.com/vectorize-io/hindsight/hindsight-clients/go"
)

func main() {
    client, err := hindsight.New("http://localhost:8888")
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Retain a memory
    client.Retain(ctx, "my-bank", "Alice works at Google")

    // Recall memories
    resp, _ := client.Recall(ctx, "my-bank", "What does Alice do?")
    for _, r := range resp.Results {
        fmt.Println(r.Text)
    }

    // Reflect - generate response with disposition
    answer, _ := client.Reflect(ctx, "my-bank", "Tell me about Alice")
    fmt.Println(answer.Text)
}
```

## Client Initialization

```go
import hindsight "github.com/vectorize-io/hindsight/hindsight-clients/go"

// Default client
client, err := hindsight.New("http://localhost:8888")

// With API key authentication
client, err := hindsight.New("http://localhost:8888",
    hindsight.WithAPIKey("your-api-key"),
)

// With custom HTTP client
client, err := hindsight.New("http://localhost:8888",
    hindsight.WithHTTPClient(&http.Client{Timeout: 30 * time.Second}),
)
```

## Core Operations

### Retain (Store Memory)

```go
// Simple
_, err := client.Retain(ctx, "my-bank", "Alice works at Google as a software engineer")

// With options
_, err := client.Retain(ctx, "my-bank", "Alice got promoted",
    hindsight.WithContext("career update"),
    hindsight.WithTimestamp(time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC)),
    hindsight.WithDocumentID("conversation_001"),
    hindsight.WithMetadata(map[string]string{"source": "slack"}),
    hindsight.WithTags([]string{"career", "updates"}),
)
```

### Retain Batch

```go
items := []hindsight.MemoryItem{
    {Content: "Alice works at Google"},
    {Content: "Bob is a data scientist"},
}

_, err := client.RetainBatch(ctx, "my-bank", items,
    hindsight.WithDocumentTags([]string{"team-info"}),
    hindsight.WithAsync(false), // Set true for background processing
)
```

### Recall (Search)

```go
// Simple
resp, err := client.Recall(ctx, "my-bank", "What does Alice do?")

for _, r := range resp.Results {
    fmt.Printf("  %s (type: %s)\n", r.Text, r.Type.Or("unknown"))
}

// With options
resp, err := client.Recall(ctx, "my-bank", "What does Alice do?",
    hindsight.WithTypes([]string{"world", "experience"}), // Filter by fact type
    hindsight.WithMaxTokens(4096),
    hindsight.WithBudget(hindsight.BudgetHigh), // BudgetLow, BudgetMid, BudgetHigh
    hindsight.WithTrace(true),                   // Include execution trace
    hindsight.WithRecallTags([]string{"career"}),
    hindsight.WithRecallTagsMatch(hindsight.TagsMatchAnyStrict),
)
```

### Reflect (Generate Response)

```go
resp, err := client.Reflect(ctx, "my-bank", "What should I know about Alice?",
    hindsight.WithReflectBudget(hindsight.BudgetMid),
    hindsight.WithReflectMaxTokens(2048),
)

fmt.Println(resp.Text) // Generated markdown response
```

### Reflect with Structured Output

```go
resp, err := client.Reflect(ctx, "my-bank",
    "What programming language should I learn for data science?",
    hindsight.WithResponseSchema(map[string]any{
        "type": "object",
        "properties": map[string]any{
            "recommendation": map[string]any{"type": "string"},
            "reasons":        map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
            "confidence":     map[string]any{"type": "string"},
        },
        "required": []any{"recommendation", "reasons"},
    }),
    hindsight.WithReflectMaxTokens(4096),
)

// resp.StructuredOutput contains the parsed JSON schema result
```

## Bank Management

### Create Bank

```go
_, err := client.CreateBank(ctx, "my-bank",
    hindsight.WithBankName("Assistant"),
    hindsight.WithMission("Helpful AI assistant tracking user preferences."),
    hindsight.WithDisposition(hindsight.DispositionTraits{
        Skepticism: 3, // 1-5: trusting to skeptical
        Literalism: 3, // 1-5: flexible to literal
        Empathy:    3, // 1-5: detached to empathetic
    }),
)
```

### Other Bank Operations

```go
// Get bank profile
profile, err := client.GetBankProfile(ctx, "my-bank")
fmt.Println(profile.Mission)

// List all banks
banks, err := client.ListBanks(ctx)
for _, b := range banks.Banks {
    fmt.Println(b.BankID)
}

// Update mission
_, err = client.SetMission(ctx, "my-bank", "New mission statement")

// Update disposition
_, err = client.UpdateDisposition(ctx, "my-bank", hindsight.DispositionTraits{
    Skepticism: 4,
    Literalism: 2,
    Empathy:    5,
})

// Delete bank (destructive, cannot be undone)
err = client.DeleteBank(ctx, "my-bank")
```

## Tag Filtering

Tags provide visibility scoping for memories. Use them to partition memories within a bank.

```go
// Store tagged memories
client.Retain(ctx, "my-bank", "Project X meeting notes",
    hindsight.WithTags([]string{"project_x", "meetings"}),
)

// Recall only project_x memories (strict - excludes untagged)
resp, _ := client.Recall(ctx, "my-bank", "What happened in meetings?",
    hindsight.WithRecallTags([]string{"project_x"}),
    hindsight.WithRecallTagsMatch(hindsight.TagsMatchAnyStrict),
)

// Reflect scoped to tags
resp, _ := client.Reflect(ctx, "my-bank", "Summarize project X",
    hindsight.WithReflectTags([]string{"project_x"}),
    hindsight.WithReflectTagsMatch(hindsight.TagsMatchAnyStrict),
)
```

| Match Mode | Behavior |
|-----------|----------|
| `TagsMatchAny` | OR matching, includes untagged memories |
| `TagsMatchAll` | AND matching, includes untagged memories |
| `TagsMatchAnyStrict` | OR matching, excludes untagged memories |
| `TagsMatchAllStrict` | AND matching, excludes untagged memories |

## Advanced Usage (ogen Client)

For operations not covered by the high-level wrapper (documents, entities, mental models, directives, operations), access the generated ogen client directly:

```go
import "github.com/vectorize-io/hindsight/hindsight-clients/go/internal/ogenapi"

ogen := client.OgenClient()

// List entities
entities, err := ogen.ListEntities(ctx, ogenapi.ListEntitiesParams{
    BankID: "my-bank",
})

// Create a mental model
model, err := ogen.CreateMentalModel(ctx,
    &ogenapi.CreateMentalModelRequest{
        Name:        ogenapi.NewOptString("User Preferences"),
        SourceQuery: "What are this user's preferences and habits?",
    },
    ogenapi.CreateMentalModelParams{BankID: "my-bank"},
)

// Create a directive
directive, err := ogen.CreateDirective(ctx,
    &ogenapi.CreateDirectiveRequest{
        Name:    "Response Style",
        Content: "Always respond in a friendly, concise manner",
    },
    ogenapi.CreateDirectiveParams{BankID: "my-bank"},
)

// List operations (async tasks)
ops, err := ogen.ListOperations(ctx, ogenapi.ListOperationsParams{
    BankID: "my-bank",
})

// Get bank stats
stats, err := ogen.GetAgentStats(ctx, ogenapi.GetAgentStatsParams{
    BankID: "my-bank",
})
```

## Error Handling

The client returns standard Go errors. HTTP errors from the API are returned as ogen error types:

```go
resp, err := client.Recall(ctx, "nonexistent-bank", "query")
if err != nil {
    // Handle error - could be network, HTTP 4xx/5xx, etc.
    log.Printf("recall failed: %v", err)
}
```

## Code Generation

The Go client is built on [ogen](https://github.com/ogen-go/ogen). The generated code lives in `internal/ogenapi/` and provides full type safety with no `interface{}` or reflection.

To regenerate after API changes:

```bash
cd hindsight-clients/go
go generate ./...
go build ./...
```
