# Hindsight Rust Client

Auto-generated Rust client library for the Hindsight semantic memory system API.

## Features

- 🦀 **Fully typed** - Complete type safety with Rust's type system
- 🔄 **Auto-generated** - Stays in sync with the OpenAPI spec automatically
- ⚡ **Async/await** - Built on tokio and reqwest for modern async Rust
- 📦 **Standalone** - Can be published to crates.io independently

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hindsight-client = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```rust
use hindsight_client::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client
    let client = Client::new("http://localhost:8888");

    // List all agents
    let agents = client.list_agents().await?;
    for agent in agents {
        println!("Agent: {} - {}", agent.agent_id, agent.name);
    }

    // Get agent profile
    let profile = client.get_agent_profile("my-agent").await?;
    println!("Background: {}", profile.background);

    // Search memories
    let search_request = hindsight_client::types::SearchRequest {
        query: "What did I learn today?".to_string(),
        fact_type: None,
        thinking_budget: Some(100),
        max_tokens: Some(4096),
        trace: Some(false),
    };
    let results = client.search_memories("my-agent", &search_request).await?;
    for result in results.results {
        println!("- {}", result.text);
    }

    // Store a memory
    let memory_request = hindsight_client::types::BatchMemoryRequest {
        items: vec![
            hindsight_client::types::MemoryItem {
                content: "I learned about Rust today".to_string(),
                context: Some("Daily learning".to_string()),
            }
        ],
        document_id: Some("my-doc".to_string()),
    };
    client.batch_put_memories("my-agent", &memory_request).await?;

    Ok(())
}
```

## How It Works

This library uses [progenitor](https://github.com/oxidecomputer/progenitor) to generate the client code from the OpenAPI specification at **build time**.

The generation happens automatically when you run `cargo build`, so the client always stays in sync with the API schema.

### Build Process

1. `build.rs` reads the OpenAPI spec from `../../openapi.json`
2. Converts OpenAPI 3.1 → 3.0 (for progenitor compatibility)
3. Generates Rust client code using progenitor
4. Code is included in the library via `include!()` macro

## API Methods

All API endpoints are available as async methods on the `Client` struct:

### Agent Management
- `list_agents()` - List all agents
- `create_or_update_agent()` - Create or update an agent
- `get_agent_profile()` - Get agent profile with personality
- `update_agent_personality()` - Update agent personality traits
- `add_agent_background()` - Add/merge agent background
- `get_agent_stats()` - Get memory statistics

### Memory Operations
- `search_memories()` - Semantic search across memories
- `think()` - Generate contextual answers using agent identity
- `batch_put_memories()` - Store multiple memories
- `batch_put_async()` - Queue memories for background processing
- `list_memories()` - List memory units with pagination
- `delete_memory_unit()` - Delete a specific memory
- `clear_agent_memories()` - Clear all or filtered memories

### Document Management
- `list_documents()` - List documents with optional search
- `get_document()` - Get document details and content
- `delete_document()` - Delete document and its memories

### Operations (Async Tasks)
- `list_operations()` - List async operations
- `cancel_operation()` - Cancel a pending operation

### Visualization
- `get_graph()` - Get memory graph data for visualization

## Error Handling

The client uses `progenitor_client::Error` for all errors:

```rust
match client.get_agent_profile("my-agent").await {
    Ok(profile) => println!("Got profile: {}", profile.name),
    Err(progenitor_client::Error::ErrorResponse(resp)) => {
        println!("API error: {} - {}", resp.status, resp.body);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## Development

### Building

```bash
cargo build
```

The OpenAPI spec is automatically converted and the client is generated during build.

### Testing

```bash
cargo test
```

### Releasing

This client can be published to crates.io independently of the CLI:

```bash
cargo publish
```

## Architecture

```
hindsight-clients/rust/
├── Cargo.toml          # Package definition
├── build.rs            # Build script (generates client)
├── src/
│   └── lib.rs          # Library entry point
└── target/
    └── debug/build/
        └── hindsight-client-.../out/
            └── hindsight_client_generated.rs  # Generated code
```

## License

MIT
