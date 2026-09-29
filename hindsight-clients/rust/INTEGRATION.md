# CLI Integration Guide

This guide shows how to integrate the auto-generated Rust client into the CLI.

## Approach

The generated client is **async** (uses tokio), while the CLI currently uses **blocking** code. There are two integration strategies:

### Option 1: Minimal Wrapper (Current Approach)

Create a thin `api.rs` wrapper that uses `tokio::runtime::Runtime` to bridge sync→async:

```rust
pub struct ApiClient {
    client: hindsight_client::Client,
    runtime: tokio::runtime::Runtime,
}

impl ApiClient {
    pub fn new(base_url: String) -> Result<Self> {
        let runtime = tokio::runtime::Runtime::new()?;
        let client = hindsight_client::Client::new(&base_url);
        Ok(ApiClient { client, runtime })
    }

    pub fn list_agents(&self) -> Result<Vec<hindsight_client::types::AgentListItem>> {
        self.runtime.block_on(async {
            self.client.list_agents().await
        })
    }
    // ... other methods
}
```

### Option 2: Full Async (Recommended for New CLI)

Make the CLI fully async using `#[tokio::main]`:

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = hindsight_client::Client::new(&config.api_url);

    match cli.command {
        Commands::Agent(AgentCommands::List) => {
            let agents = client.list_agents().await?;
            for agent in agents {
                println!("  - {}", agent.agent_id);
            }
        }
        // ... other commands
    }

    Ok(())
}
```

## Type Mapping

The generated types are in `hindsight_client::types::*`. Here's the mapping:

| CLI Expected | Generated Type |
|-------------|----------------|
| `Agent` | `AgentListItem` |
| `AgentProfile` | `AgentProfileResponse` |
| `AgentStats` | From `/stats` endpoint |
| `BatchMemoryRequest` | `BatchPutRequest` |
| `BatchMemoryResponse` | `BatchPutResponse` |
| `DocumentsResponse` | `ListDocumentsResponse` |
| `Document` | Item in `ListDocumentsResponse` |
| `DocumentDetails` | `DocumentResponse` |
| `OperationsResponse` | From `/operations` endpoint |

## Example: List Agents Command

### Before (Manual API Code):
```rust
pub fn list_agents(&self, verbose: bool) -> Result<Vec<Agent>> {
    let url = format!("{}/api/v1/agents", self.base_url);
    let response = self.client.get(&url).send()?;
    // ... manual parsing
}
```

### After (Generated Client):
```rust
pub fn list_agents(&self) -> Result<Vec<AgentListItem>> {
    self.runtime.block_on(async {
        self.client.list_agents().await
    })
}
```

## Benefits

✅ **No manual maintenance** - API client updates automatically with OpenAPI spec
✅ **Type safe** - Compiler catches API changes
✅ **Full coverage** - All endpoints generated
✅ **Better errors** - Typed error responses
✅ **Async ready** - Built for modern Rust

## Next Steps for Full Integration

1. **Update Type Imports** - Fix type names in `main.rs` to use generated types
2. **Test Each Command** - Verify each CLI command works with new client
3. **Remove Old Code** - Delete `api.rs.old` once migration complete
4. **Optional**: Make CLI fully async for better UX with long operations

## Quick Test

Test the client library directly:

```bash
cd hindsight-clients/rust
cargo test
```

Build CLI with new client:

```bash
cd ../../hindsight-cli
cargo build
```
