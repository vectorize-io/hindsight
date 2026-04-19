mod api;
mod commands;
mod config;
mod errors;
mod output;
mod ui;
mod utils;

use anyhow::Result;
use api::ApiClient;
use clap::{Parser, Subcommand, ValueEnum};
use config::Config;
use output::OutputFormat;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Format {
    Pretty,
    Json,
    Yaml,
}

impl From<Format> for OutputFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::Pretty => OutputFormat::Pretty,
            Format::Json => OutputFormat::Json,
            Format::Yaml => OutputFormat::Yaml,
        }
    }
}

#[derive(Parser)]
#[command(name = "hindsight")]
#[command(about = "Hindsight CLI - Semantic memory system", long_about = None)]
#[command(version)]
#[command(before_help = get_before_help())]
#[command(after_help = get_after_help())]
struct Cli {
    /// Output format (pretty, json, yaml)
    #[arg(short = 'o', long, global = true, default_value = "pretty")]
    output: Format,

    /// Show verbose output including full requests and responses
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    /// Named profile to load from ~/.hindsight/cli-profiles/<name>.toml
    /// (env var HINDSIGHT_PROFILE is used if this flag is omitted).
    /// Environment variables (HINDSIGHT_API_URL / HINDSIGHT_API_KEY) still override profile values.
    #[arg(short = 'p', long, global = true, env = "HINDSIGHT_PROFILE")]
    profile: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

fn get_after_help() -> String {
    let config = config::Config::load().ok();
    let (api_url, source) = match &config {
        Some(c) => (c.api_url.as_str(), c.source.to_string()),
        None => ("http://localhost:8888", "default".to_string()),
    };
    format!(
        "Current API URL: {} (from {})\n\nRun 'hindsight configure' to change the API URL.",
        api_url, source
    )
}

fn get_before_help() -> &'static str {
    ui::get_logo()
}

#[derive(Subcommand)]
enum Commands {
    /// Manage banks (list, create, update, profile, stats, mission, graph, delete)
    #[command(subcommand)]
    Bank(BankCommands),

    /// Manage memories (list, get, recall, reflect, retain, clear)
    #[command(subcommand)]
    Memory(MemoryCommands),

    /// Manage documents (list, get, delete)
    #[command(subcommand)]
    Document(DocumentCommands),

    /// Manage entities (list, get, regenerate)
    #[command(subcommand)]
    Entity(EntityCommands),

    /// Manage tags (list)
    #[command(subcommand)]
    Tag(TagCommands),

    /// Manage chunks (get)
    #[command(subcommand)]
    Chunk(ChunkCommands),

    /// Manage async operations (list, get, cancel)
    #[command(subcommand)]
    Operation(OperationCommands),

    /// Manage mental models (user-curated summaries)
    #[command(subcommand)]
    MentalModel(MentalModelCommands),

    /// Manage directives (behavioral rules)
    #[command(subcommand)]
    Directive(DirectiveCommands),

    /// Manage webhooks (list, create, update, delete, deliveries)
    #[command(subcommand)]
    Webhook(WebhookCommands),

    /// Inspect audit logs (list, stats)
    #[command(subcommand)]
    Audit(AuditCommands),

    /// Check API health status
    Health,

    /// Get Prometheus metrics
    Metrics,

    /// Get API version information
    Version,

    /// Interactive TUI explorer (k9s-style) for navigating banks, memories, entities, and performing recall/reflect
    #[command(alias = "tui")]
    Explore,

    /// Launch the web-based control plane UI
    Ui,

    /// Configure the CLI (API URL, API key, etc.)
    #[command(
        after_help = "Configuration priority:\n  1. Environment variables (HINDSIGHT_API_URL, HINDSIGHT_API_KEY) - highest priority\n  2. Named profile (-p / HINDSIGHT_PROFILE, see 'hindsight profile')\n  3. Config file (~/.hindsight/config)\n  4. Default (http://localhost:8888)"
    )]
    Configure {
        /// API URL to connect to (interactive prompt if not provided)
        #[arg(long)]
        api_url: Option<String>,
        /// API key for authentication (sent as Bearer token)
        #[arg(long)]
        api_key: Option<String>,
    },

    /// Manage named connection profiles (~/.hindsight/cli-profiles/<name>.toml)
    #[command(subcommand)]
    Profile(ProfileCommands),
}

#[derive(Subcommand)]
enum ProfileCommands {
    /// Create or overwrite a profile
    Create {
        /// Profile name (used with -p/--profile or $HINDSIGHT_PROFILE)
        name: String,
        /// API URL (required)
        #[arg(long)]
        api_url: String,
        /// API key (optional; stored in profile file with 0600 permissions)
        #[arg(long)]
        api_key: Option<String>,
    },
    /// List all known profiles
    List,
    /// Show the contents of a profile
    Show {
        /// Profile name
        name: String,
    },
    /// Delete a profile
    Delete {
        /// Profile name
        name: String,
        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum BankCommands {
    /// List all banks
    List,

    /// Create a new bank
    Create {
        /// Bank ID
        bank_id: String,

        /// Bank name
        #[arg(short = 'n', long)]
        name: Option<String>,

        /// Mission statement
        #[arg(short = 'm', long)]
        mission: Option<String>,

        /// Skepticism trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        skepticism: Option<i64>,

        /// Literalism trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        literalism: Option<i64>,

        /// Empathy trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        empathy: Option<i64>,
    },

    /// Update bank properties (partial update)
    Update {
        /// Bank ID
        bank_id: String,

        /// Bank name
        #[arg(short = 'n', long)]
        name: Option<String>,

        /// Mission statement
        #[arg(short = 'm', long)]
        mission: Option<String>,

        /// Skepticism trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        skepticism: Option<i64>,

        /// Literalism trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        literalism: Option<i64>,

        /// Empathy trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        empathy: Option<i64>,
    },

    /// Get bank disposition and profile
    Disposition {
        /// Bank ID
        bank_id: String,
    },

    /// Get memory statistics for a bank
    Stats {
        /// Bank ID
        bank_id: String,
    },

    /// Set bank name
    Name {
        /// Bank ID
        bank_id: String,

        /// Bank name
        name: String,
    },

    /// Set bank mission
    Mission {
        /// Bank ID
        bank_id: String,

        /// Mission statement
        mission: String,
    },

    /// Set or merge bank background (deprecated: use mission instead)
    #[command(hide = true)]
    Background {
        /// Bank ID
        bank_id: String,

        /// Background content
        content: String,

        /// Skip automatic disposition inference
        #[arg(long)]
        no_update_disposition: bool,
    },

    /// Get memory graph data
    Graph {
        /// Bank ID
        bank_id: String,

        /// Filter by fact type (world, experience, opinion)
        #[arg(short = 't', long)]
        fact_type: Option<String>,

        /// Maximum nodes to return
        #[arg(short = 'l', long, default_value = "1000")]
        limit: i64,
    },

    /// Delete a bank and all its data
    Delete {
        /// Bank ID
        bank_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Trigger consolidation to create/update observations
    Consolidate {
        /// Bank ID
        bank_id: String,

        /// Wait for consolidation to complete (poll for status)
        #[arg(long)]
        wait: bool,

        /// Poll interval in seconds (only used with --wait)
        #[arg(long, default_value = "10")]
        poll_interval: u64,
    },

    /// Clear all observations for a bank
    ClearObservations {
        /// Bank ID
        bank_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Get bank configuration (hierarchical overrides)
    Config {
        /// Bank ID
        bank_id: String,

        /// Show only bank-specific overrides (not full resolved config)
        #[arg(long)]
        overrides_only: bool,
    },

    /// Update bank configuration (set hierarchical overrides)
    SetConfig {
        /// Bank ID
        bank_id: String,

        /// LLM provider override
        #[arg(long)]
        llm_provider: Option<String>,

        /// LLM model override
        #[arg(long)]
        llm_model: Option<String>,

        /// LLM API key override
        #[arg(long)]
        llm_api_key: Option<String>,

        /// LLM base URL override
        #[arg(long)]
        llm_base_url: Option<String>,

        /// Retain mission: what to focus on during fact extraction
        #[arg(long)]
        retain_mission: Option<String>,

        /// Retain extraction mode (concise, verbose, custom)
        #[arg(long)]
        retain_extraction_mode: Option<String>,

        /// Observations mission: what to synthesize into durable observations
        #[arg(long)]
        observations_mission: Option<String>,

        /// Reflect mission: first-person identity for reflect operations
        #[arg(long)]
        reflect_mission: Option<String>,

        /// Disposition skepticism trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        disposition_skepticism: Option<i64>,

        /// Disposition literalism trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        disposition_literalism: Option<i64>,

        /// Disposition empathy trait (1-5)
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..=5))]
        disposition_empathy: Option<i64>,
    },

    /// Reset bank configuration to defaults (remove all overrides)
    ResetConfig {
        /// Bank ID
        bank_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Set disposition traits directly (1-5 each, via PUT /profile)
    SetDisposition {
        /// Bank ID
        bank_id: String,

        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=5))]
        skepticism: u64,

        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=5))]
        literalism: u64,

        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=5))]
        empathy: u64,
    },

    /// Recover from a stalled consolidation
    ConsolidationRecover {
        /// Bank ID
        bank_id: String,
    },

    /// Export a bank template manifest (config + mental models + directives)
    ExportTemplate {
        /// Bank ID
        bank_id: String,

        /// Write manifest to this file instead of stdout
        #[arg(short = 'o', long)]
        out: Option<PathBuf>,
    },

    /// Import a bank template manifest from a JSON file
    ImportTemplate {
        /// Bank ID
        bank_id: String,

        /// Path to a JSON manifest file
        manifest: PathBuf,

        /// Validate the manifest without applying changes
        #[arg(long)]
        dry_run: bool,
    },

    /// Print the bank template JSON schema
    TemplateSchema,
}

#[derive(Subcommand)]
enum MemoryCommands {
    /// List memory units with pagination
    List {
        /// Bank ID
        bank_id: String,

        /// Filter by fact type (world, experience, opinion)
        #[arg(short = 't', long)]
        fact_type: Option<String>,

        /// Full-text search query
        #[arg(short = 'q', long)]
        query: Option<String>,

        /// Maximum number of results
        #[arg(short = 'l', long, default_value = "100")]
        limit: i64,

        /// Offset for pagination
        #[arg(short = 's', long, default_value = "0")]
        offset: i64,
    },

    /// Get a specific memory unit by ID
    Get {
        /// Bank ID
        bank_id: String,

        /// Memory unit ID
        memory_id: String,
    },

    /// Recall memories using semantic search
    Recall {
        /// Bank ID
        bank_id: String,

        /// Search query
        query: String,

        /// Fact types to search (world, experience, opinion)
        #[arg(short = 't', long, value_delimiter = ',', default_values = &["world", "experience", "opinion"])]
        fact_type: Vec<String>,

        /// Thinking budget (low, mid, high)
        #[arg(short = 'b', long, default_value = "mid")]
        budget: String,

        /// Maximum tokens for results
        #[arg(long, default_value = "4096")]
        max_tokens: i64,

        /// Show trace information
        #[arg(long)]
        trace: bool,

        /// Include chunks in results
        #[arg(long)]
        include_chunks: bool,

        /// Maximum tokens for chunks (only used with --include-chunks)
        #[arg(long, default_value = "8192")]
        chunk_max_tokens: i64,

        /// Filter by tags (comma-separated, e.g. user:alice,team)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,

        /// Tag matching mode: any, all, any_strict, all_strict (default: any)
        #[arg(long)]
        tags_match: Option<String>,

        /// Reference timestamp for recall (ISO 8601, e.g. 2023-05-30T23:40:00)
        #[arg(long)]
        query_timestamp: Option<String>,
    },

    /// Generate answers using bank identity (reflect/reasoning)
    Reflect {
        /// Bank ID
        bank_id: String,

        /// Query to reflect on
        query: String,

        /// Thinking budget (low, mid, high)
        #[arg(short = 'b', long, default_value = "mid")]
        budget: String,

        /// Additional context
        #[arg(short = 'c', long)]
        context: Option<String>,

        /// Maximum tokens for the response (server default: 4096)
        #[arg(short = 'm', long)]
        max_tokens: Option<i64>,

        /// Path to JSON schema file for structured output
        #[arg(short = 's', long)]
        schema: Option<PathBuf>,

        /// Filter by tags (comma-separated, e.g. user:alice,team)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,

        /// Tag matching mode: any, all, any_strict, all_strict (default: any)
        #[arg(long)]
        tags_match: Option<String>,

        /// Include source facts (based_on) in the response
        #[arg(long)]
        include_facts: bool,

        /// Restrict fact retrieval to these fact types (comma-separated: world, experience, observation)
        #[arg(long, value_delimiter = ',')]
        fact_types: Option<Vec<String>>,

        /// Exclude all mental models from the reflect loop
        #[arg(long)]
        exclude_mental_models: bool,

        /// Exclude specific mental models by ID (comma-separated)
        #[arg(long, value_delimiter = ',')]
        exclude_mental_model_ids: Option<Vec<String>>,
    },

    /// Store (retain) a single memory
    Retain {
        /// Bank ID
        bank_id: String,

        /// Memory content
        content: String,

        /// Document ID (auto-generated if not provided)
        #[arg(short = 'd', long)]
        doc_id: Option<String>,

        /// Context for the memory
        #[arg(short = 'c', long)]
        context: Option<String>,

        /// Queue for background processing
        #[arg(long)]
        r#async: bool,

        /// Deprecated document-level tags (comma-separated). Prefer item-level tags.
        #[arg(long, value_delimiter = ',')]
        document_tags: Option<Vec<String>>,
    },

    /// Bulk import memories from files (retain)
    RetainFiles {
        /// Bank ID
        bank_id: String,

        /// Path to file or directory
        path: PathBuf,

        /// Search directories recursively
        #[arg(short = 'r', long, default_value = "true")]
        recursive: bool,

        /// Context for all memories
        #[arg(short = 'c', long)]
        context: Option<String>,

        /// Queue for background processing
        #[arg(long)]
        r#async: bool,
    },

    /// Delete a memory unit
    Delete {
        /// Bank ID
        bank_id: String,

        /// Memory unit ID
        unit_id: String,
    },

    /// Clear all memories for a bank
    Clear {
        /// Bank ID
        bank_id: String,

        /// Fact type to clear (world, agent, opinion). If not specified, clears all types.
        #[arg(short = 't', long, value_parser = ["world", "agent", "opinion"])]
        fact_type: Option<String>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Show the observation history for a memory unit
    History {
        /// Bank ID
        bank_id: String,

        /// Memory unit ID
        memory_id: String,
    },

    /// Clear the observations derived from a single memory unit
    ClearObservations {
        /// Bank ID
        bank_id: String,

        /// Memory unit ID
        memory_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum DocumentCommands {
    /// List documents for a bank
    List {
        /// Bank ID
        bank_id: String,

        /// Search query to filter documents
        #[arg(short = 'q', long)]
        query: Option<String>,

        /// Filter by date (yesterday, today, YYYY-MM-DD, or all)
        #[arg(short = 'd', long)]
        date: Option<String>,

        /// Maximum number of results
        #[arg(short = 'l', long, default_value = "100")]
        limit: i32,

        /// Offset for pagination
        #[arg(short = 's', long, default_value = "0")]
        offset: i32,
    },

    /// Get a specific document by ID
    Get {
        /// Bank ID
        bank_id: String,

        /// Document ID
        document_id: String,
    },

    /// Delete a document and all its memory units
    Delete {
        /// Bank ID
        bank_id: String,

        /// Document ID
        document_id: String,
    },

    /// Update a document (currently only supports replacing tags)
    Update {
        /// Bank ID
        bank_id: String,

        /// Document ID
        document_id: String,

        /// New tag list (comma-separated). Triggers observation invalidation + re-consolidation.
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
    },
}

#[derive(Subcommand)]
enum EntityCommands {
    /// List entities for a bank
    List {
        /// Bank ID
        bank_id: String,

        /// Maximum number of results
        #[arg(short = 'l', long, default_value = "100")]
        limit: i64,
    },

    /// Get detailed information about an entity
    Get {
        /// Bank ID
        bank_id: String,

        /// Entity ID
        entity_id: String,
    },

    /// Regenerate observations for an entity
    Regenerate {
        /// Bank ID
        bank_id: String,

        /// Entity ID
        entity_id: String,
    },
}

#[derive(Subcommand)]
enum OperationCommands {
    /// List async operations for a bank
    List {
        /// Bank ID
        bank_id: String,
    },

    /// Get the status of a specific operation
    Get {
        /// Bank ID
        bank_id: String,

        /// Operation ID
        operation_id: String,
    },

    /// Cancel a pending async operation
    Cancel {
        /// Bank ID
        bank_id: String,

        /// Operation ID
        operation_id: String,
    },

    /// Retry a failed async operation
    Retry {
        /// Bank ID
        bank_id: String,

        /// Operation ID
        operation_id: String,
    },
}

#[derive(Subcommand)]
enum WebhookCommands {
    /// List webhooks configured for a bank
    List {
        /// Bank ID
        bank_id: String,
    },

    /// Create a new webhook
    Create {
        /// Bank ID
        bank_id: String,

        /// Target URL (http/https)
        url: String,

        /// Event types (comma-separated). Defaults to consolidation.completed
        #[arg(long, value_delimiter = ',')]
        event_types: Vec<String>,

        /// Start disabled
        #[arg(long)]
        disabled: bool,

        /// HMAC-SHA256 signing secret
        #[arg(long)]
        secret: Option<String>,
    },

    /// Update an existing webhook
    Update {
        /// Bank ID
        bank_id: String,

        /// Webhook ID
        webhook_id: String,

        /// New target URL
        #[arg(long)]
        url: Option<String>,

        /// Replace event types (comma-separated)
        #[arg(long, value_delimiter = ',')]
        event_types: Option<Vec<String>>,

        /// Enable or disable
        #[arg(long)]
        enabled: Option<bool>,

        /// Replace the signing secret
        #[arg(long)]
        secret: Option<String>,
    },

    /// Delete a webhook
    Delete {
        /// Bank ID
        bank_id: String,

        /// Webhook ID
        webhook_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// List recent delivery attempts for a webhook
    Deliveries {
        /// Bank ID
        bank_id: String,

        /// Webhook ID
        webhook_id: String,

        /// Pagination cursor
        #[arg(long)]
        cursor: Option<String>,

        /// Maximum number of deliveries to return
        #[arg(short = 'l', long)]
        limit: Option<i64>,
    },
}

#[derive(Subcommand)]
enum AuditCommands {
    /// List audit log entries for a bank
    List {
        /// Bank ID
        bank_id: String,

        /// Filter by action (e.g. recall, retain)
        #[arg(long)]
        action: Option<String>,

        /// Filter by transport (e.g. http, mcp)
        #[arg(long)]
        transport: Option<String>,

        /// Start date/time (ISO 8601)
        #[arg(long)]
        start_date: Option<String>,

        /// End date/time (ISO 8601)
        #[arg(long)]
        end_date: Option<String>,

        /// Maximum number of entries
        #[arg(short = 'l', long)]
        limit: Option<u64>,

        /// Offset for pagination
        #[arg(short = 's', long)]
        offset: Option<u64>,
    },

    /// Show audit log statistics bucketed over time
    Stats {
        /// Bank ID
        bank_id: String,

        /// Filter by action
        #[arg(long)]
        action: Option<String>,

        /// Time period (e.g. day, week, month)
        #[arg(long)]
        period: Option<String>,
    },
}

#[derive(Subcommand)]
enum TagCommands {
    /// List tags in a bank
    List {
        /// Bank ID
        bank_id: String,

        /// Wildcard search query (e.g., 'user:*')
        #[arg(short = 'q', long)]
        query: Option<String>,

        /// Maximum number of results
        #[arg(short = 'l', long, default_value = "100")]
        limit: i64,

        /// Offset for pagination
        #[arg(short = 's', long, default_value = "0")]
        offset: i64,
    },
}

#[derive(Subcommand)]
enum ChunkCommands {
    /// Get a specific chunk by ID
    Get {
        /// Chunk ID
        chunk_id: String,
    },
}

#[derive(Subcommand)]
enum MentalModelCommands {
    /// List mental models for a bank
    List {
        /// Bank ID
        bank_id: String,
    },

    /// Get a specific mental model
    Get {
        /// Bank ID
        bank_id: String,

        /// Mental model ID
        mental_model_id: String,
    },

    /// Create a new mental model
    Create {
        /// Bank ID
        bank_id: String,

        /// Mental model name
        name: String,

        /// Source query to generate the mental model from
        source_query: String,

        /// Optional custom ID for the mental model (alphanumeric lowercase with hyphens)
        #[arg(long)]
        id: Option<String>,

        /// Tags for scoped visibility (comma-separated)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,

        /// Maximum tokens for generated content (256-8192)
        #[arg(long, default_value = "2048")]
        max_tokens: i64,

        /// Refresh this mental model automatically after observations consolidation
        #[arg(long)]
        trigger_refresh_after_consolidation: bool,
    },

    /// Update a mental model
    Update {
        /// Bank ID
        bank_id: String,

        /// Mental model ID
        mental_model_id: String,

        /// New name
        #[arg(long)]
        name: Option<String>,

        /// New source query
        #[arg(long)]
        source_query: Option<String>,

        /// New maximum tokens for generated content
        #[arg(long)]
        max_tokens: Option<i64>,

        /// Replace tags (comma-separated)
        #[arg(long, value_delimiter = ',')]
        tags: Option<Vec<String>>,

        /// Enable/disable automatic refresh after observations consolidation
        #[arg(long)]
        trigger_refresh_after_consolidation: Option<bool>,
    },

    /// Delete a mental model
    Delete {
        /// Bank ID
        bank_id: String,

        /// Mental model ID
        mental_model_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Refresh a mental model (re-run the source query)
    Refresh {
        /// Bank ID
        bank_id: String,

        /// Mental model ID
        mental_model_id: String,
    },

    /// Get the change history of a mental model
    History {
        /// Bank ID
        bank_id: String,

        /// Mental model ID
        mental_model_id: String,
    },
}

#[derive(Subcommand)]
enum DirectiveCommands {
    /// List directives for a bank
    List {
        /// Bank ID
        bank_id: String,
    },

    /// Get a specific directive
    Get {
        /// Bank ID
        bank_id: String,

        /// Directive ID
        directive_id: String,
    },

    /// Create a new directive
    Create {
        /// Bank ID
        bank_id: String,

        /// Directive name
        name: String,

        /// Directive content (the text to inject into prompts)
        content: String,

        /// Priority — higher-priority directives are injected first
        #[arg(long, default_value = "0")]
        priority: i64,
    },

    /// Update a directive
    Update {
        /// Bank ID
        bank_id: String,

        /// Directive ID
        directive_id: String,

        /// New name
        #[arg(long)]
        name: Option<String>,

        /// New content
        #[arg(long)]
        content: Option<String>,

        /// Enable or disable the directive
        #[arg(long)]
        is_active: Option<bool>,

        /// New priority (higher = injected first)
        #[arg(long)]
        priority: Option<i64>,
    },

    /// Delete a directive
    Delete {
        /// Bank ID
        bank_id: String,

        /// Directive ID
        directive_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

fn main() {
    if let Err(e) = run() {
        ui::print_error(&format!("{:#}", e));
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let output_format: OutputFormat = cli.output.into();
    let verbose = cli.verbose;
    let profile = cli.profile.clone();

    // Handle configure command before loading full config (it doesn't need API client)
    if let Commands::Configure { api_url, api_key } = cli.command {
        return handle_configure(api_url, api_key, output_format);
    }

    // Handle profile management commands — no API client required.
    if let Commands::Profile(cmd) = cli.command {
        return handle_profile(cmd, output_format);
    }

    // Handle ui command - needs config but not API client
    if let Commands::Ui = cli.command {
        return handle_ui(profile.as_deref(), output_format);
    }

    // Load configuration
    let config = Config::load_with_profile(profile.as_deref()).unwrap_or_else(|e| {
        ui::print_error(&format!("Configuration error: {}", e));
        errors::print_config_help();
        std::process::exit(1);
    });

    let api_url = config.api_url().to_string();
    let api_key = config.api_key.clone();

    // Create API client
    let client = ApiClient::new(api_url.clone(), api_key).unwrap_or_else(|e| {
        errors::handle_api_error(e, &api_url);
    });

    // Execute command and handle errors
    let result: Result<()> = match cli.command {
        Commands::Configure { .. } => unreachable!(), // Handled above
        Commands::Profile(_) => unreachable!(),       // Handled above
        Commands::Ui => unreachable!(),               // Handled above
        Commands::Explore => commands::explore::run(&client),

        // Health, Metrics, and Version
        Commands::Health => commands::health::health(&client, verbose, output_format),
        Commands::Metrics => commands::health::metrics(&client, verbose, output_format),
        Commands::Version => commands::health::version(&client, verbose, output_format),

        // Bank commands
        Commands::Bank(bank_cmd) => match bank_cmd {
            BankCommands::List => commands::bank::list(&client, verbose, output_format),
            BankCommands::Create {
                bank_id,
                name,
                mission,
                skepticism,
                literalism,
                empathy,
            } => commands::bank::create(
                &client,
                &bank_id,
                name,
                mission,
                skepticism,
                literalism,
                empathy,
                verbose,
                output_format,
            ),
            BankCommands::Update {
                bank_id,
                name,
                mission,
                skepticism,
                literalism,
                empathy,
            } => commands::bank::update(
                &client,
                &bank_id,
                name,
                mission,
                skepticism,
                literalism,
                empathy,
                verbose,
                output_format,
            ),
            BankCommands::Disposition { bank_id } => {
                commands::bank::disposition(&client, &bank_id, verbose, output_format)
            }
            BankCommands::Stats { bank_id } => {
                commands::bank::stats(&client, &bank_id, verbose, output_format)
            }
            BankCommands::Name { bank_id, name } => {
                commands::bank::update_name(&client, &bank_id, &name, verbose, output_format)
            }
            BankCommands::Mission { bank_id, mission } => {
                commands::bank::mission(&client, &bank_id, &mission, verbose, output_format)
            }
            BankCommands::Background {
                bank_id,
                content,
                no_update_disposition,
            } => commands::bank::update_background(
                &client,
                &bank_id,
                &content,
                no_update_disposition,
                verbose,
                output_format,
            ),
            BankCommands::Graph {
                bank_id,
                fact_type,
                limit,
            } => commands::bank::graph(&client, &bank_id, fact_type, limit, verbose, output_format),
            BankCommands::Delete { bank_id, yes } => {
                commands::bank::delete(&client, &bank_id, yes, verbose, output_format)
            }
            BankCommands::Consolidate {
                bank_id,
                wait,
                poll_interval,
            } => commands::bank::consolidate(
                &client,
                &bank_id,
                wait,
                poll_interval,
                verbose,
                output_format,
            ),
            BankCommands::ClearObservations { bank_id, yes } => {
                commands::bank::clear_observations(&client, &bank_id, yes, verbose, output_format)
            }
            BankCommands::Config {
                bank_id,
                overrides_only,
            } => commands::bank::config(&client, &bank_id, overrides_only, verbose, output_format),
            BankCommands::SetConfig {
                bank_id,
                llm_provider,
                llm_model,
                llm_api_key,
                llm_base_url,
                retain_mission,
                retain_extraction_mode,
                observations_mission,
                reflect_mission,
                disposition_skepticism,
                disposition_literalism,
                disposition_empathy,
            } => commands::bank::set_config(
                &client,
                &bank_id,
                llm_provider,
                llm_model,
                llm_api_key,
                llm_base_url,
                retain_mission,
                retain_extraction_mode,
                observations_mission,
                reflect_mission,
                disposition_skepticism,
                disposition_literalism,
                disposition_empathy,
                verbose,
                output_format,
            ),
            BankCommands::ResetConfig { bank_id, yes } => {
                commands::bank::reset_config(&client, &bank_id, yes, verbose, output_format)
            }
            BankCommands::SetDisposition {
                bank_id,
                skepticism,
                literalism,
                empathy,
            } => commands::bank::set_disposition(
                &client,
                &bank_id,
                skepticism,
                literalism,
                empathy,
                verbose,
                output_format,
            ),
            BankCommands::ConsolidationRecover { bank_id } => {
                commands::bank::consolidation_recover(&client, &bank_id, verbose, output_format)
            }
            BankCommands::ExportTemplate { bank_id, out } => {
                commands::bank::export_template(&client, &bank_id, out, verbose, output_format)
            }
            BankCommands::ImportTemplate {
                bank_id,
                manifest,
                dry_run,
            } => commands::bank::import_template(
                &client,
                &bank_id,
                &manifest,
                dry_run,
                verbose,
                output_format,
            ),
            BankCommands::TemplateSchema => {
                commands::bank::template_schema(&client, verbose, output_format)
            }
        },

        // Memory commands
        Commands::Memory(memory_cmd) => match memory_cmd {
            MemoryCommands::List {
                bank_id,
                fact_type,
                query,
                limit,
                offset,
            } => commands::memory::list(
                &client,
                &bank_id,
                fact_type,
                query,
                limit,
                offset,
                verbose,
                output_format,
            ),
            MemoryCommands::Get { bank_id, memory_id } => {
                commands::memory::get(&client, &bank_id, &memory_id, verbose, output_format)
            }
            MemoryCommands::Recall {
                bank_id,
                query,
                fact_type,
                budget,
                max_tokens,
                trace,
                include_chunks,
                chunk_max_tokens,
                tags,
                tags_match,
                query_timestamp,
            } => commands::memory::recall(
                &client,
                &bank_id,
                query,
                fact_type,
                budget,
                max_tokens,
                trace,
                include_chunks,
                chunk_max_tokens,
                tags,
                tags_match,
                query_timestamp,
                verbose,
                output_format,
            ),
            MemoryCommands::Reflect {
                bank_id,
                query,
                budget,
                context,
                max_tokens,
                schema,
                tags,
                tags_match,
                include_facts,
                fact_types,
                exclude_mental_models,
                exclude_mental_model_ids,
            } => commands::memory::reflect(
                &client,
                &bank_id,
                query,
                budget,
                context,
                max_tokens,
                schema,
                tags,
                tags_match,
                include_facts,
                fact_types,
                exclude_mental_models,
                exclude_mental_model_ids,
                verbose,
                output_format,
            ),
            MemoryCommands::Retain {
                bank_id,
                content,
                doc_id,
                context,
                r#async,
                document_tags,
            } => commands::memory::retain(
                &client,
                &bank_id,
                content,
                doc_id,
                context,
                r#async,
                document_tags,
                verbose,
                output_format,
            ),
            MemoryCommands::RetainFiles {
                bank_id,
                path,
                recursive,
                context,
                r#async,
            } => commands::memory::retain_files(
                &client,
                &bank_id,
                path,
                recursive,
                context,
                r#async,
                verbose,
                output_format,
            ),
            MemoryCommands::Delete { bank_id, unit_id } => {
                commands::memory::delete(&client, &bank_id, &unit_id, verbose, output_format)
            }
            MemoryCommands::Clear {
                bank_id,
                fact_type,
                yes,
            } => commands::memory::clear(&client, &bank_id, fact_type, yes, verbose, output_format),
            MemoryCommands::History { bank_id, memory_id } => {
                commands::memory::history(&client, &bank_id, &memory_id, verbose, output_format)
            }
            MemoryCommands::ClearObservations {
                bank_id,
                memory_id,
                yes,
            } => commands::memory::clear_observations(
                &client,
                &bank_id,
                &memory_id,
                yes,
                verbose,
                output_format,
            ),
        },

        // Document commands
        Commands::Document(doc_cmd) => match doc_cmd {
            DocumentCommands::List {
                bank_id,
                query,
                date,
                limit,
                offset,
            } => commands::document::list(
                &client,
                &bank_id,
                query,
                date,
                limit,
                offset,
                verbose,
                output_format,
            ),
            DocumentCommands::Get {
                bank_id,
                document_id,
            } => commands::document::get(&client, &bank_id, &document_id, verbose, output_format),
            DocumentCommands::Delete {
                bank_id,
                document_id,
            } => {
                commands::document::delete(&client, &bank_id, &document_id, verbose, output_format)
            }
            DocumentCommands::Update {
                bank_id,
                document_id,
                tags,
            } => {
                let tag_opt = if tags.is_empty() { None } else { Some(tags) };
                commands::document::update(
                    &client,
                    &bank_id,
                    &document_id,
                    tag_opt,
                    verbose,
                    output_format,
                )
            }
        },

        // Entity commands
        Commands::Entity(entity_cmd) => match entity_cmd {
            EntityCommands::List { bank_id, limit } => {
                commands::entity::list(&client, &bank_id, limit, verbose, output_format)
            }
            EntityCommands::Get { bank_id, entity_id } => {
                commands::entity::get(&client, &bank_id, &entity_id, verbose, output_format)
            }
            EntityCommands::Regenerate { bank_id, entity_id } => {
                commands::entity::regenerate(&client, &bank_id, &entity_id, verbose, output_format)
            }
        },

        // Tag commands
        Commands::Tag(tag_cmd) => match tag_cmd {
            TagCommands::List {
                bank_id,
                query,
                limit,
                offset,
            } => commands::tag::list(
                &client,
                &bank_id,
                query,
                limit,
                offset,
                verbose,
                output_format,
            ),
        },

        // Chunk commands
        Commands::Chunk(chunk_cmd) => match chunk_cmd {
            ChunkCommands::Get { chunk_id } => {
                commands::chunk::get(&client, &chunk_id, verbose, output_format)
            }
        },

        // Operation commands
        Commands::Operation(op_cmd) => match op_cmd {
            OperationCommands::List { bank_id } => {
                commands::operation::list(&client, &bank_id, verbose, output_format)
            }
            OperationCommands::Get {
                bank_id,
                operation_id,
            } => commands::operation::get(&client, &bank_id, &operation_id, verbose, output_format),
            OperationCommands::Cancel {
                bank_id,
                operation_id,
            } => commands::operation::cancel(
                &client,
                &bank_id,
                &operation_id,
                verbose,
                output_format,
            ),
            OperationCommands::Retry {
                bank_id,
                operation_id,
            } => {
                commands::operation::retry(&client, &bank_id, &operation_id, verbose, output_format)
            }
        },

        // Mental model commands
        Commands::MentalModel(mm_cmd) => match mm_cmd {
            MentalModelCommands::List { bank_id } => {
                commands::mental_model::list(&client, &bank_id, verbose, output_format)
            }
            MentalModelCommands::Get {
                bank_id,
                mental_model_id,
            } => commands::mental_model::get(
                &client,
                &bank_id,
                &mental_model_id,
                verbose,
                output_format,
            ),
            MentalModelCommands::Create {
                bank_id,
                name,
                source_query,
                id,
                tags,
                max_tokens,
                trigger_refresh_after_consolidation,
            } => commands::mental_model::create(
                &client,
                &bank_id,
                &name,
                &source_query,
                id.as_deref(),
                tags,
                max_tokens,
                trigger_refresh_after_consolidation,
                verbose,
                output_format,
            ),
            MentalModelCommands::Update {
                bank_id,
                mental_model_id,
                name,
                source_query,
                max_tokens,
                tags,
                trigger_refresh_after_consolidation,
            } => commands::mental_model::update(
                &client,
                &bank_id,
                &mental_model_id,
                name,
                source_query,
                max_tokens,
                tags,
                trigger_refresh_after_consolidation,
                verbose,
                output_format,
            ),
            MentalModelCommands::Delete {
                bank_id,
                mental_model_id,
                yes,
            } => commands::mental_model::delete(
                &client,
                &bank_id,
                &mental_model_id,
                yes,
                verbose,
                output_format,
            ),
            MentalModelCommands::Refresh {
                bank_id,
                mental_model_id,
            } => commands::mental_model::refresh(
                &client,
                &bank_id,
                &mental_model_id,
                verbose,
                output_format,
            ),
            MentalModelCommands::History {
                bank_id,
                mental_model_id,
            } => commands::mental_model::history(
                &client,
                &bank_id,
                &mental_model_id,
                verbose,
                output_format,
            ),
        },

        // Directive commands
        Commands::Directive(dir_cmd) => match dir_cmd {
            DirectiveCommands::List { bank_id } => {
                commands::directive::list(&client, &bank_id, verbose, output_format)
            }
            DirectiveCommands::Get {
                bank_id,
                directive_id,
            } => commands::directive::get(&client, &bank_id, &directive_id, verbose, output_format),
            DirectiveCommands::Create {
                bank_id,
                name,
                content,
                priority,
            } => commands::directive::create(
                &client,
                &bank_id,
                &name,
                &content,
                priority,
                verbose,
                output_format,
            ),
            DirectiveCommands::Update {
                bank_id,
                directive_id,
                name,
                content,
                is_active,
                priority,
            } => commands::directive::update(
                &client,
                &bank_id,
                &directive_id,
                name,
                content,
                is_active,
                priority,
                verbose,
                output_format,
            ),
            DirectiveCommands::Delete {
                bank_id,
                directive_id,
                yes,
            } => commands::directive::delete(
                &client,
                &bank_id,
                &directive_id,
                yes,
                verbose,
                output_format,
            ),
        },

        // Webhook commands
        Commands::Webhook(wh_cmd) => match wh_cmd {
            WebhookCommands::List { bank_id } => {
                commands::webhook::list(&client, &bank_id, verbose, output_format)
            }
            WebhookCommands::Create {
                bank_id,
                url,
                event_types,
                disabled,
                secret,
            } => commands::webhook::create(
                &client,
                &bank_id,
                &url,
                event_types,
                !disabled,
                secret,
                verbose,
                output_format,
            ),
            WebhookCommands::Update {
                bank_id,
                webhook_id,
                url,
                event_types,
                enabled,
                secret,
            } => commands::webhook::update(
                &client,
                &bank_id,
                &webhook_id,
                url,
                event_types,
                enabled,
                secret,
                verbose,
                output_format,
            ),
            WebhookCommands::Delete {
                bank_id,
                webhook_id,
                yes,
            } => commands::webhook::delete(
                &client,
                &bank_id,
                &webhook_id,
                yes,
                verbose,
                output_format,
            ),
            WebhookCommands::Deliveries {
                bank_id,
                webhook_id,
                cursor,
                limit,
            } => commands::webhook::deliveries(
                &client,
                &bank_id,
                &webhook_id,
                cursor,
                limit,
                verbose,
                output_format,
            ),
        },

        // Audit commands
        Commands::Audit(audit_cmd) => match audit_cmd {
            AuditCommands::List {
                bank_id,
                action,
                transport,
                start_date,
                end_date,
                limit,
                offset,
            } => commands::audit::list(
                &client,
                &bank_id,
                action,
                transport,
                start_date,
                end_date,
                limit,
                offset,
                verbose,
                output_format,
            ),
            AuditCommands::Stats {
                bank_id,
                action,
                period,
            } => commands::audit::stats(&client, &bank_id, action, period, verbose, output_format),
        },
    };

    // Handle API errors with nice messages
    if let Err(e) = result {
        errors::handle_api_error(e, &api_url);
    }

    Ok(())
}

fn handle_configure(
    api_url: Option<String>,
    api_key: Option<String>,
    output_format: OutputFormat,
) -> Result<()> {
    // Load current config to show current state
    let current_config = Config::load().ok();

    if output_format == OutputFormat::Pretty {
        ui::print_info("Hindsight CLI Configuration");
        println!();

        // Show current configuration
        if let Some(ref config) = current_config {
            println!("  Current API URL: {}", config.api_url);
            if let Some(ref key) = config.api_key {
                // Mask the API key for display
                let masked = if key.len() > 8 {
                    format!("{}...{}", &key[..4], &key[key.len() - 4..])
                } else {
                    "****".to_string()
                };
                println!("  Current API Key: {}", masked);
            }
            println!("  Source: {}", config.source);
            println!();
        }
    }

    // Get the new API URL (from argument or prompt)
    let new_api_url = match api_url {
        Some(url) => url,
        None => {
            // Interactive prompt
            let current = current_config.as_ref().map(|c| c.api_url.as_str());
            config::prompt_api_url(current)?
        }
    };

    // Validate the URL
    if !new_api_url.starts_with("http://") && !new_api_url.starts_with("https://") {
        ui::print_error(&format!(
            "Invalid API URL: {}. Must start with http:// or https://",
            new_api_url
        ));
        return Ok(());
    }

    // Use provided api_key, or keep existing one if not provided
    let new_api_key = api_key.or_else(|| current_config.as_ref().and_then(|c| c.api_key.clone()));

    // Save to config file
    let config_path = Config::save_config(&new_api_url, new_api_key.as_deref())?;

    if output_format == OutputFormat::Pretty {
        ui::print_success(&format!("Configuration saved to {}", config_path.display()));
        println!();
        println!("  API URL: {}", new_api_url);
        if let Some(ref key) = new_api_key {
            let masked = if key.len() > 8 {
                format!("{}...{}", &key[..4], &key[key.len() - 4..])
            } else {
                "****".to_string()
            };
            println!("  API Key: {}", masked);
        }
        println!();
        println!("Note: Environment variables HINDSIGHT_API_URL and HINDSIGHT_API_KEY will override these settings.");
    } else {
        let result = serde_json::json!({
            "api_url": new_api_url,
            "api_key_set": new_api_key.is_some(),
            "config_path": config_path.display().to_string(),
        });
        output::print_output(&result, output_format)?;
    }

    Ok(())
}

fn handle_ui(profile: Option<&str>, output_format: OutputFormat) -> Result<()> {
    use std::process::Command;

    // Load configuration to get the API URL
    let config = Config::load_with_profile(profile).unwrap_or_else(|e| {
        ui::print_error(&format!("Configuration error: {}", e));
        errors::print_config_help();
        std::process::exit(1);
    });

    let api_url = config.api_url();

    if output_format == OutputFormat::Pretty {
        ui::print_info("Launching Hindsight Control Plane UI...");
        println!();
        println!("  API URL: {}", api_url);
        println!();
    }

    // Run npx @vectorize-io/hindsight-control-plane --api-url {api_url}
    let status = Command::new("npx")
        .arg("@vectorize-io/hindsight-control-plane")
        .arg("--api-url")
        .arg(api_url)
        .status();

    match status {
        Ok(exit_status) => {
            if !exit_status.success() {
                if let Some(code) = exit_status.code() {
                    std::process::exit(code);
                } else {
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            ui::print_error(&format!("Failed to launch control plane UI: {}", e));
            ui::print_info("Make sure you have Node.js and npm installed.");
            ui::print_info("You can also install the control plane globally: npm install -g @vectorize-io/hindsight-control-plane");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn mask_api_key(key: &str) -> String {
    if key.len() > 8 {
        format!("{}...{}", &key[..4], &key[key.len() - 4..])
    } else {
        "****".to_string()
    }
}

fn handle_profile(cmd: ProfileCommands, output_format: OutputFormat) -> Result<()> {
    match cmd {
        ProfileCommands::Create {
            name,
            api_url,
            api_key,
        } => {
            let path = Config::save_profile(&name, &api_url, api_key.as_deref())?;
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Profile '{}' saved to {}", name, path.display()));
                println!();
                println!("  API URL: {}", api_url);
                if let Some(ref key) = api_key {
                    println!("  API Key: {}", mask_api_key(key));
                }
                println!();
                println!(
                    "Use with: hindsight -p {} <command>  (or export HINDSIGHT_PROFILE={})",
                    name, name
                );
            } else {
                let result = serde_json::json!({
                    "name": name,
                    "api_url": api_url,
                    "api_key_set": api_key.is_some(),
                    "path": path.display().to_string(),
                });
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        ProfileCommands::List => {
            let names = Config::list_profiles()?;
            if output_format == OutputFormat::Pretty {
                if names.is_empty() {
                    ui::print_info("No profiles found.");
                    println!();
                    println!("Create one with: hindsight profile create <name> --api-url <url>");
                } else {
                    ui::print_info("Profiles:");
                    for name in &names {
                        println!("  • {}", name);
                    }
                }
            } else {
                output::print_output(&serde_json::json!({ "profiles": names }), output_format)?;
            }
            Ok(())
        }
        ProfileCommands::Show { name } => {
            let (api_url, api_key) = Config::load_profile(&name)?;
            let path = Config::profile_file_path(&name)
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            if output_format == OutputFormat::Pretty {
                ui::print_info(&format!("Profile '{}'", name));
                println!();
                println!("  Path:    {}", path);
                println!("  API URL: {}", api_url);
                if let Some(ref key) = api_key {
                    println!("  API Key: {}", mask_api_key(key));
                }
            } else {
                let result = serde_json::json!({
                    "name": name,
                    "path": path,
                    "api_url": api_url,
                    "api_key_set": api_key.is_some(),
                });
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        ProfileCommands::Delete { name, yes } => {
            let path = Config::profile_file_path(&name)
                .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
            if !path.exists() {
                anyhow::bail!("profile '{}' not found at {}", name, path.display());
            }
            if !yes && output_format == OutputFormat::Pretty {
                print!("Delete profile '{}' at {}? [y/N]: ", name, path.display());
                std::io::Write::flush(&mut std::io::stdout())?;
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
                    ui::print_info("Aborted.");
                    return Ok(());
                }
            }
            let deleted = Config::delete_profile(&name)?;
            if output_format == OutputFormat::Pretty {
                ui::print_success(&format!("Deleted profile '{}' ({})", name, deleted.display()));
            } else {
                output::print_output(
                    &serde_json::json!({
                        "name": name,
                        "path": deleted.display().to_string(),
                        "deleted": true,
                    }),
                    output_format,
                )?;
            }
            Ok(())
        }
    }
}
