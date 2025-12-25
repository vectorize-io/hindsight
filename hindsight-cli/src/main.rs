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
    /// Manage banks (list, profile, stats)
    #[command(subcommand)]
    Bank(BankCommands),

    /// Manage memories (recall, reflect, retain, delete)
    #[command(subcommand)]
    Memory(MemoryCommands),

    /// Manage documents (list, get, delete)
    #[command(subcommand)]
    Document(DocumentCommands),

    /// Manage entities (list, get, regenerate)
    #[command(subcommand)]
    Entity(EntityCommands),

    /// Manage async operations (list, cancel)
    #[command(subcommand)]
    Operation(OperationCommands),

    /// Interactive TUI explorer (k9s-style) for navigating banks, memories, entities, and performing recall/reflect
    #[command(alias = "tui")]
    Explore,

    /// Launch the web-based control plane UI
    Ui,

    /// Configure the CLI (API URL, API key, etc.)
    #[command(after_help = "Configuration priority:\n  1. Environment variables (HINDSIGHT_API_URL, HINDSIGHT_API_KEY) - highest priority\n  2. Config file (~/.hindsight/config)\n  3. Default (http://localhost:8888)")]
    Configure {
        /// API URL to connect to (interactive prompt if not provided)
        #[arg(long)]
        api_url: Option<String>,
        /// API key for authentication (sent as Bearer token)
        #[arg(long)]
        api_key: Option<String>,
    },
}

#[derive(Subcommand)]
enum BankCommands {
    /// List all banks
    List,

    /// Get bank disposition and background
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

    /// Set or merge bank background
    Background {
        /// Bank ID
        bank_id: String,

        /// Background content
        content: String,

        /// Skip automatic disposition inference
        #[arg(long)]
        no_update_disposition: bool,
    },

    /// Delete a bank and all its data
    Delete {
        /// Bank ID
        bank_id: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum MemoryCommands {
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

    /// Cancel a pending async operation
    Cancel {
        /// Bank ID
        bank_id: String,

        /// Operation ID
        operation_id: String,
    },
}

fn main() {
    if let Err(_) = run() {
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let output_format: OutputFormat = cli.output.into();
    let verbose = cli.verbose;

    // Handle configure command before loading full config (it doesn't need API client)
    if let Commands::Configure { api_url, api_key } = cli.command {
        return handle_configure(api_url, api_key, output_format);
    }

    // Handle ui command - needs config but not API client
    if let Commands::Ui = cli.command {
        return handle_ui(output_format);
    }

    // Load configuration
    let config = Config::from_env().unwrap_or_else(|e| {
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
        Commands::Ui => unreachable!(), // Handled above
        Commands::Explore => commands::explore::run(&client),
        Commands::Bank(bank_cmd) => match bank_cmd {
            BankCommands::List => commands::bank::list(&client, verbose, output_format),
            BankCommands::Disposition { bank_id } => commands::bank::disposition(&client, &bank_id, verbose, output_format),
            BankCommands::Stats { bank_id } => commands::bank::stats(&client, &bank_id, verbose, output_format),
            BankCommands::Name { bank_id, name } => commands::bank::update_name(&client, &bank_id, &name, verbose, output_format),
            BankCommands::Background { bank_id, content, no_update_disposition } => {
                commands::bank::update_background(&client, &bank_id, &content, no_update_disposition, verbose, output_format)
            }
            BankCommands::Delete { bank_id, yes } => {
                commands::bank::delete(&client, &bank_id, yes, verbose, output_format)
            }
        },

        Commands::Memory(memory_cmd) => match memory_cmd {
            MemoryCommands::Recall { bank_id, query, fact_type, budget, max_tokens, trace, include_chunks, chunk_max_tokens } => {
                commands::memory::recall(&client, &bank_id, query, fact_type, budget, max_tokens, trace, include_chunks, chunk_max_tokens, verbose, output_format)
            }
            MemoryCommands::Reflect { bank_id, query, budget, context, max_tokens, schema } => {
                commands::memory::reflect(&client, &bank_id, query, budget, context, max_tokens, schema, verbose, output_format)
            }
            MemoryCommands::Retain { bank_id, content, doc_id, context, r#async } => {
                commands::memory::retain(&client, &bank_id, content, doc_id, context, r#async, verbose, output_format)
            }
            MemoryCommands::RetainFiles { bank_id, path, recursive, context, r#async } => {
                commands::memory::retain_files(&client, &bank_id, path, recursive, context, r#async, verbose, output_format)
            }
            MemoryCommands::Delete { bank_id, unit_id } => {
                commands::memory::delete(&client, &bank_id, &unit_id, verbose, output_format)
            }
            MemoryCommands::Clear { bank_id, fact_type, yes } => {
                commands::memory::clear(&client, &bank_id, fact_type, yes, verbose, output_format)
            }
        },

        Commands::Document(doc_cmd) => match doc_cmd {
            DocumentCommands::List { bank_id, query, limit, offset } => {
                commands::document::list(&client, &bank_id, query, limit, offset, verbose, output_format)
            }
            DocumentCommands::Get { bank_id, document_id } => {
                commands::document::get(&client, &bank_id, &document_id, verbose, output_format)
            }
            DocumentCommands::Delete { bank_id, document_id } => {
                commands::document::delete(&client, &bank_id, &document_id, verbose, output_format)
            }
        },

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

        Commands::Operation(op_cmd) => match op_cmd {
            OperationCommands::List { bank_id } => {
                commands::operation::list(&client, &bank_id, verbose, output_format)
            }
            OperationCommands::Cancel { bank_id, operation_id } => {
                commands::operation::cancel(&client, &bank_id, &operation_id, verbose, output_format)
            }
        },
    };

    // Handle API errors with nice messages
    if let Err(e) = result {
        errors::handle_api_error(e, &api_url);
    }

    Ok(())
}

fn handle_configure(api_url: Option<String>, api_key: Option<String>, output_format: OutputFormat) -> Result<()> {
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
                    format!("{}...{}", &key[..4], &key[key.len()-4..])
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
                format!("{}...{}", &key[..4], &key[key.len()-4..])
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

fn handle_ui(output_format: OutputFormat) -> Result<()> {
    use std::process::Command;

    // Load configuration to get the API URL
    let config = Config::load().unwrap_or_else(|e| {
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
