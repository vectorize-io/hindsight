use anyhow::{Context, Result};
use crate::api::ApiClient;
use crate::config::Config;
use crate::output::OutputFormat;

/// Get API client from config
pub fn get_client(config: &Config) -> Result<ApiClient> {
    ApiClient::new(config.api_url.clone(), config.api_key.clone())
        .context("Failed to create API client")
}

/// Get output format, preferring CLI arg over default
pub fn get_output_format(cli_format: Option<OutputFormat>, _config: &Config) -> OutputFormat {
    cli_format.unwrap_or(OutputFormat::Pretty)
}
