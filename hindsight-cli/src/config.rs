use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const DEFAULT_API_URL: &str = "http://localhost:8888";
const CONFIG_FILE_NAME: &str = "config";
const CONFIG_DIR_NAME: &str = ".hindsight";

pub struct Config {
    pub api_url: String,
    pub api_key: Option<String>,
    pub source: ConfigSource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigSource {
    LocalFile,
    Environment,
    Default,
}

impl std::fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigSource::LocalFile => write!(f, "config file"),
            ConfigSource::Environment => write!(f, "environment variable"),
            ConfigSource::Default => write!(f, "default"),
        }
    }
}

impl Config {
    /// Load configuration with the following priority:
    /// 1. Environment variable (HINDSIGHT_API_URL, HINDSIGHT_API_KEY) - highest priority, for overrides
    /// 2. Local config file (~/.hindsight/config.toml)
    /// 3. Default (http://localhost:8888)
    pub fn load() -> Result<Self> {
        // Load API key from environment (highest priority)
        let env_api_key = env::var("HINDSIGHT_API_KEY").ok();

        // 1. Environment variable takes highest priority (for overrides)
        if let Ok(api_url) = env::var("HINDSIGHT_API_URL") {
            return Self::validate_and_create(api_url, env_api_key, ConfigSource::Environment);
        }

        // 2. Try local config file
        if let Some((api_url, file_api_key)) = Self::load_from_file()? {
            // Environment api_key takes precedence over file api_key
            let api_key = env_api_key.or(file_api_key);
            return Self::validate_and_create(api_url, api_key, ConfigSource::LocalFile);
        }

        // 3. Fall back to default
        Self::validate_and_create(DEFAULT_API_URL.to_string(), env_api_key, ConfigSource::Default)
    }

    /// Legacy method for backwards compatibility
    pub fn from_env() -> Result<Self> {
        Self::load()
    }

    fn validate_and_create(api_url: String, api_key: Option<String>, source: ConfigSource) -> Result<Self> {
        if !api_url.starts_with("http://") && !api_url.starts_with("https://") {
            anyhow::bail!(
                "Invalid API URL: {}. Must start with http:// or https://",
                api_url
            );
        }
        Ok(Config { api_url, api_key, source })
    }

    fn config_dir() -> Option<PathBuf> {
        dirs::home_dir().map(|home| home.join(CONFIG_DIR_NAME))
    }

    fn config_file_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join(CONFIG_FILE_NAME))
    }

    fn load_from_file() -> Result<Option<(String, Option<String>)>> {
        let config_path = match Self::config_file_path() {
            Some(path) => path,
            None => return Ok(None),
        };

        if !config_path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

        let mut api_url: Option<String> = None;
        let mut api_key: Option<String> = None;

        // Simple TOML parsing for api_url and api_key
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("api_url") {
                if let Some(value) = line.split('=').nth(1) {
                    let value = value.trim().trim_matches('"').trim_matches('\'');
                    if !value.is_empty() {
                        api_url = Some(value.to_string());
                    }
                }
            } else if line.starts_with("api_key") {
                if let Some(value) = line.split('=').nth(1) {
                    let value = value.trim().trim_matches('"').trim_matches('\'');
                    if !value.is_empty() {
                        api_key = Some(value.to_string());
                    }
                }
            }
        }

        match api_url {
            Some(url) => Ok(Some((url, api_key))),
            None => Ok(None),
        }
    }

    pub fn save_api_url(api_url: &str) -> Result<PathBuf> {
        Self::save_config(api_url, None)
    }

    pub fn save_config(api_url: &str, api_key: Option<&str>) -> Result<PathBuf> {
        let config_dir = Self::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;

        // Create config directory if it doesn't exist
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir)
                .with_context(|| format!("Failed to create config directory: {}", config_dir.display()))?;
        }

        let config_path = config_dir.join(CONFIG_FILE_NAME);
        let mut content = format!("api_url = \"{}\"\n", api_url);
        if let Some(key) = api_key {
            content.push_str(&format!("api_key = \"{}\"\n", key));
        }

        fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;

        Ok(config_path)
    }

    pub fn api_url(&self) -> &str {
        &self.api_url
    }
}

/// Prompt user for API URL interactively
pub fn prompt_api_url(current_url: Option<&str>) -> Result<String> {
    let default = current_url.unwrap_or(DEFAULT_API_URL);

    print!("Enter API URL [{}]: ", default);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let input = input.trim();
    if input.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(input.to_string())
    }
}

pub fn generate_doc_id() -> String {
    let now = chrono::Local::now();
    format!("cli_put_{}", now.format("%Y%m%d_%H%M%S"))
}
