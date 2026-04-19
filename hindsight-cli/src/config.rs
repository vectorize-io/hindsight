use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const DEFAULT_API_URL: &str = "http://localhost:8888";
const CONFIG_FILE_NAME: &str = "config";
const CONFIG_DIR_NAME: &str = ".hindsight";
const PROFILE_DIR_NAME: &str = "cli-profiles";
const PROFILE_ENV_VAR: &str = "HINDSIGHT_PROFILE";

#[derive(Debug)]
pub struct Config {
    pub api_url: String,
    pub api_key: Option<String>,
    pub source: ConfigSource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigSource {
    LocalFile,
    Profile(String),
    Environment,
    Default,
}

impl std::fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigSource::LocalFile => write!(f, "config file"),
            ConfigSource::Profile(name) => write!(f, "profile '{}'", name),
            ConfigSource::Environment => write!(f, "environment variable"),
            ConfigSource::Default => write!(f, "default"),
        }
    }
}

impl Config {
    /// Load configuration with no explicit profile. See [`Self::load_with_profile`].
    pub fn load() -> Result<Self> {
        Self::load_with_profile(None)
    }

    /// Load configuration with the following priority:
    /// 1. Environment variable (HINDSIGHT_API_URL/HINDSIGHT_API_KEY) - highest priority
    /// 2. Named profile (from `profile_name` arg, else `$HINDSIGHT_PROFILE`)
    ///    at `~/.hindsight/cli-profiles/<name>.toml`
    /// 3. Local config file (`~/.hindsight/config`)
    /// 4. Default (http://localhost:8888)
    pub fn load_with_profile(profile_name: Option<&str>) -> Result<Self> {
        let env_api_key = env::var("HINDSIGHT_API_KEY").ok();

        // 1. Environment variable takes highest priority
        if let Ok(api_url) = env::var("HINDSIGHT_API_URL") {
            return Self::validate_and_create(api_url, env_api_key, ConfigSource::Environment);
        }

        // 2. Named profile (explicit flag takes precedence over env var)
        let resolved_profile: Option<String> = profile_name
            .map(|s| s.to_string())
            .or_else(|| env::var(PROFILE_ENV_VAR).ok().filter(|s| !s.is_empty()));

        if let Some(name) = resolved_profile {
            let (api_url, file_api_key) = Self::load_profile(&name)?;
            let api_key = env_api_key.or(file_api_key);
            return Self::validate_and_create(api_url, api_key, ConfigSource::Profile(name));
        }

        // 3. Local config file
        if let Some((api_url, file_api_key)) = Self::load_from_file()? {
            let api_key = env_api_key.or(file_api_key);
            return Self::validate_and_create(api_url, api_key, ConfigSource::LocalFile);
        }

        // 4. Fall back to default
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

    // ---------- profile support ----------

    pub fn profile_dir() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join(PROFILE_DIR_NAME))
    }

    pub fn profile_file_path(name: &str) -> Option<PathBuf> {
        Self::profile_dir().map(|dir| dir.join(format!("{}.toml", name)))
    }

    /// Load a named profile. Returns (api_url, api_key) or an error if the profile
    /// file is missing / malformed.
    pub fn load_profile(name: &str) -> Result<(String, Option<String>)> {
        validate_profile_name(name)?;
        let dir = Self::profile_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        load_profile_from_dir(&dir, name)
    }

    /// Save a named profile to `~/.hindsight/cli-profiles/<name>.toml`.
    /// Sets file permissions to 0600 on Unix to protect the API key.
    pub fn save_profile(name: &str, api_url: &str, api_key: Option<&str>) -> Result<PathBuf> {
        validate_profile_name(name)?;
        let dir = Self::profile_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        save_profile_to_dir(&dir, name, api_url, api_key)
    }

    /// List profile names (without `.toml` extension), sorted alphabetically.
    pub fn list_profiles() -> Result<Vec<String>> {
        let dir = match Self::profile_dir() {
            Some(d) => d,
            None => return Ok(vec![]),
        };
        list_profiles_in_dir(&dir)
    }

    /// Delete a named profile. Returns Ok(path) on success. Errors if the profile
    /// does not exist.
    pub fn delete_profile(name: &str) -> Result<PathBuf> {
        validate_profile_name(name)?;
        let path = Self::profile_file_path(name)
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        if !path.exists() {
            anyhow::bail!("profile '{}' not found at {}", name, path.display());
        }
        fs::remove_file(&path)
            .with_context(|| format!("Failed to delete profile file: {}", path.display()))?;
        Ok(path)
    }
}

fn load_profile_from_dir(dir: &std::path::Path, name: &str) -> Result<(String, Option<String>)> {
    let path = dir.join(format!("{}.toml", name));
    if !path.exists() {
        anyhow::bail!(
            "profile '{}' not found at {}; create with: hindsight profile create {} --api-url <url>",
            name,
            path.display(),
            name
        );
    }

    let content = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read profile file: {}", path.display()))?;

    let mut api_url: Option<String> = None;
    let mut api_key: Option<String> = None;

    for line in content.lines() {
        if let Some(v) = parse_config_value(line, "api_url") {
            api_url = Some(v);
        } else if let Some(v) = parse_config_value(line, "api_key") {
            api_key = Some(v);
        }
    }

    let api_url = api_url.ok_or_else(|| {
        anyhow::anyhow!(
            "profile '{}' at {} is missing required 'api_url' field",
            name,
            path.display()
        )
    })?;

    Ok((api_url, api_key))
}

fn save_profile_to_dir(
    dir: &std::path::Path,
    name: &str,
    api_url: &str,
    api_key: Option<&str>,
) -> Result<PathBuf> {
    if !api_url.starts_with("http://") && !api_url.starts_with("https://") {
        anyhow::bail!(
            "Invalid API URL: {}. Must start with http:// or https://",
            api_url
        );
    }

    if !dir.exists() {
        fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create profile directory: {}", dir.display()))?;
    }

    let path = dir.join(format!("{}.toml", name));
    let mut content = format!("api_url = \"{}\"\n", api_url);
    if let Some(key) = api_key {
        content.push_str(&format!("api_key = \"{}\"\n", key));
    }

    fs::write(&path, content)
        .with_context(|| format!("Failed to write profile file: {}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o600);
        fs::set_permissions(&path, perms).with_context(|| {
            format!("Failed to set permissions on profile file: {}", path.display())
        })?;
    }

    Ok(path)
}

fn list_profiles_in_dir(dir: &std::path::Path) -> Result<Vec<String>> {
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut names: Vec<String> = fs::read_dir(dir)
        .with_context(|| format!("Failed to read profile directory: {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("toml") {
                return None;
            }
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
        .collect();
    names.sort();
    Ok(names)
}

/// Reject empty, path-like, or hidden profile names so they can't escape the
/// profile directory.
fn validate_profile_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("profile name cannot be empty");
    }
    if name.starts_with('.')
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.contains(char::is_whitespace)
    {
        anyhow::bail!(
            "invalid profile name '{}': must not contain path separators, whitespace, or start with '.'",
            name
        );
    }
    Ok(())
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

/// Parse a simple TOML-like config line and extract value.
/// Handles both quoted and unquoted values.
pub fn parse_config_value(line: &str, key: &str) -> Option<String> {
    let line = line.trim();
    if !line.starts_with(key) {
        return None;
    }
    line.split('=').nth(1).map(|value| {
        value.trim().trim_matches('"').trim_matches('\'').to_string()
    }).filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_source_display() {
        assert_eq!(format!("{}", ConfigSource::LocalFile), "config file");
        assert_eq!(format!("{}", ConfigSource::Environment), "environment variable");
        assert_eq!(format!("{}", ConfigSource::Default), "default");
        assert_eq!(
            format!("{}", ConfigSource::Profile("prod".to_string())),
            "profile 'prod'"
        );
    }

    fn tempdir() -> PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = std::sync::atomic::AtomicU64::new(0);
        let n = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("hindsight-cli-test-{}-{}-{}", pid, nanos, n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_validate_profile_name_ok() {
        assert!(validate_profile_name("prod").is_ok());
        assert!(validate_profile_name("staging-1").is_ok());
        assert!(validate_profile_name("openclaw-plugin").is_ok());
        assert!(validate_profile_name("a_b_c").is_ok());
    }

    #[test]
    fn test_validate_profile_name_rejects_unsafe() {
        assert!(validate_profile_name("").is_err());
        assert!(validate_profile_name(".hidden").is_err());
        assert!(validate_profile_name("a/b").is_err());
        assert!(validate_profile_name("a\\b").is_err());
        assert!(validate_profile_name("..").is_err());
        assert!(validate_profile_name("foo bar").is_err());
    }

    #[test]
    fn test_save_and_load_profile_roundtrip() {
        let dir = tempdir();
        let path = save_profile_to_dir(&dir, "prod", "https://api.example.com", Some("hsk_abc"))
            .unwrap();
        assert!(path.exists());

        let (url, key) = load_profile_from_dir(&dir, "prod").unwrap();
        assert_eq!(url, "https://api.example.com");
        assert_eq!(key.as_deref(), Some("hsk_abc"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn test_save_profile_rejects_invalid_url() {
        let dir = tempdir();
        let err = save_profile_to_dir(&dir, "foo", "localhost:8888", None).unwrap_err();
        assert!(err.to_string().contains("Invalid API URL"));
    }

    #[test]
    fn test_load_profile_missing_returns_helpful_error() {
        let dir = tempdir();
        let err = load_profile_from_dir(&dir, "nope").unwrap_err().to_string();
        assert!(err.contains("profile 'nope' not found"));
        assert!(err.contains("hindsight profile create nope"));
    }

    #[test]
    fn test_load_profile_missing_api_url_fails() {
        let dir = tempdir();
        let path = dir.join("broken.toml");
        std::fs::write(&path, "api_key = \"x\"\n").unwrap();
        let err = load_profile_from_dir(&dir, "broken").unwrap_err().to_string();
        assert!(err.contains("missing required 'api_url'"));
    }

    #[test]
    fn test_list_profiles_returns_sorted_names() {
        let dir = tempdir();
        save_profile_to_dir(&dir, "prod", "https://prod.example.com", None).unwrap();
        save_profile_to_dir(&dir, "dev", "https://dev.example.com", None).unwrap();
        save_profile_to_dir(&dir, "staging", "https://staging.example.com", None).unwrap();
        // Non-toml files should be ignored.
        std::fs::write(dir.join("README"), "hi").unwrap();

        let names = list_profiles_in_dir(&dir).unwrap();
        assert_eq!(names, vec!["dev", "prod", "staging"]);
    }

    #[test]
    fn test_list_profiles_missing_dir_is_empty() {
        let dir = tempdir().join("nonexistent");
        let names = list_profiles_in_dir(&dir).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn test_validate_and_create_valid_http() {
        let config = Config::validate_and_create(
            "http://localhost:8888".to_string(),
            None,
            ConfigSource::Default,
        );
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.api_url, "http://localhost:8888");
        assert_eq!(config.source, ConfigSource::Default);
    }

    #[test]
    fn test_validate_and_create_valid_https() {
        let config = Config::validate_and_create(
            "https://api.example.com".to_string(),
            Some("secret-key".to_string()),
            ConfigSource::Environment,
        );
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.api_url, "https://api.example.com");
        assert_eq!(config.api_key, Some("secret-key".to_string()));
        assert_eq!(config.source, ConfigSource::Environment);
    }

    #[test]
    fn test_validate_and_create_invalid_url() {
        let config = Config::validate_and_create(
            "localhost:8888".to_string(),
            None,
            ConfigSource::Default,
        );
        assert!(config.is_err());
        let err = config.unwrap_err().to_string();
        assert!(err.contains("Invalid API URL"));
        assert!(err.contains("Must start with http:// or https://"));
    }

    #[test]
    fn test_validate_and_create_ftp_url() {
        let config = Config::validate_and_create(
            "ftp://example.com".to_string(),
            None,
            ConfigSource::Default,
        );
        assert!(config.is_err());
    }

    #[test]
    fn test_generate_doc_id_format() {
        let doc_id = generate_doc_id();
        assert!(doc_id.starts_with("cli_put_"));
        // Should be cli_put_YYYYMMDD_HHMMSS format
        assert!(doc_id.len() > 20); // cli_put_ (8) + date (8) + _ (1) + time (6) = 23
    }

    #[test]
    fn test_generate_doc_id_uniqueness() {
        let id1 = generate_doc_id();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let id2 = generate_doc_id();
        // IDs generated at different times should be different
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_parse_config_value_quoted() {
        assert_eq!(
            parse_config_value(r#"api_url = "http://localhost:8888""#, "api_url"),
            Some("http://localhost:8888".to_string())
        );
    }

    #[test]
    fn test_parse_config_value_single_quoted() {
        assert_eq!(
            parse_config_value("api_url = 'http://localhost:8888'", "api_url"),
            Some("http://localhost:8888".to_string())
        );
    }

    #[test]
    fn test_parse_config_value_unquoted() {
        assert_eq!(
            parse_config_value("api_url = http://localhost:8888", "api_url"),
            Some("http://localhost:8888".to_string())
        );
    }

    #[test]
    fn test_parse_config_value_with_spaces() {
        assert_eq!(
            parse_config_value("  api_url  =  \"http://localhost:8888\"  ", "api_url"),
            Some("http://localhost:8888".to_string())
        );
    }

    #[test]
    fn test_parse_config_value_wrong_key() {
        assert_eq!(
            parse_config_value("api_key = secret", "api_url"),
            None
        );
    }

    #[test]
    fn test_parse_config_value_empty() {
        assert_eq!(
            parse_config_value("api_url = ", "api_url"),
            None
        );
        assert_eq!(
            parse_config_value("api_url = \"\"", "api_url"),
            None
        );
    }

    #[test]
    fn test_config_api_url_accessor() {
        let config = Config {
            api_url: "http://test:8080".to_string(),
            api_key: None,
            source: ConfigSource::Default,
        };
        assert_eq!(config.api_url(), "http://test:8080");
    }
}
