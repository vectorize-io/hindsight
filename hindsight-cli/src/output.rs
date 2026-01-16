use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Pretty,
    Json,
    Yaml,
}

impl OutputFormat {
    /// Parse output format from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(OutputFormat::Json),
            "yaml" | "yml" => Some(OutputFormat::Yaml),
            "pretty" | "text" => Some(OutputFormat::Pretty),
            _ => None,
        }
    }
}

/// Format data as JSON string
pub fn to_json<T: Serialize>(data: &T) -> Result<String> {
    Ok(serde_json::to_string_pretty(data)?)
}

/// Format data as YAML string
pub fn to_yaml<T: Serialize>(data: &T) -> Result<String> {
    Ok(serde_yaml::to_string(data)?)
}

pub fn print_output<T: Serialize>(data: &T, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            println!("{}", to_json(data)?);
        }
        OutputFormat::Yaml => {
            println!("{}", to_yaml(data)?);
        }
        OutputFormat::Pretty => {
            // This should not be called - pretty printing is handled in ui.rs
            unreachable!("Pretty format should be handled separately")
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestData {
        name: String,
        count: i32,
        active: bool,
    }

    #[test]
    fn test_output_format_from_str_json() {
        assert_eq!(OutputFormat::from_str("json"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("JSON"), Some(OutputFormat::Json));
        assert_eq!(OutputFormat::from_str("Json"), Some(OutputFormat::Json));
    }

    #[test]
    fn test_output_format_from_str_yaml() {
        assert_eq!(OutputFormat::from_str("yaml"), Some(OutputFormat::Yaml));
        assert_eq!(OutputFormat::from_str("YAML"), Some(OutputFormat::Yaml));
        assert_eq!(OutputFormat::from_str("yml"), Some(OutputFormat::Yaml));
        assert_eq!(OutputFormat::from_str("YML"), Some(OutputFormat::Yaml));
    }

    #[test]
    fn test_output_format_from_str_pretty() {
        assert_eq!(OutputFormat::from_str("pretty"), Some(OutputFormat::Pretty));
        assert_eq!(OutputFormat::from_str("PRETTY"), Some(OutputFormat::Pretty));
        assert_eq!(OutputFormat::from_str("text"), Some(OutputFormat::Pretty));
    }

    #[test]
    fn test_output_format_from_str_invalid() {
        assert_eq!(OutputFormat::from_str("xml"), None);
        assert_eq!(OutputFormat::from_str("csv"), None);
        assert_eq!(OutputFormat::from_str(""), None);
    }

    #[test]
    fn test_to_json() {
        let data = TestData {
            name: "test".to_string(),
            count: 42,
            active: true,
        };
        let json = to_json(&data).unwrap();
        assert!(json.contains("\"name\": \"test\""));
        assert!(json.contains("\"count\": 42"));
        assert!(json.contains("\"active\": true"));
    }

    #[test]
    fn test_to_yaml() {
        let data = TestData {
            name: "test".to_string(),
            count: 42,
            active: true,
        };
        let yaml = to_yaml(&data).unwrap();
        assert!(yaml.contains("name: test"));
        assert!(yaml.contains("count: 42"));
        assert!(yaml.contains("active: true"));
    }

    #[test]
    fn test_to_json_array() {
        let data = vec![
            TestData { name: "a".to_string(), count: 1, active: true },
            TestData { name: "b".to_string(), count: 2, active: false },
        ];
        let json = to_json(&data).unwrap();
        assert!(json.contains("\"name\": \"a\""));
        assert!(json.contains("\"name\": \"b\""));
    }

    #[test]
    fn test_to_yaml_array() {
        let data = vec![
            TestData { name: "a".to_string(), count: 1, active: true },
            TestData { name: "b".to_string(), count: 2, active: false },
        ];
        let yaml = to_yaml(&data).unwrap();
        assert!(yaml.contains("name: a"));
        assert!(yaml.contains("name: b"));
    }

    #[test]
    fn test_output_format_equality() {
        assert_eq!(OutputFormat::Json, OutputFormat::Json);
        assert_ne!(OutputFormat::Json, OutputFormat::Yaml);
        assert_ne!(OutputFormat::Yaml, OutputFormat::Pretty);
    }

    #[test]
    fn test_output_format_clone() {
        let format = OutputFormat::Json;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_to_json_special_chars() {
        let data = TestData {
            name: "test\"with\\special\nchars".to_string(),
            count: 0,
            active: false,
        };
        let json = to_json(&data).unwrap();
        // JSON should properly escape special characters
        assert!(json.contains("\\\""));
        assert!(json.contains("\\\\"));
        assert!(json.contains("\\n"));
    }
}
