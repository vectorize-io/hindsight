//! Health and metrics commands.

use anyhow::Result;
use serde::Deserialize;

use crate::api::ApiClient;
use crate::output::{self, OutputFormat};
use crate::ui;

// Local type for health response
#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    database: Option<String>,
    version: Option<String>,
}

/// Check API health
pub fn health(client: &ApiClient, verbose: bool, output_format: OutputFormat) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Checking health..."))
    } else {
        None
    };

    let response = client.health(verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(value) => {
            if output_format == OutputFormat::Pretty {
                let result: HealthResponse =
                    serde_json::from_value(value.clone()).unwrap_or(HealthResponse {
                        status: "unknown".to_string(),
                        database: None,
                        version: None,
                    });

                let status_str = if result.status == "healthy" {
                    ui::gradient_start(&result.status)
                } else {
                    ui::gradient_end(&result.status)
                };

                ui::print_section_header("Health Check");
                println!("  {} {}", ui::dim("Status:"), status_str);

                if let Some(db_status) = &result.database {
                    let db_str = if db_status == "connected" {
                        ui::gradient_start(db_status)
                    } else {
                        ui::gradient_end(db_status)
                    };
                    println!("  {} {}", ui::dim("Database:"), db_str);
                }

                if let Some(version) = &result.version {
                    println!("  {} {}", ui::dim("Version:"), version);
                }

                println!();
            } else {
                output::print_output(&value, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Get API version information
pub fn version(client: &ApiClient, verbose: bool, output_format: OutputFormat) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching version..."))
    } else {
        None
    };

    let response = client.get_version(verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header("API Version");
                println!("  {} {}", ui::dim("Version:"), result.api_version);

                println!();
                println!("  {}", ui::dim("Features:"));
                println!(
                    "    {} MCP Server: {}",
                    ui::gradient_start("•"),
                    if result.features.mcp {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!(
                    "    {} Observations: {}",
                    ui::gradient_start("•"),
                    if result.features.observations {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!(
                    "    {} Background Worker: {}",
                    ui::gradient_start("•"),
                    if result.features.worker {
                        "enabled"
                    } else {
                        "disabled"
                    }
                );
                println!();
            } else {
                output::print_output(&result, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Get Prometheus metrics
pub fn metrics(client: &ApiClient, verbose: bool, output_format: OutputFormat) -> Result<()> {
    let spinner = if output_format == OutputFormat::Pretty {
        Some(ui::create_spinner("Fetching metrics..."))
    } else {
        None
    };

    let response = client.metrics(verbose);

    if let Some(mut sp) = spinner {
        sp.finish();
    }

    match response {
        Ok(result) => {
            if output_format == OutputFormat::Pretty {
                ui::print_section_header("Prometheus Metrics");
                println!("{}", result);
            } else {
                // For JSON/YAML, wrap in an object
                let wrapped = serde_json::json!({ "metrics": result });
                output::print_output(&wrapped, output_format)?;
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_deserialization() {
        let json = r#"{
            "status": "healthy",
            "database": "connected",
            "version": "0.3.0"
        }"#;

        let value: serde_json::Value = serde_json::from_str(json).unwrap();
        let result: HealthResponse = serde_json::from_value(value).unwrap();

        assert_eq!(result.status, "healthy");
        assert_eq!(result.database, Some("connected".to_string()));
        assert_eq!(result.version, Some("0.3.0".to_string()));
    }

    #[test]
    fn test_health_response_minimal() {
        let json = r#"{"status": "healthy"}"#;

        let value: serde_json::Value = serde_json::from_str(json).unwrap();
        let result: HealthResponse = serde_json::from_value(value).unwrap();

        assert_eq!(result.status, "healthy");
        assert_eq!(result.database, None);
        assert_eq!(result.version, None);
    }

    #[test]
    fn test_health_response_unhealthy() {
        let json = r#"{
            "status": "unhealthy",
            "database": "disconnected"
        }"#;

        let value: serde_json::Value = serde_json::from_str(json).unwrap();
        let result: HealthResponse = serde_json::from_value(value).unwrap();

        assert_eq!(result.status, "unhealthy");
        assert_eq!(result.database, Some("disconnected".to_string()));
    }
}
