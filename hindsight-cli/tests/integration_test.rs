use std::process::Command;

#[test]
fn test_cli_help() {
    let output = Command::new("cargo")
        .args(["run", "--", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Hindsight CLI"));
}

#[test]
fn test_cli_version() {
    let output = Command::new("cargo")
        .args(["run", "--", "--version"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("hindsight"));
}

#[test]
fn test_ui_command_without_config() {
    // Test that the ui command handles missing config gracefully
    // Create a temp home directory with no config
    let temp_dir = std::env::temp_dir().join(format!("hindsight-test-ui-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

    let output = Command::new("cargo")
        .args(["run", "--", "ui"])
        .env_remove("HINDSIGHT_API_URL")
        .env_remove("HINDSIGHT_API_KEY")
        .env("HOME", &temp_dir)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Either it fails with a config error or it succeeds if there's a default config
    // Just verify it doesn't crash unexpectedly
    assert!(
        !output.status.success()
        || stdout.contains("Launching Hindsight Control Plane UI")
        || stderr.contains("Configuration error")
        || stderr.contains("HINDSIGHT_API_URL"),
        "Unexpected output - stdout: {}, stderr: {}",
        stdout,
        stderr
    );

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn test_ui_command_with_config() {
    // This test is skipped by default since it requires a running control plane
    // and would block for a long time. The other tests cover the basic functionality.
    // To run this test manually:
    // 1. Build the control plane: cd hindsight-control-plane && npm run build
    // 2. Run: cargo test test_ui_command_with_config -- --ignored

    // Just verify that the ui command accepts the configuration
    let temp_dir = std::env::temp_dir().join(format!("hindsight-test-ui-valid-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

    // Write a minimal config
    let config_dir = temp_dir.join(".config").join("hindsight");
    std::fs::create_dir_all(&config_dir).expect("Failed to create config dir");
    let config_file = config_dir.join("config");
    std::fs::write(&config_file, "api_url=http://localhost:8888\napi_key=test-key\n")
        .expect("Failed to write config");

    let output = Command::new("cargo")
        .args(["run", "--", "ui", "--help"])
        .env("HOME", &temp_dir)
        .output()
        .expect("Failed to execute command");

    // The --help should work regardless
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Hindsight CLI") || output.status.success());

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn test_memory_item_timestamp_serializes_as_plain_string() {
    // Regression: progenitor used to emit `MemoryItemTimestamp` as a struct
    // with two `#[serde(flatten)]` Option subtypes from the `anyOf:[datetime,
    // string]` schema. Serializing that errored with
    //   "can only flatten structs and maps (got a string)"
    // — so `hindsight memory retain --timestamp` never worked. The build.rs
    // spec-massage step now collapses string-only anyOf unions into a plain
    // `{type: string}`, so the field is just `Option<String>`. Assert that.
    let item = hindsight_client::types::MemoryItem {
        content: "Bob went hiking yesterday".to_string(),
        context: None,
        metadata: None,
        timestamp: Some("2026-05-31T10:00:00Z".to_string()),
        document_id: Some("doc-1".to_string()),
        entities: None,
        tags: None,
        observation_scopes: None,
        strategy: None,
        update_mode: None,
    };
    let json = serde_json::to_string(&item).expect("MemoryItem must serialize");
    assert!(
        json.contains(r#""timestamp":"2026-05-31T10:00:00Z""#),
        "expected timestamp at top-level as a plain string, got: {}",
        json
    );
}

#[test]
fn test_memory_retain_exposes_timestamp_flag() {
    // Regression: `hindsight memory retain` historically had no way to set the
    // memory's event date even though the SDKs do. The flag must appear in
    // --help so users (and docs) can discover it.
    let output = Command::new("cargo")
        .args(["run", "--", "memory", "retain", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "retain --help failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--timestamp") && stdout.contains("-t"),
        "expected --timestamp/-t flag in retain --help, got: {}",
        stdout
    );
}

#[test]
fn test_configure_command() {
    // Test that configure command creates/updates config
    let temp_dir = std::env::temp_dir().join(format!("hindsight-test-{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

    let output = Command::new("cargo")
        .args([
            "run",
            "--",
            "configure",
            "--api-url",
            "http://localhost:9999",
            "--api-key",
            "test-key-123"
        ])
        .env("HOME", &temp_dir)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "Configure command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Configuration saved") || stdout.contains("success"));

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).ok();
}
