//! Integration tests for the hindsight-agent CLI.
//!
//! Tests that exercise the binary via subprocess, verifying command parsing,
//! config file handling, and error messages.

use std::process::Command;

fn agent_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hindsight-agent"))
}

// ── Help & version ──────────────────────────────────────

#[test]
fn test_help() {
    let output = agent_bin().arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("Agent CLI for Hindsight Wiki"));
    assert!(stdout.contains("setup"));
    assert!(stdout.contains("wiki"));
    assert!(stdout.contains("recall"));
    assert!(stdout.contains("ingest"));
    assert!(stdout.contains("documents"));
    assert!(stdout.contains("agents"));
    assert!(stdout.contains("retain"));
}

#[test]
fn test_version() {
    let output = agent_bin().arg("--version").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("hindsight-agent"));
}

// ── Wiki subcommand help ────────────────────────────────

#[test]
fn test_wiki_help() {
    let output = agent_bin().args(["wiki", "--help"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("list"));
    assert!(stdout.contains("get"));
    assert!(stdout.contains("create"));
    assert!(stdout.contains("update"));
    assert!(stdout.contains("delete"));
}

// ── Agents subcommand ───────────────────────────────────

#[test]
fn test_agents_help() {
    let output = agent_bin().args(["agents", "--help"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("list"));
    assert!(stdout.contains("show"));
}

#[test]
fn test_agents_list_reads_config() {
    // This reads the real ~/.hindsight-agent/config.json
    // Should succeed even if empty
    let output = agent_bin().args(["agents", "list"]).output().unwrap();
    assert!(output.status.success());
}

// ── Error handling ──────────────────────────────────────

#[test]
fn test_wiki_list_unknown_agent() {
    let output = agent_bin()
        .args(["wiki", "list", "nonexistent-agent-xyz-123"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found"));
}

#[test]
fn test_recall_unknown_agent() {
    let output = agent_bin()
        .args(["recall", "nonexistent-agent-xyz-123", "test query"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found"));
}

#[test]
fn test_ingest_unknown_agent() {
    let output = agent_bin()
        .args(["ingest", "nonexistent-agent-xyz-123", "title", "-c", "content"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found"));
}

#[test]
fn test_documents_unknown_agent() {
    let output = agent_bin()
        .args(["documents", "nonexistent-agent-xyz-123"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found"));
}

// ── Setup validation ────────────────────────────────────

#[test]
fn test_setup_requires_bank_id() {
    let output = agent_bin()
        .args(["setup", "test-agent", "--harness", "hermes"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--bank-id"));
}

#[test]
fn test_setup_requires_harness() {
    let output = agent_bin()
        .args(["setup", "test-agent", "--bank-id", "test"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--harness"));
}

#[test]
fn test_setup_rejects_invalid_harness() {
    let output = agent_bin()
        .args([
            "setup",
            "test-agent",
            "--bank-id",
            "test",
            "--harness",
            "invalid",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

// ── Wiki create validation ──────────────────────────────

#[test]
fn test_wiki_create_requires_all_args() {
    // Missing source_query
    let output = agent_bin()
        .args(["wiki", "create", "agent", "page-id", "Name"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn test_wiki_update_requires_flag() {
    let output = agent_bin()
        .args(["wiki", "update", "nonexistent-xyz", "page-id"])
        .output()
        .unwrap();
    // Should fail because no --name or --source-query provided
    // (may fail on agent lookup first)
    assert!(!output.status.success());
}
