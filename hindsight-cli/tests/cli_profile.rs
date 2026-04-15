//! End-to-end tests for the `hindsight profile` CRUD subcommands and the
//! global `-p/--profile` flag.
//!
//! These tests do not require a running Hindsight API server: they exercise
//! the binary against a temporary HOME directory and assert on the profile
//! files it reads/writes. The only time a command is expected to contact the
//! API is the `-p` precedence test, which uses `hindsight version` pointed at
//! a deliberately-unreachable URL so we can assert that the profile value
//! (not the default `http://localhost:8888`) was picked up from the error
//! message.
//!
//! These integration tests are Unix-only because they rely on overriding
//! `$HOME` to redirect `dirs::home_dir()` at a tempdir. On Windows
//! `dirs::home_dir()` resolves via `FOLDERID_Profile` (the Win32 shell API)
//! and ignores the env var, so running these tests there would pollute the
//! real user profile directory. The Windows runtime path is still exercised
//! by the `config::tests::*` unit tests, which drive the path-based
//! `save_profile_to_dir` / `load_profile_from_dir` helpers directly.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_tempdir(tag: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("hindsight-profile-test-{}-{}-{}-{}", tag, pid, nanos, n));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn hindsight_binary() -> String {
    std::env::var("CARGO_BIN_EXE_hindsight").unwrap_or_else(|_| {
        let debug = "./target/debug/hindsight";
        let release = "./target/release/hindsight";
        if std::path::Path::new(debug).exists() {
            debug.to_string()
        } else if std::path::Path::new(release).exists() {
            release.to_string()
        } else {
            "hindsight".to_string()
        }
    })
}

fn run_with_home(home: &std::path::Path, args: &[&str]) -> Output {
    Command::new(hindsight_binary())
        .env("HOME", home)
        // Unset anything that would bypass the profile/config-file resolution
        // we're trying to exercise here.
        .env_remove("HINDSIGHT_API_URL")
        .env_remove("HINDSIGHT_API_KEY")
        .env_remove("HINDSIGHT_PROFILE")
        .args(args)
        .output()
        .expect("failed to spawn hindsight binary")
}

fn assert_success(out: &Output) {
    if !out.status.success() {
        panic!(
            "command failed: status={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn profile_create_writes_toml_and_json_output() {
    let home = unique_tempdir("create");
    let out = run_with_home(
        &home,
        &[
            "--output",
            "json",
            "profile",
            "create",
            "prod",
            "--api-url",
            "https://api.example.com",
            "--api-key",
            "hsk_abcdef1234",
        ],
    );
    assert_success(&out);

    let payload: serde_json::Value =
        serde_json::from_str(&stdout(&out)).expect("expected JSON output");
    assert_eq!(payload["name"], "prod");
    assert_eq!(payload["api_url"], "https://api.example.com");
    assert_eq!(payload["api_key_set"], true);

    let path = home.join(".hindsight/cli-profiles/prod.toml");
    assert!(path.exists(), "profile file was not created at {}", path.display());
    let body = std::fs::read_to_string(&path).unwrap();
    assert!(body.contains("api_url = \"https://api.example.com\""));
    assert!(body.contains("api_key = \"hsk_abcdef1234\""));

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "profile file should be mode 0600");
    }
}

#[test]
fn profile_create_rejects_invalid_api_url() {
    let home = unique_tempdir("invalid-url");
    let out = run_with_home(
        &home,
        &["profile", "create", "foo", "--api-url", "localhost:8888"],
    );
    assert!(!out.status.success());
    assert!(stderr(&out).contains("Invalid API URL"));
}

#[test]
fn profile_create_rejects_unsafe_names() {
    let home = unique_tempdir("unsafe-name");
    for bad in &["..", ".hidden", "a/b", "a b"] {
        let out = run_with_home(
            &home,
            &["profile", "create", bad, "--api-url", "https://example.com"],
        );
        assert!(
            !out.status.success(),
            "expected failure for profile name {:?}",
            bad
        );
    }
}

#[test]
fn profile_list_returns_sorted_names() {
    let home = unique_tempdir("list");
    for name in &["prod", "dev", "staging"] {
        let out = run_with_home(
            &home,
            &[
                "profile",
                "create",
                name,
                "--api-url",
                &format!("https://{}.example.com", name),
            ],
        );
        assert_success(&out);
    }

    let out = run_with_home(&home, &["--output", "json", "profile", "list"]);
    assert_success(&out);
    let payload: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    let names: Vec<&str> = payload["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["dev", "prod", "staging"]);
}

#[test]
fn profile_list_empty_when_no_profiles() {
    let home = unique_tempdir("list-empty");
    let out = run_with_home(&home, &["--output", "json", "profile", "list"]);
    assert_success(&out);
    let payload: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    assert!(payload["profiles"].as_array().unwrap().is_empty());
}

#[test]
fn profile_show_returns_stored_values() {
    let home = unique_tempdir("show");
    assert_success(&run_with_home(
        &home,
        &[
            "profile",
            "create",
            "prod",
            "--api-url",
            "https://api.example.com",
            "--api-key",
            "hsk_xyz",
        ],
    ));

    let out = run_with_home(&home, &["--output", "json", "profile", "show", "prod"]);
    assert_success(&out);
    let payload: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(payload["name"], "prod");
    assert_eq!(payload["api_url"], "https://api.example.com");
    assert_eq!(payload["api_key_set"], true);
}

#[test]
fn profile_show_missing_profile_errors_with_hint() {
    let home = unique_tempdir("show-missing");
    let out = run_with_home(&home, &["profile", "show", "nope"]);
    assert!(!out.status.success());
    let err = stderr(&out) + &stdout(&out);
    assert!(err.contains("profile 'nope' not found"));
    assert!(err.contains("hindsight profile create nope"));
}

#[test]
fn profile_delete_removes_file() {
    let home = unique_tempdir("delete");
    assert_success(&run_with_home(
        &home,
        &[
            "profile",
            "create",
            "prod",
            "--api-url",
            "https://api.example.com",
        ],
    ));
    let path = home.join(".hindsight/cli-profiles/prod.toml");
    assert!(path.exists());

    let out = run_with_home(&home, &["profile", "delete", "prod", "-y"]);
    assert_success(&out);
    assert!(!path.exists(), "profile file should have been removed");
}

#[test]
fn profile_delete_missing_errors() {
    let home = unique_tempdir("delete-missing");
    let out = run_with_home(&home, &["profile", "delete", "nope", "-y"]);
    assert!(!out.status.success());
}

#[test]
fn p_flag_overrides_config_file_api_url() {
    // Build a HOME that contains BOTH a legacy ~/.hindsight/config pointing
    // at URL_A and a named profile pointing at URL_B. Running `hindsight -p`
    // should pick the profile URL, not the config-file URL — we verify by
    // looking at the connection-error message (no API server needed).
    let home = unique_tempdir("precedence");
    let hindsight_dir = home.join(".hindsight");
    std::fs::create_dir_all(&hindsight_dir).unwrap();
    std::fs::write(
        hindsight_dir.join("config"),
        "api_url = \"http://127.0.0.1:9/from-config\"\n",
    )
    .unwrap();

    assert_success(&run_with_home(
        &home,
        &[
            "profile",
            "create",
            "prod",
            "--api-url",
            "http://127.0.0.1:9/from-profile",
        ],
    ));

    // `version` will fail to connect (port 9 is "discard"), but the error
    // message echoes the API URL actually used.
    let out = run_with_home(&home, &["-p", "prod", "version"]);
    assert!(!out.status.success());
    let err = stderr(&out) + &stdout(&out);
    assert!(
        err.contains("from-profile"),
        "expected profile URL in output, got:\n{}",
        err
    );
    assert!(
        !err.contains("from-config"),
        "config-file URL should not have been used:\n{}",
        err
    );
}

#[test]
fn hindsight_profile_env_var_is_honored() {
    let home = unique_tempdir("env-var");
    assert_success(&run_with_home(
        &home,
        &[
            "profile",
            "create",
            "staging",
            "--api-url",
            "http://127.0.0.1:9/from-env",
        ],
    ));

    // Re-run without `-p` but with HINDSIGHT_PROFILE set.
    let out = Command::new(hindsight_binary())
        .env("HOME", &home)
        .env_remove("HINDSIGHT_API_URL")
        .env_remove("HINDSIGHT_API_KEY")
        .env("HINDSIGHT_PROFILE", "staging")
        .args(["version"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr).into_owned()
        + &String::from_utf8_lossy(&out.stdout);
    assert!(err.contains("from-env"), "expected profile URL in output:\n{}", err);
}
