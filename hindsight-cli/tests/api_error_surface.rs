//! Structural guard: every generated-client call must surface the HTTP
//! response body.
//!
//! A bare `self.client.foo(..).await?` converts progenitor's `Error` into
//! `anyhow::Error` through its `Display`, which for the common
//! `UnexpectedResponse` case is "Unexpected Response: Response { .. }" — the
//! body is never read, so the CLI cannot print what the server actually said
//! and falls back to generic guidance ("API endpoint not found (404)" for a
//! plain "Document not found"; see issue #4049). `.humanized()` reads the body
//! into the error instead.
//!
//! Nothing in the type system stops the next call site from writing the bare
//! form — it compiles and only misbehaves against a real error response, which
//! no unit test exercises. So the family is checked here, over the whole file,
//! rather than one call at a time.

/// Statement boundary for the crude scan below: `api.rs` writes one client call
/// per statement, so everything since the last `;`/`{`/`}` is the call
/// expression.
fn statement_before(source: &str, offset: usize) -> &str {
    let start = [';', '{', '}']
        .iter()
        .filter_map(|delimiter| source[..offset].rfind(*delimiter))
        .max()
        .unwrap_or(0);
    &source[start..offset]
}

#[test]
fn every_generated_client_call_surfaces_the_response_body() {
    let source = include_str!("../src/api.rs");

    let mut bare_calls = Vec::new();
    for (offset, _) in source.match_indices(".await?") {
        let statement = statement_before(source, offset);
        let compact: String = statement.chars().filter(|c| !c.is_whitespace()).collect();

        // `http_client` is the hand-rolled reqwest path (multipart upload,
        // template import); it reads the body itself and formats its own error.
        if !compact.contains("self.client") || compact.contains("http_client") {
            continue;
        }
        if !compact.contains(".humanized()") {
            let line = source[..offset].lines().count();
            let call = statement
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .trim_start_matches(['{', '}', ';'])
                .trim()
                .to_string();
            bare_calls.push(format!("  src/api.rs:{}: {}", line, call));
        }
    }

    assert!(
        bare_calls.is_empty(),
        "generated-client calls that swallow the server's response body — \
         end them with `.humanized().await?` instead of `.await?`:\n{}",
        bare_calls.join("\n")
    );
}

#[test]
fn the_guard_notices_a_bare_call() {
    // Guards the guard: a scan that silently matches nothing would pass the
    // test above forever.
    let bare = "let response = self\n.client\n.get_document(bank_id, document_id, None)\n.await?;";
    let statement = statement_before(bare, bare.find(".await?").unwrap());
    let compact: String = statement.chars().filter(|c| !c.is_whitespace()).collect();

    assert!(compact.contains("self.client"));
    assert!(!compact.contains(".humanized()"));
}
