use std::env;
use std::fs;
use std::path::PathBuf;

/// Convert OpenAPI 3.1 spec to 3.0 for progenitor compatibility
fn convert_31_to_30(spec: &mut serde_json::Value) {
    // Change version from 3.1.x to 3.0.3
    if let Some(obj) = spec.as_object_mut() {
        obj.insert("openapi".to_string(), serde_json::json!("3.0.3"));
    }

    // Recursively convert anyOf with null to nullable
    convert_anyof_to_nullable(spec);
}

/// Remove paths with multipart/form-data content type (not supported by progenitor)
fn filter_multipart_endpoints(spec: &mut serde_json::Value) {
    if let Some(paths) = spec.get_mut("paths").and_then(|v| v.as_object_mut()) {
        let mut paths_to_remove = Vec::new();

        for (path_name, path_item) in paths.iter() {
            if let Some(operations) = path_item.as_object() {
                for (_method, operation) in operations.iter() {
                    if let Some(request_body) = operation.get("requestBody") {
                        if let Some(content) = request_body.get("content") {
                            if let Some(content_obj) = content.as_object() {
                                if content_obj.contains_key("multipart/form-data") {
                                    eprintln!(
                                        "Filtering out endpoint with multipart/form-data: {}",
                                        path_name
                                    );
                                    paths_to_remove.push(path_name.clone());
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Remove the paths
        for path in paths_to_remove {
            paths.remove(&path);
        }
    }
}

/// Binary response bodies with typed headers currently trigger a progenitor
/// multi-response assertion. They are exposed by a small maintained helper.
fn filter_binary_response_operations(spec: &mut serde_json::Value) {
    let Some(paths) = spec
        .get_mut("paths")
        .and_then(|value| value.as_object_mut())
    else {
        return;
    };
    for (path_name, path_item) in paths.iter_mut() {
        let Some(operations) = path_item.as_object_mut() else {
            continue;
        };
        let methods: Vec<String> = operations
            .iter()
            .filter_map(|(method, operation)| {
                let has_binary = operation
                    .get("responses")
                    .and_then(|value| value.as_object())
                    .into_iter()
                    .flat_map(|responses| responses.values())
                    .filter_map(|response| response.get("content")?.as_object())
                    .flat_map(|content| content.values())
                    .any(|media| {
                        media
                            .get("schema")
                            .and_then(|schema| schema.get("format"))
                            .and_then(|value| value.as_str())
                            == Some("binary")
                    });
                has_binary.then(|| method.clone())
            })
            .collect();
        for method in methods {
            eprintln!("Filtering binary endpoint for maintained helper: {method} {path_name}");
            operations.remove(&method);
        }
    }
}

/// Collapse `anyOf` whose members are all plain string types into a single
/// `{"type": "string"}`. Without this, progenitor emits a struct like
/// `MemoryItemTimestamp { #[serde(flatten)] subtype_0: Option<DateTime>, ... }`
/// where flatten on a primitive is a runtime error
/// (`"can only flatten structs and maps (got a string)"`). For client purposes
/// the `format: date-time` distinction is lossless — the wire value is just a
/// string either way — so collapsing to a plain string is safe and lets the
/// generated client actually serialize.
fn collapse_string_anyof_unions(value: &mut serde_json::Value) {
    if let serde_json::Value::Object(obj) = value {
        let all_strings = obj
            .get("anyOf")
            .and_then(|v| v.as_array())
            .map(|arr| {
                !arr.is_empty()
                    && arr.iter().all(|v| {
                        v.as_object()
                            .and_then(|m| m.get("type"))
                            .and_then(|t| t.as_str())
                            .map(|s| s == "string")
                            .unwrap_or(false)
                    })
            })
            .unwrap_or(false);

        if all_strings {
            obj.remove("anyOf");
            obj.insert("type".to_string(), serde_json::json!("string"));
        }
    }

    match value {
        serde_json::Value::Object(obj) => {
            for (_k, v) in obj.iter_mut() {
                collapse_string_anyof_unions(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr.iter_mut() {
                collapse_string_anyof_unions(v);
            }
        }
        _ => {}
    }
}

fn convert_anyof_to_nullable(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(obj) => {
            // Check if this object has anyOf with null and process it
            let has_null_in_anyof = obj
                .get("anyOf")
                .and_then(|v| v.as_array())
                .map(|array| {
                    array.iter().any(|v| {
                        v.get("type")
                            .and_then(|t| t.as_str())
                            .map(|s| s == "null")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

            if has_null_in_anyof {
                // Clone the anyOf array to avoid borrow issues
                if let Some(any_of) = obj.get("anyOf").cloned() {
                    if let Some(array) = any_of.as_array() {
                        let non_null_schemas: Vec<_> = array
                            .iter()
                            .filter(|v| {
                                v.get("type")
                                    .and_then(|t| t.as_str())
                                    .map(|s| s != "null")
                                    .unwrap_or(true)
                            })
                            .cloned()
                            .collect();

                        obj.remove("anyOf");
                        if non_null_schemas.len() == 1 {
                            // Single non-null type: inline it with nullable: true
                            if let Some(non_null_obj) = non_null_schemas[0].as_object() {
                                for (k, v) in non_null_obj.iter() {
                                    obj.insert(k.clone(), v.clone());
                                }
                            }
                        } else {
                            // Multiple non-null types: keep anyOf with nulls removed
                            obj.insert("anyOf".to_string(), serde_json::json!(non_null_schemas));
                        }
                        obj.insert("nullable".to_string(), serde_json::json!(true));
                    }
                }
            }

            // Recursively process all values
            for (_key, val) in obj.iter_mut() {
                convert_anyof_to_nullable(val);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                convert_anyof_to_nullable(item);
            }
        }
        _ => {}
    }
}

fn main() {
    // Get the OpenAPI spec path from hindsight-docs/static (single source of truth)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let openapi_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("hindsight-docs")
        .join("static")
        .join("openapi.json");

    // Tell Cargo to rebuild if the OpenAPI spec changes
    println!("cargo:rerun-if-changed={}", openapi_path.display());

    // Read the OpenAPI spec
    let spec_content = fs::read_to_string(&openapi_path)
        .expect("Failed to read openapi.json. Make sure it exists in the project root.");

    // Parse as generic JSON first to convert 3.1 to 3.0
    let mut spec_json: serde_json::Value =
        serde_json::from_str(&spec_content).expect("Failed to parse openapi.json");

    // Convert OpenAPI 3.1.0 to 3.0.3 for progenitor compatibility
    if let Some(version) = spec_json.get("openapi").and_then(|v| v.as_str()) {
        if version.starts_with("3.1") {
            eprintln!("Converting OpenAPI 3.1 to 3.0 for compatibility...");
            convert_31_to_30(&mut spec_json);
        }
    }

    // Collapse anyOf-of-only-strings into a plain string. Must run AFTER the
    // 3.1→3.0 nullable pass so we see `anyOf: [datetime, string]` without the
    // null arm. Keeps progenitor from emitting unserializable flatten-of-
    // primitive structs (see collapse_string_anyof_unions docs).
    collapse_string_anyof_unions(&mut spec_json);
    filter_binary_response_operations(&mut spec_json);

    // Filter out multipart/form-data endpoints (progenitor doesn't support them)
    filter_multipart_endpoints(&mut spec_json);

    // Now parse as OpenAPI struct
    let spec: openapiv3::OpenAPI =
        serde_json::from_value(spec_json).expect("Failed to parse converted OpenAPI spec");

    // Generate the client
    let mut generator = progenitor::Generator::default();

    // Generate code
    let tokens = generator
        .generate_tokens(&spec)
        .expect("Failed to generate client code from OpenAPI spec");

    // Write to the output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest_path = out_dir.join("hindsight_client_generated.rs");

    let syntax_tree = syn::parse2(tokens).expect("Failed to parse generated tokens");
    let mut formatted = prettyplease::unparse(&syntax_tree);

    // Fix progenitor bug with optional header parameters
    // The generated code tries to call .to_string() on Option<&str> which doesn't work
    // We need to unwrap the Option first
    formatted = fix_optional_header_params(&formatted);

    fs::write(&dest_path, formatted).expect("Failed to write generated client code");

    println!("Generated client at: {}", dest_path.display());
}

/// Fix progenitor's generated code for optional header parameters
/// Replaces patterns like `value.to_string().try_into()?` where value is Option<&str>
/// with `value.unwrap_or_default().to_string().try_into()?`
fn fix_optional_header_params(code: &str) -> String {
    use regex::Regex;

    // Pattern: header_map.append("authorization", value.to_string().try_into()?);
    // Should become: header_map.append("authorization", value.unwrap_or_default().to_string().try_into()?);
    let re = Regex::new(
        r#"header_map\.append\("authorization", value\.to_string\(\)\.try_into\(\)\?\)"#,
    )
    .expect("Invalid regex");

    re.replace_all(
        code,
        r#"header_map.append("authorization", value.unwrap_or_default().to_string().try_into()?)"#,
    )
    .to_string()
}
