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

fn convert_anyof_to_nullable(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(obj) => {
            // Check if this object has anyOf with null and process it
            let should_convert = obj.get("anyOf")
                .and_then(|v| v.as_array())
                .map(|array| {
                    if array.len() == 2 {
                        let has_null = array.iter().any(|v| {
                            v.get("type")
                                .and_then(|t| t.as_str())
                                .map(|s| s == "null")
                                .unwrap_or(false)
                        });
                        has_null
                    } else {
                        false
                    }
                })
                .unwrap_or(false);

            if should_convert {
                // Clone the anyOf array to avoid borrow issues
                if let Some(any_of) = obj.get("anyOf").cloned() {
                    if let Some(array) = any_of.as_array() {
                        // Find the non-null schema
                        if let Some(non_null_schema) = array.iter().find(|v| {
                            v.get("type")
                                .and_then(|t| t.as_str())
                                .map(|s| s != "null")
                                .unwrap_or(true)
                        }).cloned() {
                            // Replace anyOf with the non-null schema + nullable: true
                            obj.remove("anyOf");
                            if let Some(non_null_obj) = non_null_schema.as_object() {
                                for (k, v) in non_null_obj.iter() {
                                    obj.insert(k.clone(), v.clone());
                                }
                            }
                            obj.insert("nullable".to_string(), serde_json::json!(true));
                        }
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
    let mut spec_json: serde_json::Value = serde_json::from_str(&spec_content)
        .expect("Failed to parse openapi.json");

    // Convert OpenAPI 3.1.0 to 3.0.3 for progenitor compatibility
    if let Some(version) = spec_json.get("openapi").and_then(|v| v.as_str()) {
        if version.starts_with("3.1") {
            eprintln!("Converting OpenAPI 3.1 to 3.0 for compatibility...");
            convert_31_to_30(&mut spec_json);
        }
    }

    // Now parse as OpenAPI struct
    let spec: openapiv3::OpenAPI = serde_json::from_value(spec_json)
        .expect("Failed to parse converted OpenAPI spec");

    // Generate the client
    let mut generator = progenitor::Generator::default();

    // Generate code
    let tokens = generator.generate_tokens(&spec)
        .expect("Failed to generate client code from OpenAPI spec");

    // Write to the output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest_path = out_dir.join("hindsight_client_generated.rs");

    let syntax_tree = syn::parse2(tokens)
        .expect("Failed to parse generated tokens");
    let mut formatted = prettyplease::unparse(&syntax_tree);

    // Fix progenitor bug with optional header parameters
    // The generated code tries to call .to_string() on Option<&str> which doesn't work
    // We need to unwrap the Option first
    formatted = fix_optional_header_params(&formatted);

    fs::write(&dest_path, formatted)
        .expect("Failed to write generated client code");

    println!("Generated client at: {}", dest_path.display());
}

/// Fix progenitor's generated code for optional header parameters
/// Replaces patterns like `value.to_string().try_into()?` where value is Option<&str>
/// with `value.unwrap_or_default().to_string().try_into()?`
fn fix_optional_header_params(code: &str) -> String {
    use regex::Regex;

    // Pattern: header_map.append("authorization", value.to_string().try_into()?);
    // Should become: header_map.append("authorization", value.unwrap_or_default().to_string().try_into()?);
    let re = Regex::new(r#"header_map\.append\("authorization", value\.to_string\(\)\.try_into\(\)\?\)"#)
        .expect("Invalid regex");

    re.replace_all(code, r#"header_map.append("authorization", value.unwrap_or_default().to_string().try_into()?)"#)
        .to_string()
}
