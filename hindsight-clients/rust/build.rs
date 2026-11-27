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
                                    eprintln!("Filtering out endpoint with multipart/form-data: {}", path_name);
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

/// Drop error responses that carry no schema (the documented 404 on bank-scoped
/// reads, say).
///
/// progenitor models at most one error type per operation --
/// `assertion failed: response_types.len() <= 1` in method.rs -- and FastAPI
/// already gives every one of these operations a typed 422. A bodyless 404 adds
/// a second, empty error type and panics the generator. Dropping it costs the
/// Rust client nothing: an undeclared status still surfaces as
/// `Error::UnexpectedResponse`, status code intact.
fn filter_bodyless_error_responses(spec: &mut serde_json::Value) {
    let Some(paths) = spec.get_mut("paths").and_then(|v| v.as_object_mut()) else {
        return;
    };
    for (_path_name, path_item) in paths.iter_mut() {
        let Some(operations) = path_item.as_object_mut() else {
            continue;
        };
        for (_method, operation) in operations.iter_mut() {
            let Some(responses) = operation.get_mut("responses").and_then(|v| v.as_object_mut()) else {
                continue;
            };
            responses.retain(|status, response| {
                let is_error = status
                    .chars()
                    .next()
                    .map(|c| c == '4' || c == '5')
                    .unwrap_or(false);
                !(is_error && response.get("content").is_none())
            });
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
            let has_null_in_anyof = obj.get("anyOf")
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
                        let non_null_schemas: Vec<_> = array.iter().filter(|v| {
                            v.get("type")
                                .and_then(|t| t.as_str())
                                .map(|s| s != "null")
                                .unwrap_or(true)
                        }).cloned().collect();

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

/// Rename inline enums whose `title` collides with a different enum.
///
/// typify names an inline enum after its `title`, and pydantic titles a field
/// (FastAPI: a query parameter) after its name. So two models whose `outcome`
/// (or `kind`) fields allow different values collapse into ONE generated
/// `Outcome` type -- whichever is generated first -- and the other model then
/// rejects values its own schema allows
/// (`RefreshMentalModelOperationDetails.outcome = "refresh_failed_error"`,
/// `PromptBlockModel.kind = "text"`). Query parameters collide the same way
/// (#6997: two endpoints' `detail` params). The first enum with a title keeps
/// it, so no existing type is renamed; later ones with a different value set
/// become `{Schema}{Title}` for a model property and `{OperationId}{Title}` for
/// a query parameter.
fn dedupe_inline_enum_titles(spec: &mut serde_json::Value) {
    let mut first_values: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    if let Some(schemas) = spec
        .pointer_mut("/components/schemas")
        .and_then(|v| v.as_object_mut())
    {
        for (schema_name, schema) in schemas.iter_mut() {
            let Some(props) = schema.get_mut("properties").and_then(|v| v.as_object_mut()) else {
                continue;
            };
            let prefix: String = schema_name.chars().filter(|c| c.is_alphanumeric()).collect();
            for prop in props.values_mut() {
                dedupe_enum_title(prop, &prefix, &mut first_values);
            }
        }
    }
    let Some(paths) = spec.get_mut("paths").and_then(|v| v.as_object_mut()) else {
        return;
    };
    for operation in paths.values_mut().filter_map(|p| p.as_object_mut()).flat_map(|p| p.values_mut()) {
        let prefix = pascal_case(operation.get("operationId").and_then(|v| v.as_str()).unwrap_or(""));
        let Some(params) = operation.get_mut("parameters").and_then(|v| v.as_array_mut()) else {
            continue;
        };
        for schema in params.iter_mut().filter_map(|p| p.get_mut("schema")) {
            dedupe_enum_title(schema, &prefix, &mut first_values);
        }
    }
}

/// Keep `schema`'s enum title if it is the first with that title or has the
/// same values as the first; otherwise prefix it with `prefix`.
fn dedupe_enum_title(
    schema: &mut serde_json::Value,
    prefix: &str,
    first_values: &mut std::collections::HashMap<String, Vec<String>>,
) {
    let (Some(title), Some(values)) = (
        schema.get("title").and_then(|t| t.as_str()).map(str::to_owned),
        schema.get("enum").and_then(|e| e.as_array()),
    ) else {
        return;
    };
    let mut values: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    values.sort();
    match first_values.get(&title) {
        None => {
            first_values.insert(title, values);
        }
        Some(first) if *first == values => {}
        Some(_) => {
            let title: String = title.chars().filter(|c| c.is_alphanumeric()).collect();
            schema["title"] = serde_json::json!(format!("{prefix}{title}"));
        }
    }
}

/// `list_mental_models` -> `ListMentalModels`.
fn pascal_case(snake: &str) -> String {
    snake
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            chars.next().map(|c| c.to_uppercase().chain(chars).collect::<String>()).unwrap_or_default()
        })
        .collect()
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

    // Collapse anyOf-of-only-strings into a plain string. Must run AFTER the
    // 3.1→3.0 nullable pass so we see `anyOf: [datetime, string]` without the
    // null arm. Keeps progenitor from emitting unserializable flatten-of-
    // primitive structs (see collapse_string_anyof_unions docs).
    collapse_string_anyof_unions(&mut spec_json);

    // Give same-titled inline enums with different values distinct names, so
    // one model's enum doesn't silently replace another's.
    dedupe_inline_enum_titles(&mut spec_json);

    // Filter out multipart/form-data endpoints (progenitor doesn't support them)
    filter_multipart_endpoints(&mut spec_json);

    // Drop schema-less error responses; progenitor allows only one error type
    // per operation and the typed 422 is the one worth keeping.
    filter_bodyless_error_responses(&mut spec_json);

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

    let mut syntax_tree: syn::File = syn::parse2(tokens)
        .expect("Failed to parse generated tokens");

    // Let every enum take a value or shape the server added after this client
    // was built, instead of failing the whole response (see accept_unknown_values).
    let mut opened = OpenedEnums::default();
    accept_unknown_values(&mut syntax_tree.items, &mut opened);
    display_unknown_values(&mut syntax_tree.items, &opened);

    let mut formatted = prettyplease::unparse(&syntax_tree);

    // Fix progenitor bug with optional header parameters
    // The generated code tries to call .to_string() on Option<&str> which doesn't work
    // We need to unwrap the Option first
    formatted = fix_optional_header_params(&formatted);

    fs::write(&dest_path, formatted)
        .expect("Failed to write generated client code");

    println!("Generated client at: {}", dest_path.display());
}

/// Names of the enums `accept_unknown_values` opened, by kind; their `Display`
/// needs an arm for the new variant.
#[derive(Default)]
struct OpenedEnums {
    strings: std::collections::HashSet<String>,
    unions: std::collections::HashSet<String>,
}

/// Add a catch-all last variant to every generated enum (#6912).
///
/// typify turns each OpenAPI `enum` into a closed Rust enum, so a value the
/// server adds later (the server only ever adds them) fails serde with "unknown
/// variant" and takes the whole response down -- and the `hindsight` CLI with
/// it. The same goes for a union whose shape is newer than the client.
///
/// - A string enum gets `#[serde(untagged)] Unknown(String)`: serde tries the
///   named variants first, so known values parse exactly as before, and an
///   unknown one keeps its text and serializes back unchanged. It drops `Copy`,
///   which a `String` field rules out. `FromStr` stays strict, so a value typed
///   by a user is still checked.
/// - An untagged union gets `Unknown(serde_json::Value)`; untagged variants are
///   tried in order, so it only catches what no known shape matches.
fn accept_unknown_values(items: &mut [syn::Item], opened: &mut OpenedEnums) {
    use syn::punctuated::Punctuated;
    use syn::{parse_quote, Fields, Item, Path, Token};

    for item in items {
        match item {
            Item::Mod(module) => {
                if let Some((_, inner)) = &mut module.content {
                    accept_unknown_values(inner, opened);
                }
            }
            Item::Enum(e) => {
                let derives_serde = e.attrs.iter().any(|a| {
                    a.path().is_ident("derive") && attr_args(a).contains("Deserialize")
                });
                if !derives_serde {
                    continue;
                }
                let untagged = e
                    .attrs
                    .iter()
                    .any(|a| a.path().is_ident("serde") && attr_args(a).contains("untagged"));
                if e.variants.iter().all(|v| matches!(v.fields, Fields::Unit)) {
                    for attr in &mut e.attrs {
                        if attr.path().is_ident("derive") {
                            let kept: Vec<Path> = attr
                                .parse_args_with(Punctuated::<Path, Token![,]>::parse_terminated)
                                .expect("derive attribute")
                                .into_iter()
                                .filter(|p| !p.is_ident("Copy"))
                                .collect();
                            *attr = parse_quote!(#[derive(#(#kept),*)]);
                        }
                    }
                    e.variants.push(parse_quote! {
                        /// A value added on the server after this client was built.
                        #[serde(untagged)]
                        Unknown(::std::string::String)
                    });
                    opened.strings.insert(e.ident.to_string());
                } else if untagged {
                    e.variants.push(parse_quote! {
                        /// A shape added on the server after this client was built.
                        Unknown(::serde_json::Value)
                    });
                    opened.unions.insert(e.ident.to_string());
                }
            }
            _ => {}
        }
    }
}

/// Give each opened enum's `Display` (an exhaustive match) an arm for the
/// `Unknown` variant added by `accept_unknown_values`; it prints the value (a
/// union's as JSON). A string enum matches on `*self`, a union on `self`.
fn display_unknown_values(items: &mut [syn::Item], opened: &OpenedEnums) {
    use syn::{parse_quote, Expr, ImplItem, Item, Stmt, Type};

    for item in items {
        match item {
            Item::Mod(module) => {
                if let Some((_, inner)) = &mut module.content {
                    display_unknown_values(inner, opened);
                }
            }
            Item::Impl(imp) => {
                let is_display = imp
                    .trait_
                    .as_ref()
                    .and_then(|(_, path, _)| path.segments.last())
                    .is_some_and(|s| s.ident == "Display");
                let Type::Path(self_ty) = imp.self_ty.as_ref() else {
                    continue;
                };
                let name = self_ty.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
                let arm: syn::Arm = if opened.strings.contains(&name) {
                    parse_quote!(Self::Unknown(ref value) => f.write_str(value),)
                } else if opened.unions.contains(&name) {
                    parse_quote!(Self::Unknown(value) => value.fmt(f),)
                } else {
                    continue;
                };
                if !is_display {
                    continue;
                }
                for impl_item in &mut imp.items {
                    if let ImplItem::Fn(func) = impl_item {
                        match func.block.stmts.last_mut() {
                            Some(Stmt::Expr(Expr::Match(m), _)) => {
                                m.arms.push(arm.clone());
                            }
                            _ => panic!("Display for {name} is not a match; update display_unknown_values"),
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// The text inside an attribute's parentheses, e.g. `untagged` for `#[serde(untagged)]`.
fn attr_args(attr: &syn::Attribute) -> String {
    match &attr.meta {
        syn::Meta::List(list) => list.tokens.to_string(),
        _ => String::new(),
    }
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
