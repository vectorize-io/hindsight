---
name: stack-set
description: Auto-detect project tech stack and generate stack-specific references for domain skills
disable-model-invocation: true
---

# /stack-set: Stack Configuration Workflow

## Goal
Analyze project files to detect the tech stack, resolve the target domain skill, then generate language-specific references in that skill's `stack/` directory.

> **Vendor note:** This workflow executes inline (no subagent spawning). All vendors use their native file reading tools for manifest detection and file writing tools for stack generation.

---

## Step 1: Detect

Scan the project root for package manifests. Evaluate **both** tables independently — do not stop at the first match.

### Backend manifests → domain `backend`

| File | Detection |
|:---|:---|
| `pyproject.toml`, `requirements.txt`, `Pipfile` | Python |
| `package.json`, `tsconfig.json` | Node.js/TypeScript |
| `Cargo.toml` | Rust |
| `pom.xml`, `build.gradle`, `build.gradle.kts` | Java/Kotlin |
| `go.mod` | Go |
| `mix.exs` | Elixir |
| `Gemfile` | Ruby |
| `*.csproj`, `*.sln` | C#/.NET |

Read manifest contents to detect framework:
- Python: FastAPI? Django? Flask?
- Node.js: NestJS? Express? Hono?
- Rust: Axum? Actix-web? Rocket?
- Java: Spring Boot? Quarkus?

### Mobile manifests → domain `mobile`

| File | Detection |
|:---|:---|
| `Package.swift`, `*.xcodeproj`, `*.xcworkspace`, `Podfile` | Swift / iOS native |
| `pubspec.yaml` | Flutter |
| `package.json` with a `react-native` dependency | React Native |

For Swift / iOS native, additionally inspect the project to detect the UI framework:
- Look for `import SwiftUI` in source files or a `SwiftUI` framework dependency in `Package.swift`.
- Default to **SwiftUI** when ambiguous.

### Resolve `target_skill`

After scanning both tables, record every domain that has at least one detected manifest file:

| Detected domain | `target_skill` |
|:---|:---|
| backend only | `oma-backend` |
| mobile only | `oma-mobile` |
| both (monorepo) | carry **both** into Step 2 — do NOT first-match |

**Multi-domain rule (amendment B):** If manifests from both backend and mobile tables are found, collect both domains as the detected set and proceed to Step 2. Do not silently discard either.

---

## Step 2: Confirm

### Single-domain: backend

Present detection results and ask for confirmation:
```
Detected backend stack:
  Language: {language}
  Framework: {framework}
  ORM: {orm}
  Validation: {validation}
  Migration: {migration}
  Test: {test framework}

Correct? (Y/n) or modify:
```

### Single-domain: mobile (Swift)

Present detection results and ask for confirmation:
```
Detected mobile stack:
  Language: {language}        (e.g. Swift)
  UI: {ui}                    (e.g. SwiftUI)
  API generator: {api_generator}   (e.g. swift-openapi-generator)
  API spec: {api_spec}        (e.g. Core/Networking/openapi.yaml)
  Structure: {structure}      (e.g. App/Core/Features/Shared)
  Test: {test}                (e.g. XCTest)

Correct? (Y/n) or modify:
```

For Flutter or React Native mobile detection, present an equivalent confirmation block using the relevant fields (framework, sdk version, test framework, etc.).

### Multi-domain: present choice first

When more than one domain was detected in Step 1, **before** showing any per-domain confirm block, ask:

```
Multiple domains detected in this repo:
  [backend]  {backend_language} / {backend_framework}
  [mobile]   {mobile_language} / {mobile_ui}

Generate stack references for: [both / backend / mobile]
```

After the user selects, show the per-domain confirmation block(s) for the chosen domain(s) and confirm each before generating.

---

## Step 3: Generate

Write generated files into `.agents/skills/{target_skill}/stack/`.

- Backend → `.agents/skills/oma-backend/stack/`
- Mobile → `.agents/skills/oma-mobile/stack/`
- Multi-domain → run the appropriate generation sub-path for each selected domain in turn.

---

### Backend path

#### stack.yaml
```yaml
language: {language}
framework: {framework}
orm: {orm}
validation: {validation}
migration: {migration}
test: {test_framework}
source: detected
detected_from:
  - {manifest_file}
verify:                          # consumed by `oma verify backend` (see _shared/core/stack-verify.schema.json)
  detect: {manifest_file}        # e.g. package.json, pyproject.toml
  syntax:
    cmd: "{syntax_check_cmd}"    # e.g. bunx tsc --noEmit
  tests:
    cmd: "{test_cmd}"            # e.g. bun test
    skip_if_missing: "{optional_binary}"
```

#### tech-stack.md
Generate tech stack reference with these MANDATORY sections:
- Framework version and core API
- ORM/DB library and usage
- Validation library
- Migration tool
- Test framework
- Linter/formatter

#### snippets.md
Generate copy-paste code patterns. MANDATORY patterns (all 8 required):
- [ ] Route/Handler + Auth example
- [ ] Validation Schema example
- [ ] ORM Model/Entity example
- [ ] DI (Dependency Injection) example
- [ ] Repository pattern example
- [ ] Paginated Query example
- [ ] Migration example
- [ ] Test example

#### api-template.*
Generate CRUD endpoint boilerplate in the detected language.

---

### Mobile path — Swift / iOS native

**Adapt, not copy (amendment E):** Seed from `.agents/skills/oma-mobile/variants/swift-ios/` as the baseline, then adapt every value to match the detected project. Specifically:

- Replace placeholder `Features/` module names with the actual feature module names found in the project (e.g., `Features/Auth`, `Features/Home`).
- Set `api_spec` to the actual path where the OpenAPI document lives in this project (default `Core/Networking/openapi.yaml` only when no other location is found).
- Set the minimum iOS deployment target to the value detected from `Package.swift` or `.xcodeproj`; default `17.0` when not specified.
- Populate the DI wiring in the App entry snippet with the real `Client` and service types from the project, not generic placeholders.

Do **not** blind-copy the variant files; the generated `stack/` must be project-specific.

#### stack.yaml — mandatory fields
```yaml
language: swift
framework: swiftui           # or uikit — adapt to detected value
ui: swiftui                  # adapt to detected value
api_generator: swift-openapi-generator
api_spec: {actual_path_to_openapi_yaml}
structure: App/Core/Features/Shared
test: XCTest
source: detected
detected_from:
  - {manifest_file}          # e.g. Package.swift
verify:                      # consumed by `oma verify mobile` (see _shared/core/stack-verify.schema.json)
  detect: Package.swift
  syntax:
    cmd: "swift build"
    skip_if_missing: "swift"
  tests:
    cmd: "swift test"
    skip_if_missing: "swift"
```

#### tech-stack.md
Generate a Swift-specific tech stack reference with these MANDATORY sections:
- SwiftUI + Observation framework (`@Observable`, iOS 17+)
- `swift-openapi-generator` + `swift-openapi-runtime` + `swift-openapi-urlsession` — SwiftPM build plugin wiring
- `App/Core/Features/Shared` module layout (App = entry/composition root, Core = networking/generated client/DI, Features = screen + view-model verticals, Shared = reusable UI/utils)
- API spec provenance: where `{api_spec}` comes from, how it is kept in sync with the backend producer
- Test framework (XCTest / Swift Testing)
- Linter/formatter (SwiftLint if present)

#### snippets.md — mandatory Swift snippet set (all required)
- [ ] `Package.swift` with the `OpenAPIGenerator` build plugin declared and `openapi.yaml` spec discovery configured
- [ ] `openapi-generator-config.yaml` — generator configuration (namespace, accessibility, etc.)
- [ ] `@Observable` view model (Observation framework, async data loading, error state)
- [ ] SwiftUI feature view consuming the view model
- [ ] `Core/Networking` service wrapping the generated `Client` (URLSession transport, bearer auth middleware)
- [ ] Generated-client call pattern (`Operations.listItems`, `Operations.createItem`, etc.)
- [ ] App entry point + DI composition root (wiring `Client` → service → view model)
- [ ] XCTest example for the service or view model

#### api-template.swift
Generate a CRUD service built on the generated `Client` (using `Operations.*` call patterns from `swift-openapi-generator`). This is the Swift analogue of the backend `api-template.*`. The template must:
- Import and instantiate the generated `Client` (not a hand-rolled `URLSession` request builder).
- Implement list, get-by-id, create, update, and delete operations using `Operations.*` types.
- Handle transport errors and map them to domain error types.
- Use `async/await` throughout.

---

## Step 4: Verify

Confirm generated files meet requirements.

### Backend checks
- [ ] `stack.yaml` has `language`, `framework`, `orm`, `validation` fields
- [ ] `stack.yaml` has a `verify:` block with runnable `syntax.cmd` and `tests.cmd` (otherwise `oma verify backend` cannot dispatch)
- [ ] `snippets.md` contains all 8 mandatory patterns
- [ ] `tech-stack.md` contains all 6 mandatory sections
- [ ] `api-template` file uses the correct language extension
- [ ] Code follows existing project conventions

### Mobile (Swift) checks
- [ ] `stack.yaml` has `language`, `api_generator`, `api_spec`, and `structure` fields populated with project-specific values (not variant defaults)
- [ ] `stack.yaml` has a `verify:` block with runnable `syntax.cmd` and `tests.cmd` (otherwise `oma verify mobile` cannot dispatch)
- [ ] `snippets.md` includes the generator configuration snippet (`openapi-generator-config.yaml`) and at least one snippet that uses the generated `Client` via `Operations.*`
- [ ] `api-template.swift` uses the generated client — not hand-rolled `URLSession` request construction
- [ ] `tech-stack.md` documents where `api_spec` originates and how it syncs from the backend producer
- [ ] Module names in snippets reflect real `Features/` modules detected in the project, not generic placeholders

---

## Constraints

- Do NOT modify `.agents/skills/{target_skill}/SKILL.md` (abstract interface is protected)
- Do NOT modify `resources/` common files under any skill
- Only create or modify files in the resolved skill's `stack/` directory
- If `stack/` already exists for the resolved domain skill, ask before overwriting
- `target_skill` is always the resolved domain skill (`oma-backend` or `oma-mobile`); never hardcode a single skill name in generation logic
