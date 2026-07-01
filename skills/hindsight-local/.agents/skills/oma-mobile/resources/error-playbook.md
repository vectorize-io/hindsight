# Mobile Agent - Error Recovery Playbook

When you encounter a failure, find the matching scenario and follow the recovery steps.
Do NOT stop or ask for help until you have exhausted the playbook.

---

## Dart Analysis Error

**Symptoms**: `flutter analyze` errors, type mismatch, null safety issues

1. Read the error: file, line, expected type vs actual
2. Null safety issue: add `?` for nullable, `!` only if you're certain it's non-null
3. Type mismatch: check the model class to see if it matches the API response
4. Missing override: add `@override` annotation
5. **NEVER do this**: `// ignore:` comments to suppress analysis warnings

---

## Build Failure

**Symptoms**: `flutter build` fails, Gradle/Xcode errors

1. **Gradle (Android)**:
   - `Could not resolve`: dependency version conflict → check `pubspec.yaml`
   - `minSdkVersion`: update `android/app/build.gradle` minimum SDK
2. **Xcode (iOS)**:
   - `Pod install` failure: note in result; may need `pod repo update`
   - Minimum deployment target: check `ios/Podfile`
3. Clean and retry: `flutter clean && flutter pub get`
4. If persists: note in result with full error; may be an environment issue

---

## Test Failure

**Symptoms**: `flutter test` FAILED, widget test assertion errors

1. Read the error: which test, which widget, expected vs actual
2. Widget test: check if `pumpAndSettle()` is needed (async operations)
3. Provider not found: wrap test widget with `ProviderScope` (Riverpod)
4. Mock missing: ensure all dependencies are mocked
5. Re-run specific test: `flutter test test/path/to_test.dart`
6. **After 3 failures**: Try a different approach

---

## State Management Issue

**Symptoms**: UI not updating, stale state, provider errors

1. **Riverpod**: Check provider type (`StateNotifierProvider` vs `FutureProvider` vs `AsyncNotifierProvider`)
2. Is the widget watching correctly? (`ref.watch` not `ref.read` for UI)
3. Is the state being mutated instead of replaced? (create new state object)
4. Add debug print in provider to trace state changes
5. Check: is `dispose` being called prematurely?

---

## Platform-Specific Crash

**Symptoms**: Works on one platform, crashes on another

1. Check for `Platform.isIOS` / `Platform.isAndroid` guards
2. Check permissions: camera, location, storage (different per platform)
3. Check native plugin compatibility; some plugins don't support both platforms
4. If plugin issue: note in result with platform and version info
5. Test on emulator for the failing platform

---

## Memory Leak

**Symptoms**: App slows down over time, `flutter run` shows increasing memory

1. Check: are all controllers disposed? (`TextEditingController`, `AnimationController`)
2. Check: are streams closed? (`StreamSubscription.cancel()`)
3. Check: are listeners removed? (`removeListener` in `dispose`)
4. Check: are `Timer` / `Timer.periodic` cancelled?
5. Use `DevTools` memory tab to identify leak source

---

## API Integration Error

**Symptoms**: Dio errors, `DioException`, wrong response parsing

1. **Connection refused**: backend running? correct URL/port?
2. **401**: auth interceptor sending token? token expired?
3. **Parse error**: `response.data` shape doesn't match model → log raw response
4. **Timeout**: increase Dio timeout or check network conditions
5. If backend issue: document expected contract in result

---

## Rate Limit / Quota / Memory Fallback

Same as the backend playbook: see `../../oma-backend/resources/error-playbook.md` §"Rate Limit / Quota Error (Gemini API)" and §"Serena Memory Unavailable".

---

## Swift: `swift build` Fails — Missing or Invalid OpenAPI Document

**Symptoms**: Build error from the `swift-openapi-generator` plugin such as `error: openapi.yaml not found` or YAML parse error; the build fails before any Swift file is compiled.

1. Confirm the spec exists at `Core/Networking/openapi.yaml` — this is where the build plugin looks.
2. If the file is absent: re-sync it from the backend (`curl -o Sources/Core/Networking/openapi.yaml https://<backend>/api-docs/openapi.yaml` or copy from the CI artifact). The iOS project is a **consumer** of the spec; never edit `openapi.yaml` directly.
3. If the file is present but the error is a parse error: validate the YAML (`python3 -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" Sources/Core/Networking/openapi.yaml`); fix the upstream spec and re-sync.
4. After syncing, `swift build` will regenerate `Client`, `Operations`, and `Components` automatically — no manual code-gen step needed.
5. **NEVER**: commit a placeholder or empty `openapi.yaml` just to silence the build; it will produce an incomplete `Client` and cause compile errors downstream.

---

## Swift: Generated `Client`/`Operations` Symbols Not Found

**Symptoms**: Compile errors such as `cannot find type 'Client' in scope` or `use of unresolved identifier 'Operations'`; the generator plugin appears not to have run.

1. Confirm the build plugin is attached in `Package.swift`:
   - The target must list `.plugin(name: "OpenAPIGenerator", package: "swift-openapi-generator")` under its `plugins:` key.
   - The target's dependencies must include `OpenAPIRuntime` and `OpenAPIURLSession`.
2. Confirm `openapi-generator-config.yaml` exists **in the same directory** as `openapi.yaml` (`Core/Networking/`). Without this file the plugin silently skips generation.
3. Check that both config files are inside the target's declared source directory (the directory that contains the `.target` in `Package.swift`). Files outside the source root are invisible to the build plugin.
4. Run `swift package clean && swift build` to force a full regeneration.
5. If Xcode is used: Product → Clean Build Folder, then build again. Derived data caching sometimes hides generation failures.

---

## Swift: Compile Error for Undocumented Response Case

**Symptoms**: Compile error such as `expression pattern of type 'Operations.CreateTodo.Output.conflict' cannot match values of type '...'`; a response case (e.g., `.conflict`, `.created`, `.unprocessableContent`) referenced in the Swift code does not exist on the generated `Output` enum.

1. The root cause is that the OpenAPI spec does not declare that HTTP status code for the operation. The generator only emits cases for status codes listed in the spec.
2. Do NOT add the case manually to the generated file — it will be overwritten on the next `swift build`.
3. Ask the backend team to add the missing status code to the spec (e.g., add `409: description: Conflict` under the operation's `responses:`).
4. Re-sync `Core/Networking/openapi.yaml` once the spec is updated.
5. Run `swift build` to regenerate; the new response case will appear and the compile error will resolve.
6. Update the Swift `switch` in the service layer to handle the new case explicitly.

---

## Swift: `@Observable` State Not Updating the SwiftUI View

**Symptoms**: Mutating a property on the view model has no visual effect; the view does not re-render even though the value changed.

1. Confirm the view model class is annotated with `@Observable` (Observation framework, **not** `ObservableObject`). Without the macro the view has no tracking infrastructure.
2. Check the deployment target: `@Observable` requires **iOS 17.0+**. If the target is set to iOS 16 or earlier the macro compiles but observation tracking is absent — raise the minimum deployment target in `Package.swift` (`.iOS(.v17)`) and the Xcode project settings.
3. Verify the view holds the VM with `@State` (not a plain `let` or `var`): `@State private var viewModel: MyViewModel`. A non-`@State` reference is not tracked by SwiftUI.
4. If the VM is passed into a child view that needs to mutate it, use `@Bindable` in the child: `@Bindable var viewModel: MyViewModel`. Passing it as a plain argument prevents change propagation.
5. Confirm properties are mutated on the **main actor**. Mutations from a background `Task` without `await MainActor.run { … }` (or `@MainActor` on the method) can update state off-main and miss the SwiftUI render pass.
6. Avoid wrapping the `@Observable` class in another `@Published` property or `ObservableObject` — mixing the two observation systems breaks change tracking.

---

## General Principles

- **After 3 failures**: If same approach fails 3 times, must try a different method
- **Blocked**: If no progress after 5 turns, save current state, `Status: blocked`
- **Out of scope**: Backend/frontend issues; only record in result
