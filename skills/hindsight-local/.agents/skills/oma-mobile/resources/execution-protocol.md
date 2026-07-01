# Mobile Agent - Execution Protocol

## Step 0: Prepare
1. **Assess difficulty**: see `../../_shared/core/difficulty-guide.md`
   - **Simple**: Skip to Step 3 | **Medium**: All 4 steps | **Complex**: All steps + checkpoints
2. **Check lessons**: read your domain section in `../../_shared/core/lessons-learned.md`
3. **Clarify requirements**: follow `../../_shared/core/clarification-protocol.md`
   - Check **Uncertainty Triggers**: business logic, security/auth, existing code conflicts?
   - Determine level: LOW → proceed | MEDIUM → present options | HIGH → ask immediately
4. **Budget context**: follow `../../_shared/core/context-budget.md` (read symbols, not whole files)

**Intelligent Escalation**: When uncertain, escalate early. Don't blindly proceed.

Follow these steps in order (adjust depth by difficulty).

## Step 1: Analyze
- Read the task requirements carefully
- Identify target platform: check for `Package.swift` (Swift iOS), `pubspec.yaml` (Flutter), or `package.json` + `react-native` dep (React Native)
- **If Swift (Package.swift detected)**: identify which `Features/` modules are affected; check for `Core/Networking/openapi.yaml`
- **If Flutter**: identify screens, widgets, and Riverpod/Bloc providers
- Check existing code with Serena: `get_symbols_overview("Sources/Features")` (Swift) or `get_symbols_overview("lib/features")` (Flutter)
- Determine platform-specific requirements (iOS HIG vs Material Design 3)
- List assumptions; ask if unclear

## Step 2: Plan
- **Swift**: plan using `App/Core/Features/Shared` layers; define the `@Observable` view model state enum; identify which `Operations` + `Components` types the feature needs from the generated `Client`
- **Flutter**: decide on feature structure using Clean Architecture; define entities (domain) and repository interfaces; plan state management (Riverpod providers); identify navigation routes (GoRouter)
- Plan offline-first strategy if required
- Note platform differences (iOS HIG vs Material Design 3)

## Step 3: Implement
- Create/modify files in this order:
  1. Domain: entities and repository interfaces
  2. Data: models, API clients (Dio), repository implementations
  3. Presentation: providers (Riverpod), screens, widgets
  4. Navigation: GoRouter routes
  5. Tests: unit + widget tests
- Use `resources/screen-template.dart` as reference
- Follow Clean Architecture layers strictly

## Step 4: Verify
- Run `resources/checklist.md` items
- Run `../../_shared/core/common-checklist.md` items
- Test on both iOS and Android (or emulators)
- Verify 60fps performance (no jank)
- Check dark mode support

## On Error
See `resources/error-playbook.md` for recovery steps.
