# Mobile Agent - Examples

## Example 1: List Screen with CRUD

**Input**: "Create a todo list screen with add, toggle, and swipe-to-delete"

**Output**:
```
## Task: Todo List Screen

### Implementation
- Screens: TodoListScreen, AddTodoSheet (bottom sheet)
- State: Riverpod with AsyncNotifier for todo list
- API: Dio with auth interceptor -> /api/todos

### Platform Support
- [x] iOS (14.0+) - Cupertino swipe actions
- [x] Android (API 24+) - Material Dismissible
- [x] Dark mode

### Files Created
- lib/features/todo/domain/entities/todo.dart
- lib/features/todo/domain/repositories/todo_repository.dart
- lib/features/todo/data/models/todo_model.dart
- lib/features/todo/data/repositories/todo_repository_impl.dart
- lib/features/todo/presentation/providers/todo_provider.dart
- lib/features/todo/presentation/screens/todo_list_screen.dart
- lib/features/todo/presentation/widgets/todo_item.dart
- test/features/todo/todo_provider_test.dart
- test/features/todo/todo_list_screen_test.dart
```

## Example 2: Authentication Flow

**Input**: "Create login and register screens with JWT auth"

**Output**:
```
## Task: Authentication Screens

### Implementation
- Screens: LoginScreen, RegisterScreen
- State: Riverpod AuthNotifier (token storage via flutter_secure_storage)
- Navigation: GoRouter redirect guard for unauthenticated users
- Validation: Form validation for email + password

### Files Created
- lib/core/auth/auth_provider.dart
- lib/core/auth/auth_interceptor.dart (Dio interceptor)
- lib/features/auth/presentation/screens/login_screen.dart
- lib/features/auth/presentation/screens/register_screen.dart
- lib/features/auth/presentation/widgets/auth_form.dart
- test/core/auth/auth_provider_test.dart
```

## Example 3: Offline-First Feature

**Input**: "Make the notes feature work offline with sync"

**Output**:
```
## Task: Offline Notes with Sync

### Implementation
- Local DB: Drift (SQLite) for offline storage
- Sync: Queue-based sync on connectivity change
- UI: Sync status indicator (synced/pending/error)

### Architecture
- NoteLocalDataSource (Drift) + NoteRemoteDataSource (Dio)
- NoteRepository merges local + remote with conflict resolution
- ConnectivityProvider triggers sync on reconnect

### Files Created
- lib/features/notes/data/local/note_dao.dart
- lib/features/notes/data/local/note_database.dart
- lib/features/notes/data/repositories/note_sync_repository.dart
- lib/core/sync/sync_queue.dart
```

---

## Example 4: Swift Native iOS — Todo List Screen

**Platform**: Swift iOS Native (SwiftUI + `@Observable` + `swift-openapi-generator`)

**Input**: "Create a native iOS todo list screen that fetches todos from the API and supports pull-to-refresh"

**Output**:
```
## Task: Swift Native Todo List Screen (iOS 17+)

### Implementation
- State: @Observable TodosViewModel with TodosViewState enum (idle/loading/loaded/empty/error)
- Networking: TodoService wrapping the generated Client from swift-openapi-generator
- View: SwiftUI TodosView switching over TodosViewState; pull-to-refresh via .refreshable
- Tests: XCTest unit tests for the view model using a protocol-based mock service

### Platform Support
- [x] iOS 17.0+ (Observation framework requires iOS 17)
- [x] Dark mode (SwiftUI adaptive colors)
- [x] Dynamic Type (List / Label respect system font scaling)

### Project Layout (App/Core/Features/Shared)
Features/
  Todos/
    TodosView.swift             # SwiftUI view — owns the @Observable VM via @State
    TodosViewModel.swift        # @Observable; drives loading/error/empty/data states
Core/
  Networking/
    openapi.yaml                # Vendored OpenAPI spec; source of truth for generator
    openapi-generator-config.yaml
    APIClient.swift             # Wraps generated Client; URLSession transport + auth
    TodoProviding.swift         # Protocol seam the VM depends on (mockable)
    TodoService.swift           # Cached repository: generated Client + ResponseCache
  Cache/
    ResponseCache.swift         # actor over hyperoslo/Cache; read-through + invalidation
App/
  MyApp.swift                   # @main; instantiates AppDependencies
  AppDependencies.swift         # Composition root; injects TodoService into TodosView
Tests/
  TodosViewModelTests.swift     # XCTest; MockTodoService conforms to TodoProviding

### Files Created
- Sources/Features/Todos/TodosViewModel.swift
- Sources/Features/Todos/TodosView.swift
- Sources/Core/Networking/TodoProviding.swift
- Sources/Core/Networking/TodoService.swift
- Sources/Core/Networking/APIClient.swift
- Sources/Core/Cache/ResponseCache.swift
- Sources/App/AppDependencies.swift
- Tests/TodosViewModelTests.swift

### Key Patterns
- @Observable replaces ObservableObject/@Published — no Combine dependency
- View holds VM with @State (not @StateObject); init via State(wrappedValue:)
- .task { viewModel.load() } cancels automatically when view disappears
- TodoService calls the generated client — never hand-rolled URLRequest
- Read-through cache at the repository layer: ResponseCache (hyperoslo/Cache) wraps
  the generated Client; reads are stale-while-revalidate, writes invalidate keys
- VM depends on TodoProviding (protocol seam), so MockTodoService needs no mock lib
- deinit { loadTask?.cancel() } prevents Task leaks when VM is deallocated
```
