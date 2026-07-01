/**
 * Screen Template for Mobile Agent (Swift iOS Native)
 *
 * Demonstrates best practices for SwiftUI screens:
 *   - @Observable view model with explicit loading/error/empty/data states
 *   - .task modifier for async data loading (auto-cancelled on disappear)
 *   - Retry action wired to the view model
 *   - NavigationStack integration
 *   - iOS HIG-aligned layout
 */

import SwiftUI
import Observation

// MARK: - View Model

/// All possible display states for the example screen.
enum ExampleViewState {
    case idle
    case loading
    case loaded([ExampleItem])
    case empty
    case error(String)
}

/// A plain value type representing one row in the list.
struct ExampleItem: Identifiable {
    let id: String
    let title: String
    let subtitle: String
}

/// Protocol-backed so the view model is testable with a mock service.
protocol ExampleServiceProtocol {
    func fetchItems() async throws -> [ExampleItem]
}

@Observable
final class ExampleViewModel {
    // MARK: - State observed by the View

    /// Current display state. The View switches on this value.
    var viewState: ExampleViewState = .idle

    // MARK: - Private

    private let service: ExampleServiceProtocol
    /// Retained so it can be cancelled before re-triggering a load.
    private var loadTask: Task<Void, Never>?

    init(service: ExampleServiceProtocol) {
        self.service = service
    }

    // MARK: - Intents (called by the View)

    /// Starts (or restarts) data loading. Safe to call multiple times.
    func load() {
        // Cancel any in-flight request before starting a fresh one.
        loadTask?.cancel()
        viewState = .loading

        loadTask = Task { [weak self] in
            guard let self else { return }
            do {
                let items = try await service.fetchItems()
                guard !Task.isCancelled else { return }
                viewState = items.isEmpty ? .empty : .loaded(items)
            } catch is CancellationError {
                // Ignore — another load is replacing this one.
            } catch {
                viewState = .error(error.localizedDescription)
            }
        }
    }

    /// Convenience retry; identical to load() but named for the error-state button.
    func retry() { load() }

    deinit {
        // Ensure in-flight work is cleaned up when the VM is deallocated.
        loadTask?.cancel()
    }
}

// MARK: - View

struct ExampleScreen: View {
    // The View owns the view model via @State so it is scoped to this screen.
    @State private var viewModel: ExampleViewModel

    init(service: ExampleServiceProtocol) {
        // Wrap in State so @Observable tracking works correctly in SwiftUI.
        _viewModel = State(wrappedValue: ExampleViewModel(service: service))
    }

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Example")
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button {
                            viewModel.load()
                        } label: {
                            Label("Refresh", systemImage: "arrow.clockwise")
                        }
                        // Disable the button while a load is in progress.
                        .disabled({
                            if case .loading = viewModel.viewState { return true }
                            return false
                        }())
                    }
                }
        }
        // .task is preferred over .onAppear for async work:
        // it creates a structured Task that is cancelled when the View disappears.
        .task { viewModel.load() }
    }

    // MARK: - Content switch

    /// Switches over the view model's state and renders the appropriate sub-view.
    @ViewBuilder
    private var content: some View {
        switch viewModel.viewState {
        case .idle, .loading:
            loadingView

        case .loaded(let items):
            listView(items)

        case .empty:
            emptyView

        case .error(let message):
            errorView(message: message)
        }
    }

    // MARK: - State sub-views

    /// Shown while the first load is in progress.
    private var loadingView: some View {
        ProgressView("Loading…")
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Shown when data is available.
    private func listView(_ items: [ExampleItem]) -> some View {
        List(items) { item in
            NavigationLink {
                // Replace with a real detail screen.
                Text(item.title)
            } label: {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.title)
                        .font(.headline)
                    Text(item.subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
        }
        // Pull-to-refresh triggers a fresh load.
        .refreshable { viewModel.load() }
    }

    /// Shown when the API returns an empty collection.
    private var emptyView: some View {
        ContentUnavailableView(
            "Nothing Here",
            systemImage: "tray",
            description: Text("There are no items yet. Create one to get started.")
        )
    }

    /// Shown when the load fails. Includes a labelled retry button.
    private func errorView(message: String) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.red)

            Text("Something went wrong")
                .font(.title2)
                .bold()

            Text(message)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            Button {
                viewModel.retry()
            } label: {
                Label("Try Again", systemImage: "arrow.clockwise")
                    .padding(.horizontal, 8)
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: - Preview

#Preview {
    // Provide a lightweight stub for canvas previews.
    struct PreviewService: ExampleServiceProtocol {
        func fetchItems() async throws -> [ExampleItem] {
            try? await Task.sleep(nanoseconds: 500_000_000)
            return [
                ExampleItem(id: "1", title: "First Item",  subtitle: "Detail about the first item"),
                ExampleItem(id: "2", title: "Second Item", subtitle: "Detail about the second item"),
            ]
        }
    }
    return ExampleScreen(service: PreviewService())
}
