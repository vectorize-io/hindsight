/**
 * Screen Template for Mobile Agent (Flutter)
 *
 * This template demonstrates best practices for Flutter screens.
 */

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

// Provider definition (in separate file: providers/example_providers.dart)
/*
final exampleProvider = FutureProvider<List<ExampleModel>>((ref) async {
  final repository = ref.watch(exampleRepositoryProvider);
  return repository.fetchData();
});
*/

/// Example screen demonstrating common patterns
class ExampleScreen extends ConsumerStatefulWidget {
  /// Route name for navigation
  static const routeName = '/example';

  /// Constructor
  const ExampleScreen({super.key});

  @override
  ConsumerState<ExampleScreen> createState() => _ExampleScreenState();
}

class _ExampleScreenState extends ConsumerState<ExampleScreen> {
  // Local state (if needed)
  final _scrollController = ScrollController();
  bool _showScrollToTop = false;

  @override
  void initState() {
    super.initState();
    _setupScrollListener();
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _setupScrollListener() {
    _scrollController.addListener(() {
      final shouldShow = _scrollController.offset > 200;
      if (shouldShow != _showScrollToTop) {
        setState(() => _showScrollToTop = shouldShow);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    // Watch providers
    final dataAsync = ref.watch(exampleProvider);

    return Scaffold(
      // App bar
      appBar: AppBar(
        title: const Text('Example Screen'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.invalidate(exampleProvider),
            tooltip: 'Refresh',
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _navigateToSettings,
            tooltip: 'Settings',
          ),
        ],
      ),

      // Body with async handling
      body: dataAsync.when(
        // Success state
        data: (items) => _buildContent(items),

        // Loading state
        loading: () => const Center(
          child: CircularProgressIndicator(),
        ),

        // Error state
        error: (error, stackTrace) => _buildErrorState(error),
      ),

      // Floating action button
      floatingActionButton: _showScrollToTop
          ? FloatingActionButton(
              onPressed: _scrollToTop,
              child: const Icon(Icons.arrow_upward),
            )
          : null,
    );
  }

  /// Builds main content
  Widget _buildContent(List<dynamic> items) {
    if (items.isEmpty) {
      return _buildEmptyState();
    }

    return RefreshIndicator(
      onRefresh: _handleRefresh,
      child: CustomScrollView(
        controller: _scrollController,
        slivers: [
          // Header
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                '${items.length} Items',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
          ),

          // List
          SliverList(
            delegate: SliverChildBuilderDelegate(
              (context, index) {
                final item = items[index];
                return _buildListItem(item, index);
              },
              childCount: items.length,
            ),
          ),
        ],
      ),
    );
  }

  /// Builds individual list item
  Widget _buildListItem(dynamic item, int index) {
    return Card(
      margin: const EdgeInsets.symmetric(
        horizontal: 16,
        vertical: 8,
      ),
      child: ListTile(
        leading: CircleAvatar(
          child: Text('${index + 1}'),
        ),
        title: Text(item.title ?? 'Untitled'),
        subtitle: Text(item.description ?? ''),
        trailing: IconButton(
          icon: const Icon(Icons.chevron_right),
          onPressed: () => _navigateToDetail(item),
        ),
        onTap: () => _navigateToDetail(item),
      ),
    );
  }

  /// Builds empty state
  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.inbox,
            size: 64,
            color: Theme.of(context).colorScheme.secondary,
          ),
          const SizedBox(height: 16),
          Text(
            'No items yet',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 8),
          Text(
            'Add your first item to get started',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
                ),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: _showAddDialog,
            icon: const Icon(Icons.add),
            label: const Text('Add Item'),
          ),
        ],
      ),
    );
  }

  /// Builds error state
  Widget _buildErrorState(Object error) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              size: 64,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(height: 16),
            Text(
              'Oops! Something went wrong',
              style: Theme.of(context).textTheme.titleLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              error.toString(),
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Theme.of(context).colorScheme.onSurface.withOpacity(0.6),
                  ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () => ref.invalidate(exampleProvider),
              icon: const Icon(Icons.refresh),
              label: const Text('Try Again'),
            ),
          ],
        ),
      ),
    );
  }

  // Event handlers

  Future<void> _handleRefresh() async {
    await ref.refresh(exampleProvider.future);
  }

  void _scrollToTop() {
    _scrollController.animateTo(
      0,
      duration: const Duration(milliseconds: 500),
      curve: Curves.easeOut,
    );
  }

  void _navigateToDetail(dynamic item) {
    Navigator.of(context).pushNamed(
      '/detail',
      arguments: item,
    );
  }

  void _navigateToSettings() {
    Navigator.of(context).pushNamed('/settings');
  }

  void _showAddDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Add Item'),
        content: TextField(
          decoration: const InputDecoration(
            labelText: 'Title',
            hintText: 'Enter title',
          ),
          onSubmitted: (value) {
            // Handle submission
            Navigator.of(context).pop();
          },
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              // Handle add
              Navigator.of(context).pop();
            },
            child: const Text('Add'),
          ),
        ],
      ),
    );
  }
}

// Placeholder provider (move to separate file)
final exampleProvider = FutureProvider<List<dynamic>>((ref) async {
  // Simulate API call
  await Future.delayed(const Duration(seconds: 1));
  return [];
});
