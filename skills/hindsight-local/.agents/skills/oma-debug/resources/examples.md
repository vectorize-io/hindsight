# Debug Agent - Examples

## Example 1: Null Reference Error

**Input**: "TodoList crashes with 'Cannot read property map of undefined'"

**Output**:
```
## Bug Analysis

**Root Cause**: TodoList renders before API data loads. `todos` is undefined
on first render, and `.map()` is called immediately.

**Why**: useEffect fetches async, but component doesn't guard against
undefined initial state.

## Fix
- Before: `todos.map(t => <TodoItem ... />)`
- After: `todos?.map(t => <TodoItem ... />) ?? <EmptyState />`
- Also added loading spinner while fetching

## Changes Made
1. Added `if (isLoading) return <Spinner />`
2. Added optional chaining on `todos?.map()`
3. Added empty state component

## Files Modified
- src/components/TodoList.tsx (guard + loading state)
- src/components/EmptyState.tsx (NEW)
- tests/TodoList.test.tsx (regression test)

## Prevention
Use TanStack Query which provides `isLoading`/`data` pattern by default.
```

## Example 2: N+1 Query Performance Bug

**Input**: "The posts page takes 3 seconds to load with only 50 posts"

**Output**:
```
## Bug Analysis

**Root Cause**: N+1 query. For each post, a separate SQL query fetches
the author. 50 posts = 51 queries.

**Why**: SQLAlchemy lazy-loads relationships by default.

## Fix
- Before: `db.query(Post).all()`
- After: `db.query(Post).options(joinedload(Post.author)).all()`

## Changes Made
1. Added eager loading with joinedload
2. Load time: 3000ms -> 45ms

## Files Modified
- app/api/posts.py (eager loading)
- tests/test_posts.py (performance assertion: < 200ms)
```

## Example 3: Race Condition

**Input**: "Search results show wrong data when typing fast"

**Output**:
```
## Bug Analysis

**Root Cause**: Race condition. Fast typing triggers multiple API calls.
Slower early requests resolve after faster recent ones, overwriting
correct results.

**Why**: No request cancellation; last-write-wins causes stale data.

## Fix
Added AbortController to cancel previous request on new keystroke.

## Changes Made
1. useEffect cleanup cancels pending request
2. Added debounce (300ms) to reduce API calls
3. Added `cancelled` flag to ignore stale responses

## Files Modified
- src/hooks/useSearch.ts (abort + debounce)
- tests/useSearch.test.ts (race condition test)
```
