/**
 * Search input changes are intentionally not part of DataView's auto-load
 * dependencies. Only clearing an active query needs an explicit reload.
 */
export function isSearchQueryCleared(previousQuery: string, nextQuery: string): boolean {
  return previousQuery.trim().length > 0 && nextQuery.trim().length === 0;
}
