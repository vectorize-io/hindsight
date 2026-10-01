// Tag filter query params shared by the proxy routes that forward `tags` / `tags_match`.

type TagsMatch = "any" | "all" | "any_strict" | "all_strict" | "exact";
const TAGS_MATCH_MODES = new Set<string>(["any", "all", "any_strict", "all_strict", "exact"]);

type TagFilter = { tags?: string[]; tags_match?: TagsMatch };

/**
 * Read `tags` (repeated) and `tags_match` from a request. `tags_match` is only
 * forwarded alongside tags — on its own it would override the dataplane default
 * for an unfiltered read — and an unknown mode is dropped rather than sent on to
 * become a 422.
 */
export function readTagFilter(searchParams: URLSearchParams): TagFilter {
  const tags = searchParams.getAll("tags").filter((tag) => tag.length > 0);
  if (tags.length === 0) return {};
  const mode = searchParams.get("tags_match");
  // Set.has() does not narrow, so the cast is what carries the check into the type.
  return mode && TAGS_MATCH_MODES.has(mode) ? { tags, tags_match: mode as TagsMatch } : { tags };
}
