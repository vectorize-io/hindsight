// The one tag-filter rule for the control plane: the browser client, the proxy routes and the
// views all go through these, so "is a filter set" and "what is forwarded" cannot drift apart.

export type TagsMatch = "any" | "all" | "any_strict" | "all_strict" | "exact";
const TAGS_MATCH_MODES = new Set<string>(["any", "all", "any_strict", "all_strict", "exact"]);

export type TagFilter = { tags?: string[]; tags_match?: TagsMatch };

/**
 * Whether `filter` narrows anything — the dataplane's `tag_filter_active`. Empty tags mean no
 * filter in every mode but `exact`, where they select the untagged scope.
 */
export function isTagFilterActive(filter: TagFilter): boolean {
  return !!filter.tags?.length || filter.tags_match === "exact";
}

/**
 * The forwardable form of `filter`: empty tags dropped, and `tags_match` only kept while the
 * filter is active — on its own it would override the dataplane default for an unfiltered read.
 */
function normalizeTagFilter(filter: TagFilter): TagFilter {
  const tags = (filter.tags ?? []).filter((tag) => tag.length > 0);
  const tags_match = filter.tags_match;
  const out: TagFilter = tags.length ? { tags } : {};
  return tags_match && isTagFilterActive({ tags, tags_match }) ? { ...out, tags_match } : out;
}

/**
 * Read `tags` (repeated) and `tags_match` from a request, normalized. An unknown mode is
 * dropped rather than sent on to become a 422.
 */
export function readTagFilter(searchParams: URLSearchParams): TagFilter {
  const mode = searchParams.get("tags_match");
  return normalizeTagFilter({
    tags: searchParams.getAll("tags"),
    // Set.has() does not narrow, so the cast is what carries the check into the type.
    tags_match: mode && TAGS_MATCH_MODES.has(mode) ? (mode as TagsMatch) : undefined,
  });
}

/** Append `filter`, normalized, to `params` as `tags=..&tags_match=..`. */
export function appendTagFilter(params: URLSearchParams, filter: TagFilter): void {
  const { tags, tags_match } = normalizeTagFilter(filter);
  tags?.forEach((tag) => params.append("tags", tag));
  if (tags_match) params.append("tags_match", tags_match);
}
