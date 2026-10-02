import type { NextRequest } from "next/server";
import { readTagFilter } from "@/lib/tag-filter-params";

/** The knowledge-base tag filter (tags, tags_match) of `request`, as a query string. */
export function tagFilterParams(request: NextRequest): string {
  const out = new URLSearchParams();
  // Same forwarding rules the other proxy routes get from `readTagFilter`: empty
  // tags dropped, an unknown tags_match not sent on to become a dataplane 422,
  // and a mode without tags forwarded only for `exact`.
  const { tags, tags_match } = readTagFilter(request.nextUrl.searchParams);
  tags?.forEach((tag) => out.append("tags", tag));
  if (tags_match) out.set("tags_match", tags_match);
  return out.toString();
}
