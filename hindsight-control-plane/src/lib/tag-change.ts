interface TagChange {
  add_tags: string[];
  remove_tags: string[];
}

/**
 * The tags to add and remove to turn `current` into `next`. Sending only the
 * difference (not the whole list) keeps a tag someone else added meanwhile.
 */
export function tagChange(current: string[], next: string[]): TagChange {
  return {
    add_tags: next.filter((t) => !current.includes(t)),
    remove_tags: current.filter((t) => !next.includes(t)),
  };
}
