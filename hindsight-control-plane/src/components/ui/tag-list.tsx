// Tag rendering moved into the shared facet-chip module, which is the single
// place tags, entities and metadata are styled. Re-exported here so existing
// `@/components/ui/tag-list` imports keep working.
export { TagList } from "./facet-chip";
