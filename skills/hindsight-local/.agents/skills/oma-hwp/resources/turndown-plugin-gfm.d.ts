declare module "turndown-plugin-gfm" {
  import type TurndownService from "turndown";

  export function tables(turndownService: TurndownService): void;
  export function gfm(turndownService: TurndownService): void;
  export function highlightedCodeBlock(turndownService: TurndownService): void;
  export function strikethrough(turndownService: TurndownService): void;
  export function taskListItems(turndownService: TurndownService): void;
}
