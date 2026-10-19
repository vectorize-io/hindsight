import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import postcss, { type AtRule, type Root, type Rule } from "postcss";
import tailwind from "@tailwindcss/postcss";
import { beforeAll, expect, it } from "vitest";

/**
 * The phone layout rests on a few global rules in globals.css that every
 * component inherits without opting in. They only work while they stay outside
 * Tailwind's @layer (unlayered CSS beats utilities like `opacity-0` and
 * `text-[13px]`), so a move into a layer would silently undo them.
 */
let compiled: Root;

beforeAll(async () => {
  const stylesheet = fileURLToPath(new URL("../../src/app/globals.css", import.meta.url));
  const base = fileURLToPath(new URL("../../", import.meta.url));
  const result = await postcss([tailwind({ base })]).process(await readFile(stylesheet, "utf8"), {
    from: stylesheet,
  });
  compiled = result.root;
});

function findRule(selector: string, media?: string): Rule | undefined {
  let found: Rule | undefined;
  compiled.walkRules((rule) => {
    if (rule.selector.replace(/\s+/g, " ") !== selector) return;
    const at = rule.parent?.type === "atrule" ? (rule.parent as AtRule) : undefined;
    if (media ? at?.name === "media" && at.params === media : !at) found = rule;
  });
  return found;
}

function inLayer(rule: Rule): boolean {
  let node: Rule | AtRule | Root | undefined = rule;
  while (node && node.type !== "root") {
    if (node.type === "atrule" && node.name === "layer") return true;
    node = node.parent as Rule | AtRule | Root | undefined;
  }
  return false;
}

function decl(rule: Rule | undefined, prop: string): string | undefined {
  let value: string | undefined;
  rule?.walkDecls(prop, (d) => {
    value = d.value;
  });
  return value;
}

it("never lets the page itself scroll sideways", () => {
  const rule = findRule("html, body");
  expect(decl(rule, "overflow-x")).toBe("clip");
});

it("shows reveal-on-hover controls on touch screens", () => {
  const rule = findRule('[class*="opacity-0"][class*="group-hover"]', "(hover: none)");
  expect(rule && !inLayer(rule)).toBe(true);
  expect(decl(rule, "opacity")).toBe("1");
});

it("keeps form fields at 16px on touch screens so iOS does not zoom", () => {
  const rule = findRule('input, textarea, select, [contenteditable="true"]', "(pointer: coarse)");
  expect(rule && !inLayer(rule)).toBe(true);
  expect(decl(rule, "font-size")).toBe("max(16px, 1em)");
});
