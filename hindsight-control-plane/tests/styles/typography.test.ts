import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import postcss, { type Root } from "postcss";
import tailwind from "@tailwindcss/postcss";
import { beforeAll, expect, it } from "vitest";

let compiled: Root;

beforeAll(async () => {
  const stylesheet = fileURLToPath(new URL("../../src/app/globals.css", import.meta.url));
  const base = fileURLToPath(new URL("../../", import.meta.url));
  const result = await postcss([tailwind({ base })]).process(await readFile(stylesheet, "utf8"), {
    from: stylesheet,
  });
  compiled = result.root;
});

it("generates paragraph, heading and list styles for rendered Markdown", () => {
  // Check the compiled output: registering typography only in an unloaded
  // Tailwind v3 config leaves the prose class names present but inert in v4.
  const declarations = new Map<string, Set<string>>();
  compiled.walkRules((rule) => {
    for (const element of ["p", "h2", "ol", "ul"]) {
      if (!rule.selector.includes(`:where(${element})`)) continue;
      rule.walkDecls((declaration) => {
        const properties = declarations.get(element) ?? new Set<string>();
        properties.add(declaration.prop);
        declarations.set(element, properties);
      });
    }
  });
  expect(declarations.get("p")).toContain("margin-top");
  expect(declarations.get("h2")).toContain("font-size");
  expect(declarations.get("ol")).toContain("list-style-type");
  expect(declarations.get("ul")).toContain("list-style-type");
});

it("wraps long answer text without reducing table cells' intrinsic minimum widths", () => {
  // Unlike anywhere, break-word leaves min-content sizing intact. Chromium
  // verifies that a six-column table scrolls while a long paragraph URL wraps.
  const wrapping: string[] = [];
  compiled.walkRules(".reflect-answer", (rule) => {
    rule.walkDecls("overflow-wrap", (declaration) => {
      wrapping.push(declaration.value);
    });
  });
  expect(wrapping).toEqual(["break-word"]);
});
