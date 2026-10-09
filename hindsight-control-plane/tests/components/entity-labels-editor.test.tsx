// @vitest-environment jsdom
/**
 * The Entity Labels editor's "+field" button.
 *
 * Each top-level label is a single field: the label-level onChange keeps only the
 * first entry, so a "+field" at the root appended a blank field that was dropped at
 * once — a dead button. It now shows only inside a Map's nested editor, where it works.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

import { EntityLabelsEditor, type LabelGroup } from "@/components/entity-labels-editor";

afterEach(cleanup);

function label(patch: Partial<LabelGroup>): LabelGroup {
  return {
    key: "topic",
    description: "",
    type: "text",
    optional: true,
    tag: false,
    values: [],
    fields: {},
    ...patch,
  };
}

describe("EntityLabelsEditor +field", () => {
  it("is not offered on a top-level label", () => {
    render(<EntityLabelsEditor value={[label({})]} onChange={vi.fn()} />);
    expect(screen.queryByText("addField")).toBeNull();
  });

  it("adds a sub-field inside a Map label", () => {
    const onChange = vi.fn();
    render(
      <EntityLabelsEditor
        value={[label({ type: "map", fields: { city: { type: "text", description: "" } } })]}
        onChange={onChange}
      />
    );
    fireEvent.click(screen.getByText("addField"));
    expect(Object.keys(onChange.mock.calls[0][0][0].fields)).toEqual(["city", ""]);
  });
});
