// @vitest-environment jsdom
/**
 * An empty fact-types selection means "no filter" (saved as fact_types: null),
 * but empty checkboxes read as "none". The editor must say "all" inline.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

import {
  MentalModelTriggerFields,
  triggerFormFromTrigger,
} from "@/components/mental-model-trigger-fields";

afterEach(cleanup);

describe("MentalModelTriggerFields fact types", () => {
  it("says all fact types when none are selected", () => {
    render(<MentalModelTriggerFields value={triggerFormFromTrigger()} onChange={() => {}} />);
    expect(screen.getByText("triggerFactTypesAll")).toBeTruthy();
  });

  it("hides the note once a fact type is selected", () => {
    render(
      <MentalModelTriggerFields
        value={triggerFormFromTrigger({ fact_types: ["observation"] })}
        onChange={() => {}}
      />
    );
    expect(screen.queryByText("triggerFactTypesAll")).toBeNull();
  });
});
