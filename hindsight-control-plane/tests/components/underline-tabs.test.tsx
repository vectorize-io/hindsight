// @vitest-environment jsdom
/**
 * The shared sub-tab row: one place decides how page tabs look and behave on a
 * phone (they scroll inside their own row instead of widening the page).
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { UnderlineTabs } from "@/components/ui/underline-tabs";

afterEach(() => cleanup());

describe("UnderlineTabs", () => {
  it("marks the active tab, flags disabled features and reports clicks", () => {
    const onChange = vi.fn();
    render(
      <UnderlineTabs
        value="b"
        onChange={onChange}
        tabs={[
          { id: "a", label: "Alpha" },
          { id: "b", label: "Beta" },
          { id: "c", label: "Gamma", off: true },
        ]}
      />
    );

    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.getAttribute("aria-selected"))).toEqual(["false", "true", "false"]);
    expect(tabs[2].textContent).toContain("Off");
    // The row scrolls by itself so the page never does.
    expect(screen.getByRole("tablist").className).toContain("overflow-x-auto");

    fireEvent.click(tabs[0]);
    expect(onChange).toHaveBeenCalledWith("a");
  });
});
