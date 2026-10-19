// @vitest-environment jsdom
/**
 * The sidebar must not be one big button: a `role="button"` aside wrapping the nav
 * hides the links from screen readers (#5559). The bottom button is the only toggle.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@/lib/bank-context", () => ({ useBank: () => ({ currentBank: "alpha" }) }));
vi.mock("@/lib/api", () => ({
  client: { getVersion: () => Promise.resolve({ api_version: "1.0.0" }) },
}));

const { Sidebar } = await import("@/components/sidebar");

afterEach(() => cleanup());

describe("Sidebar accessibility", () => {
  it("exposes the nav links outside any button, with one labelled toggle", () => {
    render(<Sidebar currentTab="data" onTabChange={() => {}} />);

    const toggles = screen.getAllByRole("button");
    expect(toggles).toHaveLength(1);
    const toggle = toggles[0];
    expect(toggle.getAttribute("aria-label")).toBe("expandSidebar");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(toggle.getAttribute("aria-controls")).toBe("bank-sidebar-nav");

    const nav = screen.getByRole("navigation");
    expect(nav.id).toBe("bank-sidebar-nav");
    expect(nav.closest("[role=button], button")).toBeNull();

    // Collapsed links are icon-only but still named, and the active one is marked.
    const memories = screen.getByRole("link", { name: "memories" });
    expect(memories.getAttribute("aria-current")).toBe("page");
    expect(screen.getAllByRole("link")).toHaveLength(8);

    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(toggle.getAttribute("aria-label")).toBe("collapseSidebar");
  });
});
