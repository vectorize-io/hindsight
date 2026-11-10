// @vitest-environment jsdom
/**
 * The sidebar must not be one big button: a `role="button"` aside wrapping the nav
 * hides the links from screen readers (#5559). The bottom button is the only toggle.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Home } from "lucide-react";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@/lib/bank-context", () => ({ useBank: () => ({ currentBank: "alpha" }) }));
vi.mock("@/lib/api", () => ({
  client: { getVersion: () => Promise.resolve({ api_version: "1.0.0" }) },
}));

const { Sidebar, SidebarMenuButton } = await import("@/components/sidebar");

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

  it("opens the phone drawer from the header menu button and closes it on navigation", () => {
    const onTabChange = vi.fn();
    render(
      <>
        <SidebarMenuButton />
        <Sidebar currentTab="data" onTabChange={onTabChange} />
      </>
    );
    // Closed: only the rail's links exist.
    expect(screen.getAllByRole("link")).toHaveLength(8);

    fireEvent.click(screen.getByRole("button", { name: "openNavigation" }));
    const drawer = screen.getByRole("dialog");
    expect(drawer.querySelectorAll("a")).toHaveLength(8);

    const recall = Array.from(drawer.querySelectorAll("a")).find(
      (a) => a.getAttribute("aria-label") === "recall"
    )!;
    fireEvent.click(recall);
    expect(onTabChange).toHaveBeenCalledWith("recall");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("renders another bank kind's sections with their own links", () => {
    const onTabChange = vi.fn();
    render(
      <Sidebar
        currentTab="search"
        onTabChange={onTabChange}
        items={[
          { id: "home", label: "Home", icon: Home },
          { id: "search", label: "Search", icon: Home },
        ]}
        hrefFor={(id) => `/knowledge-banks/kb?view=${id}`}
      />
    );
    const links = screen.getAllByRole("link");
    expect(links.map((a) => a.getAttribute("href"))).toEqual([
      "/knowledge-banks/kb?view=home",
      "/knowledge-banks/kb?view=search",
    ]);
    expect(links[1].getAttribute("aria-current")).toBe("page");
    fireEvent.click(links[0]);
    expect(onTabChange).toHaveBeenCalledWith("home");
  });
});
