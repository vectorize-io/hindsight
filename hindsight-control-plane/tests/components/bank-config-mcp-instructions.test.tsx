// @vitest-environment jsdom
/**
 * The Memory Bank section's MCP server instructions: edits only the bank's own override,
 * shows an inherited value as the placeholder rather than copying it onto the bank, and an
 * emptied field clears the override.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("next-intl", () => {
  const t = (key: string, values?: Record<string, unknown>) =>
    values ? `${key} ${JSON.stringify(values)}` : key;
  return { useTranslations: () => Object.assign(t, { rich: t }) };
});
vi.mock("sonner", () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));
vi.mock("@/lib/bank-context", () => ({ useBank: () => ({ currentBank: "agents" }) }));
vi.mock("@/lib/features-context", () => ({ useFeatures: () => ({ features: {} }) }));

const api = vi.hoisted(() => ({
  getBankConfig: vi.fn(),
  updateBankConfig: vi.fn(),
  listBankAliases: vi.fn(),
  previewConsolidationStrategies: vi.fn(() => new Promise(() => {})),
  listTags: vi.fn(() => Promise.resolve({ items: [] })),
}));
vi.mock("@/lib/api", () => ({ client: api }));

const { BankConfigView } = await import("@/components/bank-config-view");

beforeAll(() => {
  Element.prototype.hasPointerCapture = () => false;
  Element.prototype.releasePointerCapture = () => {};
  Element.prototype.scrollIntoView = () => {};
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const configResponse = (config: Record<string, unknown>, overrides: Record<string, unknown>) => ({
  bank_id: "agents",
  config: { mcp_enabled_tools: null, ...config },
  overrides,
});

/** The instructions textarea, found by the description rendered with it. */
const instructionsBox = async () => {
  const description = await screen.findByText("mcpServerInstructionsDescription");
  return description.closest("div.space-y-2")!.querySelector("textarea") as HTMLTextAreaElement;
};

/** The save button of the section that holds the instructions field. */
const saveButton = () =>
  within(
    screen.getByText("mcpServerInstructionsDescription").closest("section") as HTMLElement
  ).getByText("saveChanges");

const renderView = (config: Record<string, unknown>, overrides: Record<string, unknown>) => {
  api.getBankConfig.mockResolvedValue(configResponse(config, overrides));
  api.listBankAliases.mockResolvedValue({ aliases: [] });
  render(<BankConfigView />);
};

describe("BankConfigView MCP server instructions", () => {
  it("shows the bank's own instructions and saves an edit", async () => {
    const text = "Read the document 'house-rules' first.";
    renderView({ mcp_server_instructions: text }, { mcp_server_instructions: text });

    const box = await instructionsBox();
    expect(box.value).toBe(text);
    expect(box.maxLength).toBe(4000);

    api.updateBankConfig.mockResolvedValue(
      configResponse({ mcp_server_instructions: "Tag new facts with 'inbox'." }, {})
    );
    fireEvent.change(box, { target: { value: "Tag new facts with 'inbox'." } });
    fireEvent.click(saveButton());

    await waitFor(() => expect(api.updateBankConfig).toHaveBeenCalled());
    expect(api.updateBankConfig.mock.calls[0][0]).toBe("agents");
    expect(api.updateBankConfig.mock.calls[0][1]).toMatchObject({
      mcp_server_instructions: "Tag new facts with 'inbox'.",
    });
  });

  it("shows an inherited value as the placeholder and does not save it onto the bank", async () => {
    const inherited = "Organization-wide instructions.";
    renderView({ mcp_server_instructions: inherited }, {});

    const box = await instructionsBox();
    expect(box.value).toBe("");
    expect(box.placeholder).toBe(inherited);

    // Saving another field in the section must not copy the inherited text onto the bank.
    api.updateBankConfig.mockResolvedValue(configResponse({ default_timezone: "Europe/Rome" }, {}));
    fireEvent.change(screen.getByPlaceholderText("UTC"), { target: { value: "Europe/Rome" } });
    fireEvent.click(saveButton());

    await waitFor(() => expect(api.updateBankConfig).toHaveBeenCalled());
    expect(api.updateBankConfig.mock.calls[0][1]).toMatchObject({ mcp_server_instructions: null });
  });

  it("clears the override when the field is emptied", async () => {
    renderView({ mcp_server_instructions: "Old." }, { mcp_server_instructions: "Old." });

    const box = await instructionsBox();
    api.updateBankConfig.mockResolvedValue(configResponse({}, {}));
    fireEvent.change(box, { target: { value: "" } });
    fireEvent.click(saveButton());

    await waitFor(() => expect(api.updateBankConfig).toHaveBeenCalled());
    expect(api.updateBankConfig.mock.calls[0][1]).toMatchObject({ mcp_server_instructions: null });
  });
});
