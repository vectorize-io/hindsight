// @vitest-environment jsdom
/**
 * A knowledge bank's Settings: General (stats, its ingest operations, the actions menu) is
 * the default tab, Configuration edits only this bank's own ingest settings — each save
 * sends just what changed, an emptied field clears that one override, the server's reason
 * for a refused value shows in the section — and nothing reads a server-wide config.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("next-intl", () => {
  const t = (key: string, values?: Record<string, unknown>) =>
    values ? `${key} ${JSON.stringify(values)}` : key;
  return { useTranslations: () => Object.assign(t, { rich: t }) };
});
vi.mock("sonner", () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));
// The knowledge bank page keeps the memory bank context empty: the operations list must
// take the bank from its prop.
vi.mock("@/lib/bank-context", () => ({ useBank: () => ({ currentBank: null }) }));
vi.mock("@/components/document-chunk-modal", () => ({ DocumentChunkModal: () => null }));

const api = vi.hoisted(() => ({
  listOperations: vi.fn(),
  getKnowledgeBankConfig: vi.fn(),
  updateKnowledgeBankConfig: vi.fn(),
  resetKnowledgeBankConfig: vi.fn(),
  deleteKnowledgeBank: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ client: api }));

const { KnowledgeBankSettings } = await import("@/components/knowledge-bank-settings");

beforeAll(() => {
  // Radix Select needs these, which jsdom lacks.
  Element.prototype.hasPointerCapture = () => false;
  Element.prototype.releasePointerCapture = () => {};
  Element.prototype.scrollIntoView = () => {};
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const bank = {
  bank_id: "docs",
  created_at: null,
  document_count: 3,
  passage_count: 12,
  ready_count: 3,
  pending_count: 0,
  failed_count: 0,
  total_bytes: 10,
  documents_by_media_type: { text: 3 },
};

const SERVER = {
  text_passage_tokens: 256,
  pdf_page_pixels: 1536,
  pdf_bands: false,
  image_tokens: 280,
  audio_window_seconds: 60,
  video_window_seconds: 30,
  media_overlap_seconds: 5,
  max_media_seconds: 7200,
  max_pdf_pages: 2000,
  disabled_media_types: [] as string[],
};

const configResponse = (overrides: Record<string, unknown> = {}, server = SERVER) => ({
  bank_id: "docs",
  config: { ...server, ...overrides },
  overrides,
});

const renderSettings = (tab: "general" | "configuration", onTabChange = vi.fn()) =>
  render(
    <KnowledgeBankSettings bank={bank} tab={tab} onTabChange={onTabChange} onDeleted={() => {}} />
  );

/** The ConfigSection whose heading is `title`. */
const section = (title: string) => screen.getByText(title).closest("section") as HTMLElement;

describe("KnowledgeBankSettings tabs", () => {
  it("opens on General: stats and this bank's operations, no config read", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    api.listOperations.mockResolvedValue({
      operations: [
        {
          id: "op-1",
          task_type: "knowledge_ingest",
          items_count: 0,
          document_id: null,
          created_at: "2026-11-13T06:02:25Z",
          status: "completed",
          error_message: null,
        },
      ],
      total: 1,
    });
    const onTabChange = vi.fn();
    renderSettings("general", onTabChange);

    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["tabGeneral", "tabConfiguration"]);
    expect(tabs[0].getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText("passages")).toBeTruthy();
    await waitFor(() => expect(api.listOperations).toHaveBeenCalled());
    expect(api.listOperations.mock.calls[0][0]).toBe("docs");
    expect(await screen.findByText("operationType.knowledgeIngest")).toBeTruthy();
    // One kind of operation: no type filter to pick from.
    expect(screen.queryByText("allTypes")).toBeNull();
    expect(api.getKnowledgeBankConfig).not.toHaveBeenCalled();
    expect(fetchSpy).not.toHaveBeenCalled();

    fireEvent.click(tabs[1]);
    expect(onTabChange).toHaveBeenCalledWith("configuration");
    fetchSpy.mockRestore();
  });

  it("shows only this bank's ingest settings on Configuration, grouped by type", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    api.getKnowledgeBankConfig.mockResolvedValue(configResponse({ image_tokens: 560 }));
    renderSettings("configuration");

    await screen.findByText("sections.text.title");
    expect(api.getKnowledgeBankConfig).toHaveBeenCalledWith("docs");
    expect(api.listOperations).not.toHaveBeenCalled();
    expect(fetchSpy).not.toHaveBeenCalled();
    for (const id of ["text", "pdf", "images", "media", "types"]) {
      expect(screen.getByText(`sections.${id}.title`)).toBeTruthy();
    }
    expect(screen.queryByText(/model|provider|dimension/i)).toBeNull();

    // Not overridden: empty, with the server's value as the placeholder.
    const tokens = screen.getByLabelText("fields.text_passage_tokens.label") as HTMLInputElement;
    expect(tokens.value).toBe("");
    expect(tokens.placeholder).toBe('serverDefaultValue {"value":256}');
    // Overridden: the bank's value, marked as such.
    const image = screen.getByLabelText("fields.image_tokens.label") as HTMLInputElement;
    expect(image.value).toBe("560");
    expect(within(section("sections.images.title")).getByText("overridden")).toBeTruthy();
    fetchSpy.mockRestore();
  });
});

describe("KnowledgeBankConfigView saving", () => {
  it("saves only the section's changed fields", async () => {
    api.getKnowledgeBankConfig.mockResolvedValue(configResponse());
    api.updateKnowledgeBankConfig.mockResolvedValue(configResponse({ text_passage_tokens: 512 }));
    renderSettings("configuration");
    await screen.findByText("sections.text.title");

    const text = section("sections.text.title");
    const save = within(text).getByRole("button", { name: "saveChanges" }) as HTMLButtonElement;
    expect(save.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("fields.text_passage_tokens.label"), {
      target: { value: "512" },
    });
    // An edit in another section is not sent with this one.
    fireEvent.change(screen.getByLabelText("fields.image_tokens.label"), {
      target: { value: "700" },
    });
    expect(save.disabled).toBe(false);
    fireEvent.click(save);

    await waitFor(() =>
      expect(api.updateKnowledgeBankConfig).toHaveBeenCalledWith("docs", {
        text_passage_tokens: 512,
      })
    );
    await waitFor(() => expect(within(text).getByText("overridden")).toBeTruthy());
    // The other section keeps its unsaved edit.
    expect((screen.getByLabelText("fields.image_tokens.label") as HTMLInputElement).value).toBe(
      "700"
    );
  });

  it("clears one override with null", async () => {
    api.getKnowledgeBankConfig.mockResolvedValue(
      configResponse({ max_media_seconds: 600, audio_window_seconds: 45 })
    );
    api.updateKnowledgeBankConfig.mockResolvedValue(configResponse({ audio_window_seconds: 45 }));
    renderSettings("configuration");
    await screen.findByText("sections.media.title");

    const media = section("sections.media.title");
    const input = screen.getByLabelText("fields.max_media_seconds.label") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "" } });
    fireEvent.click(within(media).getByRole("button", { name: "saveChanges" }));
    await waitFor(() =>
      expect(api.updateKnowledgeBankConfig).toHaveBeenCalledWith("docs", {
        max_media_seconds: null,
      })
    );
  });

  it("'Use server default' empties the field, and Save clears it", async () => {
    api.getKnowledgeBankConfig.mockResolvedValue(configResponse({ image_tokens: 560 }));
    api.updateKnowledgeBankConfig.mockResolvedValue(configResponse());
    renderSettings("configuration");
    await screen.findByText("sections.images.title");

    const images = section("sections.images.title");
    fireEvent.click(within(images).getByRole("button", { name: "useServerDefault" }));
    expect((screen.getByLabelText("fields.image_tokens.label") as HTMLInputElement).value).toBe("");
    fireEvent.click(within(images).getByRole("button", { name: "saveChanges" }));
    await waitFor(() =>
      expect(api.updateKnowledgeBankConfig).toHaveBeenCalledWith("docs", { image_tokens: null })
    );
  });

  it("shows the server's reason when it refuses a value", async () => {
    api.getKnowledgeBankConfig.mockResolvedValue(configResponse());
    api.updateKnowledgeBankConfig.mockRejectedValue(
      new Error("max_pdf_pages: may not exceed the server's limit of 2000")
    );
    renderSettings("configuration");
    await screen.findByText("sections.pdf.title");

    const pdf = section("sections.pdf.title");
    fireEvent.change(screen.getByLabelText("fields.max_pdf_pages.label"), {
      target: { value: "5000" },
    });
    fireEvent.click(within(pdf).getByRole("button", { name: "saveChanges" }));
    expect(
      await within(pdf).findByText("max_pdf_pages: may not exceed the server's limit of 2000")
    ).toBeTruthy();
    expect(api.updateKnowledgeBankConfig).toHaveBeenCalledWith("docs", { max_pdf_pages: 5000 });
  });

  it("sets PDF bands with the tri-state select", async () => {
    api.getKnowledgeBankConfig.mockResolvedValue(configResponse());
    api.updateKnowledgeBankConfig.mockResolvedValue(configResponse({ pdf_bands: true }));
    renderSettings("configuration");
    await screen.findByText("sections.pdf.title");

    fireEvent.keyDown(screen.getByRole("combobox", { name: "fields.pdf_bands.label" }), {
      key: "Enter",
    });
    fireEvent.click(await screen.findByRole("option", { name: "on" }));
    fireEvent.click(
      within(section("sections.pdf.title")).getByRole("button", { name: "saveChanges" })
    );
    await waitFor(() =>
      expect(api.updateKnowledgeBankConfig).toHaveBeenCalledWith("docs", { pdf_bands: true })
    );
  });

  it("turns a media type off for the bank; the server's own stay off and locked", async () => {
    api.getKnowledgeBankConfig.mockResolvedValue(
      configResponse({}, { ...SERVER, disabled_media_types: ["video"] })
    );
    api.updateKnowledgeBankConfig.mockResolvedValue(
      configResponse({ disabled_media_types: ["audio", "video"] })
    );
    renderSettings("configuration");
    await screen.findByText("sections.types.title");

    const video = screen.getByRole("switch", { name: 'accept {"kind":"video"}' });
    expect(video.getAttribute("aria-checked")).toBe("false");
    expect((video as HTMLButtonElement).disabled).toBe(true);
    const audio = screen.getByRole("switch", { name: 'accept {"kind":"audio"}' });
    expect(audio.getAttribute("aria-checked")).toBe("true");
    fireEvent.click(audio);
    fireEvent.click(
      within(section("sections.types.title")).getByRole("button", { name: "saveChanges" })
    );
    await waitFor(() =>
      expect(api.updateKnowledgeBankConfig).toHaveBeenCalledWith("docs", {
        disabled_media_types: ["audio"],
      })
    );
  });

  it("resets every override from the Actions menu and reloads", async () => {
    api.getKnowledgeBankConfig.mockResolvedValue(configResponse({ image_tokens: 560 }));
    api.resetKnowledgeBankConfig.mockResolvedValue(configResponse());
    renderSettings("configuration");
    await screen.findByText("sections.text.title");

    api.getKnowledgeBankConfig.mockResolvedValue(configResponse());
    fireEvent.pointerDown(screen.getByRole("button", { name: /actions/ }), {
      button: 0,
      ctrlKey: false,
    });
    fireEvent.click(await screen.findByRole("menuitem", { name: /resetConfig/ }));
    fireEvent.click(await screen.findByRole("button", { name: /resetConfig/ }));

    await waitFor(() => expect(api.resetKnowledgeBankConfig).toHaveBeenCalledWith("docs"));
    await waitFor(() => expect(api.getKnowledgeBankConfig).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText("overridden")).toBeNull());
  });
});
