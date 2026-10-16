// @vitest-environment jsdom
/**
 * Downloading a knowledge page or mental model (#5457): Markdown is a file built
 * from the page; PDF is the page rendered into a bare window and handed to the
 * browser's print dialog.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { downloadMarkdown, printPage } from "@/lib/page-download";

afterEach(() => {
  vi.restoreAllMocks();
  document.documentElement.className = "";
  document.body.innerHTML = "";
});

describe("downloadMarkdown", () => {
  it("saves the title as a heading above the body, named after the page", async () => {
    let blob: Blob | undefined;
    URL.createObjectURL = vi.fn((b: Blob) => {
      blob = b;
      return "blob:x";
    });
    URL.revokeObjectURL = vi.fn();
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement
    ) {
      expect(this.download).toBe("Résumé-v2.md");
    });

    downloadMarkdown({ name: "Résumé: v2!", body: "- one\n- two" });

    expect(click).toHaveBeenCalledOnce();
    // jsdom's Blob has no .text(); FileReader is what it implements.
    const text = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsText(blob!);
    });
    expect(text).toBe("# Résumé: v2!\n\n- one\n- two\n");
  });
});

describe("printPage", () => {
  it("renders the branded page in light colors and prints it", async () => {
    document.documentElement.className = "dark font-sans";
    const frame = document.createElement("iframe");
    document.body.append(frame);
    const w = frame.contentWindow!;
    w.print = vi.fn();
    w.focus = vi.fn();
    vi.spyOn(window, "open").mockReturnValue(w);

    await printPage({
      name: "Release process",
      body: "## Schedule\n\n- every second Tuesday",
      updatedAt: "2026-10-16T12:52:00Z",
      tags: ["team"],
    });
    w.onload!(new Event("load"));

    const doc = w.document;
    expect(doc.title).toBe("Release process");
    expect(doc.documentElement.className).toBe("font-sans");
    expect(doc.querySelector("img")!.getAttribute("src")).toBe("/logo.png");
    expect(doc.querySelector("h1")!.textContent).toBe("Release process");
    expect(doc.querySelector("li")!.textContent).toBe("every second Tuesday");
    expect(doc.body.textContent).toContain("team");
    expect(doc.body.textContent).toContain("Oct 16, 2026");
    expect(w.print).toHaveBeenCalledOnce();
  });

  it("does nothing when the browser blocks the window", async () => {
    vi.spyOn(window, "open").mockReturnValue(null);
    await expect(printPage({ name: "x", body: "y" })).resolves.toBeUndefined();
  });
});
