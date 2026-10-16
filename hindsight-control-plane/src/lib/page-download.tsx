/** Download a knowledge page or mental model, all in the browser. */
import { CompactMarkdown } from "@/components/compact-markdown";
import type { MentalModel } from "@/lib/api";
import { formatAbsoluteDateTime } from "@/lib/relative-time";
import { withBasePath } from "@/lib/base-path";

interface DownloadablePage {
  name: string;
  body: string;
  updatedAt?: string | null;
  tags?: string[];
}

export function mentalModelDownload(m: MentalModel): DownloadablePage {
  return { name: m.name, body: m.content, updatedAt: m.last_refreshed_at, tags: m.tags };
}

function filename(name: string, ext: string): string {
  return `${name.replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "") || "page"}.${ext}`;
}

export function downloadMarkdown({ name, body }: DownloadablePage) {
  const url = URL.createObjectURL(
    new Blob([`# ${name}\n\n${body}\n`], { type: "text/markdown;charset=utf-8" })
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename(name, "md");
  a.click();
  URL.revokeObjectURL(url);
}

// Margin boxes print the brand and page numbers on every sheet (Chromium; others
// ignore them). print-color-adjust keeps the gradient bar and tag chips.
const PRINT_CSS = `
@page { margin: 16mm 16mm 18mm;
  @bottom-left { content: "Hindsight"; font: 600 9px system-ui, sans-serif; color: #0074d9; }
  @bottom-right { content: counter(page) " / " counter(pages); font: 9px system-ui, sans-serif; color: #888; } }
html, body { background: #fff !important; }
body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
@media screen { body { padding: 32px 16px; } }
`;

/** Render the page into a bare window with the app's stylesheets and open the
 *  browser's print dialog ("Save as PDF" makes the file). The window never gets
 *  the `dark` class, so it prints in light colors. */
export async function printPage({ name, body, updatedAt, tags = [] }: DownloadablePage) {
  // Open before any await: a window opened after one is blocked as a popup.
  const w = window.open("", "_blank");
  if (!w) return;
  const { renderToStaticMarkup } = await import("react-dom/server");
  const html = renderToStaticMarkup(
    <div className="max-w-3xl mx-auto">
      <header className="flex items-center justify-between">
        <img src={withBasePath("/logo.png")} alt="Hindsight" className="h-9" />
        {updatedAt && (
          <span className="text-xs text-gray-500">{formatAbsoluteDateTime(updatedAt)}</span>
        )}
      </header>
      <div className="bg-primary-gradient h-1 rounded-full mt-3" />
      <h1 className="text-3xl font-bold tracking-tight text-gray-900 mt-10">{name}</h1>
      {tags.length > 0 && (
        <div className="flex gap-1.5 flex-wrap mt-3">
          {tags.map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-100"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
      <div className="prose prose-sm max-w-none mt-8">
        <CompactMarkdown>{body}</CompactMarkdown>
      </div>
    </div>
  );
  const styles = Array.from(document.querySelectorAll('link[rel="stylesheet"], style'))
    .map((n) => n.outerHTML)
    .join("");
  w.document.write(
    `<!doctype html><html><head><base href="${location.origin}/"><title></title>${styles}<style>${PRINT_CSS}</style></head><body>${html}</body></html>`
  );
  // Set after writing, so the name never has to be HTML-escaped into the markup.
  w.document.title = name;
  // Fonts are set as classes on <html>/<body>; copy them, minus the theme.
  w.document.documentElement.className = document.documentElement.className;
  w.document.documentElement.classList.remove("dark");
  w.document.body.className = document.body.className;
  w.document.close();
  w.onafterprint = () => w.close();
  w.onload = () => {
    w.focus();
    w.print();
  };
}
