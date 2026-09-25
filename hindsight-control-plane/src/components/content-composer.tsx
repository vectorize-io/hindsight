"use client";

import { useRef } from "react";

import { ChevronDown, ChevronUp, Paperclip, Plus, X } from "lucide-react";
import { useTranslations } from "next-intl";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Textarea } from "@/components/ui/textarea";
import type { RetainContentBlock } from "@/lib/api";
import { cn } from "@/lib/utils";

/** One element of content composed as an ordered block list. */
export type ComposerBlock =
  | { kind: "text"; text: string }
  | { kind: "attachment"; name: string; mediaType: string; data: string; size: number };

/**
 * How the content is being composed. "text" is the plain textarea most content
 * needs; "blocks" is the ordered list that lets an attachment sit *between* two
 * runs of prose, which is the whole point of inline attachments and cannot be
 * expressed with one textarea.
 */
export type ComposeMode = "text" | "blocks";

/** The blocks worth sending: every attachment, and text blocks that say something. */
export function nonEmptyBlocks(blocks: ComposerBlock[]): ComposerBlock[] {
  return blocks.filter((b) => (b.kind === "text" ? b.text.trim().length > 0 : true));
}

export function hasComposedContent(mode: ComposeMode, text: string, blocks: ComposerBlock[]) {
  return mode === "blocks" ? nonEmptyBlocks(blocks).length > 0 : text.trim().length > 0;
}

/**
 * The content as the API takes it: the plain string in text mode, else the blocks
 * in the order they were arranged, so an attachment reaches the extractor between
 * the sentences that frame it. An image block for images and a file block for
 * everything else — the distinction the API keeps because the providers keep it.
 */
export function toRetainContent(
  mode: ComposeMode,
  text: string,
  blocks: ComposerBlock[]
): string | RetainContentBlock[] {
  if (mode === "text") return text;
  return nonEmptyBlocks(blocks).map((b) =>
    b.kind === "text"
      ? { type: "text" as const, text: b.text }
      : b.mediaType.startsWith("image/")
        ? {
            type: "image" as const,
            source: { type: "base64" as const, media_type: b.mediaType, data: b.data },
          }
        : {
            type: "file" as const,
            filename: b.name,
            source: { type: "base64" as const, media_type: b.mediaType, data: b.data },
          }
  );
}

function readAttachment(file: File): Promise<ComposerBlock> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () =>
      resolve({
        kind: "attachment",
        name: file.name,
        // Some browsers report "" for unusual extensions; the API accepts any
        // well-formed type, so fall back to a generic one rather than refusing it.
        mediaType: file.type || "application/octet-stream",
        size: file.size,
        // readAsDataURL gives "data:<type>;base64,<payload>"; the API wants the payload alone.
        data: String(reader.result).split(",", 2)[1] ?? "",
      });
    reader.readAsDataURL(file);
  });
}

/**
 * A text/blocks switch over one piece of content, shared by the Add Document
 * dialog and the extraction tester so both compose content the same way.
 */
export function ContentComposer({
  label,
  mode,
  onModeChange,
  text,
  onTextChange,
  blocks,
  onBlocksChange,
  placeholder,
  textClassName,
  autoFocus,
}: {
  label: React.ReactNode;
  mode: ComposeMode;
  onModeChange: (mode: ComposeMode) => void;
  text: string;
  onTextChange: (text: string) => void;
  blocks: ComposerBlock[];
  onBlocksChange: (update: (current: ComposerBlock[]) => ComposerBlock[]) => void;
  placeholder?: string;
  textClassName?: string;
  autoFocus?: boolean;
}) {
  const t = useTranslations("addDocument");
  const fileInput = useRef<HTMLInputElement>(null);

  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        {label}
        {/* The switch, rather than always showing the block editor: most content is
            just text, and an ordered block list would be clutter in front of it. */}
        <div className="flex items-center overflow-hidden rounded border border-border text-xs">
          <button
            type="button"
            onClick={() => onModeChange("text")}
            className={cn(
              "px-2 py-1",
              mode === "text"
                ? "bg-muted font-semibold text-foreground"
                : "text-muted-foreground hover:bg-muted/50"
            )}
          >
            {t("composeText")}
          </button>
          <button
            type="button"
            onClick={() => {
              // Carry the typed text across so switching never silently loses it.
              onBlocksChange((current) =>
                current.length > 0 ? current : [{ kind: "text", text }]
              );
              onModeChange("blocks");
            }}
            className={cn(
              "border-l border-border px-2 py-1",
              mode === "blocks"
                ? "bg-muted font-semibold text-foreground"
                : "text-muted-foreground hover:bg-muted/50"
            )}
          >
            {t("composeBlocks")}
          </button>
        </div>
      </div>

      {mode === "text" ? (
        <Textarea
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          placeholder={placeholder ?? t("contentPlaceholder")}
          className={cn("min-h-[150px] resize-y", textClassName)}
          autoFocus={autoFocus}
        />
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">{t("composeBlocksHint")}</p>
          {blocks.map((block, index) => (
            <div
              key={index}
              className="group relative rounded border border-border bg-muted/20 p-2"
            >
              {/* Controls overlay the block rather than sitting in a column beside it:
                  at dialog width a fixed gutter squeezed the content into a sliver.
                  Revealed on hover and on keyboard focus, so they stay reachable
                  without a pointer. */}
              <div className="absolute right-1 top-1 flex items-center gap-0.5 rounded bg-background/80 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
                <button
                  type="button"
                  aria-label={t("blockMoveUp")}
                  disabled={index === 0}
                  onClick={() =>
                    onBlocksChange((current) => {
                      const next = [...current];
                      [next[index - 1], next[index]] = [next[index], next[index - 1]];
                      return next;
                    })
                  }
                  className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-25"
                >
                  <ChevronUp className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  aria-label={t("blockMoveDown")}
                  disabled={index === blocks.length - 1}
                  onClick={() =>
                    onBlocksChange((current) => {
                      const next = [...current];
                      [next[index], next[index + 1]] = [next[index + 1], next[index]];
                      return next;
                    })
                  }
                  className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-25"
                >
                  <ChevronDown className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  aria-label={t("blockRemove")}
                  onClick={() => onBlocksChange((current) => current.filter((_, i) => i !== index))}
                  className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>

              {block.kind === "text" ? (
                <Textarea
                  value={block.text}
                  onChange={(e) =>
                    onBlocksChange((current) =>
                      current.map((b, i) =>
                        i === index && b.kind === "text" ? { ...b, text: e.target.value } : b
                      )
                    )
                  }
                  placeholder={placeholder ?? t("contentPlaceholder")}
                  // Room for the overlay so the first line never runs underneath it.
                  className="min-h-[80px] resize-y border-0 bg-transparent p-0 pr-20 shadow-none focus-visible:ring-0"
                />
              ) : (
                <div className="flex items-center gap-2 pr-20 text-sm">
                  <Paperclip className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{block.name}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {(block.size / 1024).toFixed(0)} KB
                  </span>
                </div>
              )}
            </div>
          ))}

          <input
            ref={fileInput}
            type="file"
            multiple
            className="hidden"
            onChange={async (e) => {
              const encoded = await Promise.all(
                Array.from(e.target.files ?? []).map(readAttachment)
              );
              onBlocksChange((current) => [...current, ...encoded]);
              // Let the same file be picked again after removal.
              e.target.value = "";
            }}
          />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button type="button" variant="outline" size="sm">
                <Plus className="mr-1.5 h-3.5 w-3.5" />
                {t("blockAdd")}
                <ChevronDown className="ml-1.5 h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuItem
                onSelect={() =>
                  onBlocksChange((current) => [...current, { kind: "text", text: "" }])
                }
              >
                {t("blockAddText")}
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => fileInput.current?.click()}>
                {t("blockAddAttachment")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}
    </div>
  );
}
