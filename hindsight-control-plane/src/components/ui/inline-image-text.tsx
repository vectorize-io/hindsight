"use client";

/**
 * Render retained text that may carry inline-image placeholders.
 *
 * A document retained with inline images stores its text with an atomic
 * placeholder — `⟦hs-img:<id>⟧` — where each image sat. That token is
 * the real stored content, not a rendering artifact, but showing it raw would be
 * meaningless to a reader: the whole point of retaining images inline is that
 * they belong in position, next to the prose that refers to them.
 *
 * So the text is split on the placeholders and each one becomes the picture it
 * stands for, in place. Everything else renders exactly as it did.
 */

import { useState } from "react";

/** Matches an image placeholder and captures its short image id. */
const PLACEHOLDER_RE = /⟦hs-img:([0-9a-f]{12})⟧/g;

export function hasInlineImage(text: string): boolean {
  // `test` on a /g regex advances lastIndex; build a fresh matcher each call.
  return new RegExp(PLACEHOLDER_RE.source).test(text);
}

function InlineImage({ bankId, imageId }: { bankId: string; imageId: string }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    // The bytes can legitimately be gone. Say so where the image belonged
    // rather than leaving a silent gap the reader cannot interpret.
    return (
      <span className="inline-block my-1 px-2 py-1 rounded border border-dashed border-border text-[10px] text-muted-foreground">
        image unavailable ({imageId})
      </span>
    );
  }

  // A plain <img> rather than next/image: the bytes are proxied from the
  // dataplane at request time behind server-side auth, so there is no URL for
  // the image optimizer to pre-resolve.
  return (
    <img
      src={`/api/images?bank_id=${encodeURIComponent(bankId)}&id=${imageId}`}
      alt="Retained inline image"
      onError={() => setFailed(true)}
      className="block my-2 max-h-64 max-w-full rounded border border-border object-contain"
    />
  );
}

export function InlineImageText({
  text,
  bankId,
  className,
}: {
  text: string;
  bankId: string;
  className?: string;
}) {
  const nodes: React.ReactNode[] = [];
  const matcher = new RegExp(PLACEHOLDER_RE.source, "g");
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = matcher.exec(text)) !== null) {
    if (match.index > cursor) {
      nodes.push(text.slice(cursor, match.index));
    }
    nodes.push(
      <InlineImage key={`${match[1]}-${match.index}`} bankId={bankId} imageId={match[1]} />
    );
    cursor = match.index + match[0].length;
  }
  nodes.push(text.slice(cursor));

  return <div className={className}>{nodes}</div>;
}
