"use client";

// A knowledge bank as the memory banks' constellation: each document is a hub with its
// passages around it, one cluster per document, coloured by kind.

import * as React from "react";
import { useTranslations } from "next-intl";

import { client } from "@/lib/api";
import {
  documentLabel,
  formatWindow,
  type KnowledgeDocument,
  type KnowledgeDocumentDetail,
  type KnowledgePassage,
  type MediaType,
} from "@/lib/knowledge-banks";
import { Constellation, type HudStats } from "@/components/constellation";
import type { GraphData, GraphNode } from "@/components/graph-data";
import { KIND_HEX } from "@/components/knowledge-passage";
import { Spinner } from "@/components/ui/spinner";

// ponytail: one detail request per document on screen; a passages-of-many-documents
// endpoint would make this one call.
/** Passages drawn per document, so one long PDF does not swamp the others. */
const MAX_PASSAGES = 40;
const LABEL_CHARS = 40;

/** The graph for `details`: a hub per document, its passages linked to it. Exported for tests. */
export function buildKnowledgeGraph(
  details: KnowledgeDocumentDetail[],
  labels: { kind: (kind: MediaType) => string; passage: (p: KnowledgePassage) => string }
): GraphData {
  const data: GraphData = { nodes: [], links: [] };
  for (const doc of details) {
    const hub = `doc:${doc.document_id}`;
    const label = documentLabel(doc);
    const kind = labels.kind(doc.media_type);
    data.nodes.push({
      id: hub,
      label,
      group: doc.document_id,
      // `fact_type` is the constellation tooltip's badge.
      metadata: { document_id: doc.document_id, fact_type: kind, text: label, hub: true },
    });
    for (const p of doc.passages.slice(0, MAX_PASSAGES)) {
      const id = `${hub}#${p.index}`;
      const text = p.text?.trim() || labels.passage(p);
      data.nodes.push({
        id,
        label: text.length > LABEL_CHARS ? `${text.slice(0, LABEL_CHARS)}…` : text,
        group: doc.document_id,
        metadata: { document_id: doc.document_id, fact_type: kind, passage: p.index, text },
      });
      data.links.push({ source: id, target: hub, type: "passage" });
    }
  }
  return data;
}

export function KnowledgeConstellation({
  bankId,
  docs,
  height,
  onOpenDocument,
}: {
  bankId: string;
  /** The documents to draw; those not ready (or empty) are skipped. */
  docs: KnowledgeDocument[];
  height: number;
  onOpenDocument: (documentId: string, passage?: number) => void;
}) {
  const t = useTranslations("knowledgeBanks");
  const [details, setDetails] = React.useState<KnowledgeDocumentDetail[] | null>(null);

  const drawable = docs.filter((d) => d.status === "ready" && d.passage_count > 0);
  // Refetch only when the set of documents (or their content) changes, not on every poll.
  const key = JSON.stringify(drawable.map((d) => [d.document_id, d.updated_at]));
  React.useEffect(() => {
    let live = true;
    const ids = (JSON.parse(key) as [string, string | null][]).map(([id]) => id);
    Promise.all(ids.map((id) => client.getKnowledgeDocument(bankId, id).catch(() => null))).then(
      (res) => live && setDetails(res.filter((d): d is KnowledgeDocumentDetail => d !== null))
    );
    return () => {
      live = false;
    };
  }, [bankId, key]);

  const data = React.useMemo(
    () =>
      buildKnowledgeGraph(details ?? [], {
        kind: (kind) => t(`kinds.${kind}`),
        passage: (p) =>
          formatWindow(p) ??
          (p.page != null
            ? t("passage.page", { page: p.page })
            : t("passage.number", { number: p.index + 1 })),
      }),
    [details, t]
  );
  const kindOf = React.useMemo(() => {
    const m = new Map<string, string>();
    for (const d of details ?? []) m.set(d.document_id, KIND_HEX[d.media_type]);
    return m;
  }, [details]);
  const labelOf = React.useMemo(() => {
    const m = new Map<string, string>();
    for (const d of details ?? []) m.set(d.document_id, documentLabel(d));
    return m;
  }, [details]);

  // The footer counts what the dots are here: documents (the hubs) and their passages.
  const documentCount = data.nodes.filter((n) => n.metadata?.hub).length;
  const hudStats = React.useCallback(
    (s: HudStats) =>
      t("constellation.hudStats", {
        ...s,
        documents: documentCount,
        passages: s.nodes - documentCount,
      }),
    [t, documentCount]
  );

  if (details === null) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <Spinner size="lg" variant="jump" />
      </div>
    );
  }
  if (data.nodes.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-sm text-muted-foreground"
        style={{ height }}
      >
        {t("constellation.empty")}
      </div>
    );
  }
  return (
    <Constellation
      data={data}
      height={height}
      onNodeClick={(node: GraphNode) => {
        const meta = node.metadata as { document_id: string; passage?: number };
        onOpenDocument(meta.document_id, meta.passage);
      }}
      nodeSizeFn={(node) => (node.metadata?.hub ? 11 : 4)}
      clusterKeyFn={(node) => node.group ?? null}
      clusterColorFn={(key) => kindOf.get(key) ?? KIND_HEX.document}
      clusterLabelFn={(key) => labelOf.get(key) ?? key}
      linkColorFn={(link) => kindOf.get(link.target.slice(4)) ?? KIND_HEX.document}
      hudStatsFn={hudStats}
    />
  );
}
