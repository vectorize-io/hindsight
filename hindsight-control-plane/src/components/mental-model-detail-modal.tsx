"use client";

import { useState, useEffect } from "react";
import { client, MentalModel } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { Loader2, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MentalModelDetailContentProps {
  mentalModel: MentalModel;
}

const formatDateTime = (dateStr: string) => {
  const date = new Date(dateStr);
  return `${date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  })} at ${date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  })}`;
};

/**
 * Shared content component for displaying mental model details.
 * Matches the layout of MentalModelDetailPanel for consistency.
 */
export function MentalModelDetailContent({ mentalModel }: MentalModelDetailContentProps) {
  return (
    <div className="space-y-6">
      {/* Header: Name, ID, Source Query */}
      <div className="pb-5 border-b border-border">
        <div className="flex items-center gap-2">
          <h3 className="text-xl font-bold text-foreground">{mentalModel.name}</h3>
          {mentalModel.trigger?.refresh_after_consolidation && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 text-xs font-medium">
              <Zap className="w-3 h-3" />
              Auto refresh
            </span>
          )}
        </div>
        <code className="text-xs font-mono text-muted-foreground/70">{mentalModel.id}</code>
        {mentalModel.source_query && (
          <p className="text-sm text-muted-foreground mt-1">{mentalModel.source_query}</p>
        )}
      </div>

      {/* Created / Last Refreshed */}
      <div className="flex gap-8">
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Created
          </div>
          <div className="text-sm text-foreground">{formatDateTime(mentalModel.created_at)}</div>
        </div>
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
            Last Refreshed
          </div>
          <div className="text-sm text-foreground">
            {formatDateTime(mentalModel.last_refreshed_at)}
          </div>
        </div>
      </div>

      {/* Content */}
      <div>
        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
          Content
        </div>
        <div className="prose prose-base dark:prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{mentalModel.content}</ReactMarkdown>
        </div>
      </div>

      {/* Tags */}
      {mentalModel.tags && mentalModel.tags.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Tags
          </div>
          <div className="flex flex-wrap gap-1.5">
            {mentalModel.tags.map((tag: string, idx: number) => (
              <span
                key={idx}
                className="px-2 py-0.5 bg-amber-500/10 text-amber-600 dark:text-amber-400 rounded text-xs"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface MentalModelDetailModalProps {
  mentalModelId: string | null;
  onClose: () => void;
}

/**
 * Modal wrapper for MentalModelDetailContent.
 * Fetches the mental model by ID and displays it in a dialog.
 */
export function MentalModelDetailModal({ mentalModelId, onClose }: MentalModelDetailModalProps) {
  const { currentBank } = useBank();
  const [mentalModel, setMentalModel] = useState<MentalModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!mentalModelId || !currentBank) return;

    const loadMentalModel = async () => {
      setLoading(true);
      setError(null);
      setMentalModel(null);

      try {
        const data = await client.getMentalModel(currentBank, mentalModelId);
        setMentalModel(data);
      } catch (err) {
        console.error("Error loading mental model:", err);
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };

    loadMentalModel();
  }, [mentalModelId, currentBank]);

  const isOpen = mentalModelId !== null;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col p-6">
        <VisuallyHidden>
          <DialogTitle>Mental Model Details</DialogTitle>
        </VisuallyHidden>
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center text-destructive">
              <div className="text-sm">Error: {error}</div>
            </div>
          </div>
        ) : mentalModel ? (
          <div className="flex-1 overflow-y-auto">
            <MentalModelDetailContent mentalModel={mentalModel} />
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
