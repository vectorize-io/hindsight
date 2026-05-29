"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import { Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Directive {
  id: string;
  bank_id: string;
  name: string;
  content: string;
  is_active: boolean;
  priority: number;
  tags: string[];
  created_at: string;
}

interface DirectiveDetailModalProps {
  directiveId: string | null;
  onClose: () => void;
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

export function DirectiveDetailModal({ directiveId, onClose }: DirectiveDetailModalProps) {
  const t = useTranslations("directiveDetailModal");
  const { currentBank } = useBank();
  const [directive, setDirective] = useState<Directive | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!directiveId || !currentBank) return;

    const loadDirective = async () => {
      setLoading(true);
      setError(null);
      setDirective(null);

      try {
        const data = await client.getDirective(currentBank, directiveId);
        setDirective(data);
      } catch (err) {
        console.error("Error loading directive:", err);
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };

    loadDirective();
  }, [directiveId, currentBank]);

  const isOpen = directiveId !== null;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col p-6">
        <VisuallyHidden>
          <DialogTitle>{t("title")}</DialogTitle>
        </VisuallyHidden>
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center text-destructive">
              <div className="text-sm">{t("errorPrefix", { error })}</div>
            </div>
          </div>
        ) : directive ? (
          <div className="flex-1 overflow-y-auto space-y-6">
            {/* Header */}
            <div className="pb-5 border-b border-border">
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-bold text-foreground">{directive.name}</h3>
                {!directive.is_active && (
                  <span className="px-2 py-0.5 rounded-full bg-red-500/10 text-red-600 dark:text-red-400 text-xs font-medium">
                    {t("inactive")}
                  </span>
                )}
              </div>
              <code className="text-xs font-mono text-muted-foreground/70">{directive.id}</code>
            </div>

            {/* Created / Priority */}
            <div className="flex gap-8">
              <div>
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                  {t("created")}
                </div>
                <div className="text-sm text-foreground">
                  {formatDateTime(directive.created_at)}
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                  {t("priority")}
                </div>
                <div className="text-sm text-foreground">{directive.priority}</div>
              </div>
            </div>

            {/* Content */}
            <div>
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                {t("content")}
              </div>
              <div className="prose prose-base dark:prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{directive.content}</ReactMarkdown>
              </div>
            </div>

            {/* Tags */}
            {directive.tags && directive.tags.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                  {t("tags")}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {directive.tags.map((tag: string, idx: number) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 bg-purple-500/10 text-purple-600 dark:text-purple-400 rounded text-xs"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
