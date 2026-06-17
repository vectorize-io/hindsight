"use client";

import { useEffect, useState } from "react";

import { useTranslations } from "next-intl";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useBank } from "@/lib/bank-context";
import { Loader2 } from "lucide-react";
import JsonView from "react18-json-view";
import "react18-json-view/src/style.css";

/**
 * Dry-run extraction in a dialog (opened from the Memory Bank actions): preview what this bank's
 * configuration extracts from text — no ingestion, nothing stored. Input on the left, JSON on the right.
 */
export function ExtractDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useTranslations("bank");
  const tCommon = useTranslations("common");
  const { currentBank } = useBank();
  const [content, setContent] = useState("");
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setResult(null);
      setError(null);
    }
  }, [open]);

  async function run() {
    if (!currentBank) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bank_id: currentBank, content }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("dryRunExtraction")}</DialogTitle>
          <DialogDescription>{t("dryRunExtractionDescription")}</DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Input */}
          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                {t("dryRunExtractionTextLabel")}
              </label>
              <Textarea
                className="h-[30rem]"
                placeholder={t("dryRunExtractionPlaceholder")}
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
            </div>
            {error ? <p className="text-sm text-destructive">{error}</p> : null}
          </div>

          {/* Output */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              {t("dryRunExtractionResponseLabel")}
            </label>
            <div className="h-[30rem] overflow-auto rounded-md border border-border bg-muted/30 p-3 text-sm">
              {result ? (
                <JsonView src={result as object} collapsed={2} theme="default" />
              ) : (
                <span className="text-muted-foreground">{t("dryRunExtractionEmptyState")}</span>
              )}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" disabled={loading} onClick={() => onOpenChange(false)}>
            {tCommon("close")}
          </Button>
          <Button onClick={run} disabled={loading || !content.trim() || !currentBank}>
            {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
            {t("dryRunExtractionRun")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
