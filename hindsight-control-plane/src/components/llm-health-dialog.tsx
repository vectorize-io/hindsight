"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { Loader2, RefreshCw } from "lucide-react";
import { client } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";

// Small colored status dot (no chunky icons / emoji).
function StatusDot({ tone }: { tone: "ok" | "muted" | "bad" }) {
  const color =
    tone === "ok" ? "bg-emerald-500" : tone === "bad" ? "bg-red-500" : "bg-muted-foreground/40";
  return <span className={`h-2 w-2 shrink-0 rounded-full ${color}`} />;
}

/**
 * Deliberate (non-polled) connectivity check for the LLMs a bank uses across
 * retain / consolidation / reflect. Probes on open and on demand. Shows status
 * only — never the provider/model/endpoint/key.
 */
export function LlmHealthDialog({
  bankId,
  open,
  onOpenChange,
}: {
  bankId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useTranslations("bankStats");
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<Awaited<ReturnType<typeof client.testBankLlm>> | null>(null);

  const run = async () => {
    setTesting(true);
    try {
      setResult(await client.testBankLlm(bankId));
    } catch {
      // Error toast is shown by the API client interceptor.
    } finally {
      setTesting(false);
    }
  };

  // Probe automatically when the dialog opens; reset when it closes.
  useEffect(() => {
    if (open) {
      run();
    } else {
      setResult(null);
    }
  }, [open, bankId]);

  const statusLabel: Record<string, string> = {
    connected: t("llmConnected"),
    not_configured: t("llmNotConfigured"),
    auth_failed: t("llmAuthFailed"),
    unreachable: t("llmUnreachable"),
    timeout: t("llmTimeout"),
  };
  const opLabel: Record<string, string> = {
    retain: t("opRetain"),
    consolidation: t("consolidation"),
    reflect: t("opReflect"),
  };
  const tone = (status: string): "ok" | "muted" | "bad" =>
    status === "connected" ? "ok" : status === "not_configured" ? "muted" : "bad";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("llmHealthTitle")}</DialogTitle>
          <DialogDescription>{t("llmTestHint")}</DialogDescription>
        </DialogHeader>

        <div className="py-2 space-y-2">
          {testing && !result ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
              <Loader2 className="w-4 h-4 animate-spin" />
              {t("testingLlm")}
            </div>
          ) : (
            result?.operations.map((op) => (
              <div
                key={op.operation}
                className="flex items-center justify-between gap-2 text-sm py-1.5 border-b border-border/50 last:border-0"
              >
                <span className="text-muted-foreground">
                  {opLabel[op.operation] ?? op.operation}
                </span>
                <span className="inline-flex items-center gap-2">
                  <StatusDot tone={tone(op.status)} />
                  <span className={op.ok ? "text-foreground font-medium" : "text-muted-foreground"}>
                    {statusLabel[op.status] ?? op.status}
                  </span>
                  {op.ok && op.latency_ms != null && (
                    <span className="text-muted-foreground/60 tabular-nums text-xs">
                      {Math.round(op.latency_ms)}ms
                    </span>
                  )}
                </span>
              </div>
            ))
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={run} disabled={testing} className="gap-1.5">
            {testing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <RefreshCw className="w-3.5 h-3.5" />
            )}
            {testing ? t("testingLlm") : t("llmRetest")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
