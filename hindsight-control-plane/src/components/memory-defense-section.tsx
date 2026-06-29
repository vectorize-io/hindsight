"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Loader2 } from "lucide-react";
import { client } from "@/lib/api";

const DETECTORS = {
  SENSITIVE_DATA: "sensitive_data",
} as const;

type Detector = (typeof DETECTORS)[keyof typeof DETECTORS];

type Action = "allow" | "redact" | "block";

function coerceAction(raw: unknown): Action {
  if (raw === "allow" || raw === "redact" || raw === "block") return raw;
  return "block";
}

interface PolicyRule {
  on: Detector;
  action: Action;
}

interface MemoryDefensePolicy {
  enabled: boolean;
  rules: PolicyRule[];
}

function emptyPolicy(): MemoryDefensePolicy {
  return {
    enabled: false,
    rules: [],
  };
}

function readPolicy(config: Record<string, any>): MemoryDefensePolicy {
  const raw = config?.memory_defense;
  if (!raw || typeof raw !== "object") return emptyPolicy();
  return {
    enabled: Boolean(raw.enabled),
    rules: Array.isArray(raw.rules)
      ? raw.rules
          .filter((r: any) => r && typeof r.on === "string" && r.on === DETECTORS.SENSITIVE_DATA)
          .map((r: any) => ({
            on: r.on as Detector,
            action: coerceAction(r.action ?? "redact"),
          }))
      : [],
  };
}

function findRule(rules: PolicyRule[], on: Detector): PolicyRule | undefined {
  return rules.find((r) => r.on === on);
}

function upsertRule(rules: PolicyRule[], rule: PolicyRule): PolicyRule[] {
  return [...rules.filter((r) => r.on !== rule.on), rule];
}

function removeRule(rules: PolicyRule[], on: Detector): PolicyRule[] {
  return rules.filter((r) => r.on !== on);
}

function writePolicy(p: MemoryDefensePolicy): Record<string, any> {
  return {
    enabled: p.enabled,
    rules: p.rules.map((r) => ({
      on: r.on,
      action: r.action,
    })),
  };
}

interface MemoryDefenseSectionProps {
  bankId: string;
}

export function MemoryDefenseSection({ bankId }: MemoryDefenseSectionProps) {
  const t = useTranslations("bankConfig");

  const [loading, setLoading] = useState(true);
  const [baseConfig, setBaseConfig] = useState<Record<string, any>>({});
  const [edits, setEdits] = useState<MemoryDefensePolicy>(emptyPolicy());
  const [saving, setSaving] = useState(false);

  const basePolicy = useMemo(() => readPolicy(baseConfig), [baseConfig]);

  const dirty = useMemo(
    () => JSON.stringify(writePolicy(edits)) !== JSON.stringify(writePolicy(basePolicy)),
    [edits, basePolicy]
  );

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      try {
        const resp = await client.getBankConfig(bankId);
        if (cancelled) return;
        setBaseConfig(resp.config);
        setEdits(readPolicy(resp.config));
      } catch (err: any) {
        if (!cancelled) {
          toast.error(err?.message || t("memoryDefenseFailedToSave"));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [bankId, t]);

  const masterEnabled = edits.enabled;
  const setMaster = (v: boolean) => setEdits((p) => ({ ...p, enabled: v }));

  const toggleSensitiveData = (enabled: boolean) =>
    setEdits((p) => {
      const existing = findRule(p.rules, DETECTORS.SENSITIVE_DATA);
      const action: Action = existing?.action ?? "redact";
      return {
        ...p,
        rules: enabled
          ? upsertRule(p.rules, {
              on: DETECTORS.SENSITIVE_DATA,
              action,
            })
          : removeRule(p.rules, DETECTORS.SENSITIVE_DATA),
      };
    });

  const setSensitiveDataAction = (action: Action) =>
    setEdits((p) => {
      const existing = findRule(p.rules, DETECTORS.SENSITIVE_DATA);
      if (!existing) return p;
      return { ...p, rules: upsertRule(p.rules, { ...existing, action }) };
    });

  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const skipFirstSave = useRef(true);

  useEffect(() => {
    if (loading) return;
    if (skipFirstSave.current) {
      skipFirstSave.current = false;
      return;
    }
    if (!dirty) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      void performSave();
    }, 800);
    return () => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
  }, [edits, loading]);

  const performSave = async () => {
    setSaving(true);
    try {
      const payload = writePolicy(edits);
      const resp = await client.updateBankConfig(bankId, { memory_defense: payload });
      setBaseConfig(resp.config);
    } catch (err: any) {
      const msg = err?.message || t("memoryDefenseFailedToSave");
      toast.error(msg);
    } finally {
      setSaving(false);
    }
  };

  const sensitiveData = findRule(edits.rules, DETECTORS.SENSITIVE_DATA);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <section className="space-y-6">
      <div className="flex items-start justify-between gap-6">
        <div className="flex-1">
          <h2 className="text-lg font-semibold">{t("memoryDefenseTitle")}</h2>
          <p className="text-sm text-muted-foreground">{t("memoryDefenseDescription")}</p>
        </div>
        <div className="shrink-0 pt-1 scale-125 origin-right">
          <Switch checked={masterEnabled} onCheckedChange={setMaster} />
        </div>
      </div>

      <div
        className={
          masterEnabled ? "space-y-6" : "space-y-6 opacity-40 pointer-events-none select-none"
        }
        aria-disabled={!masterEnabled}
      >
        <div className="space-y-3">
          <Card className="bg-muted/20 border-border/40">
            <DetectorCard
              title={t("memoryDefenseSecretLeakTitle")}
              description={t("memoryDefenseSecretLeakDescription")}
              rule={sensitiveData}
              actions={["redact", "block", "allow"]}
              onToggle={toggleSensitiveData}
              onActionChange={setSensitiveDataAction}
              masterEnabled={masterEnabled}
            />
          </Card>
        </div>
      </div>

      {saving && (
        <div className="flex justify-end items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          {t("saving")}
        </div>
      )}
    </section>
  );
}

function SubSection({
  title,
  description,
  headerSummary,
  headerControl,
  children,
}: {
  title: string;
  description: string;
  headerSummary?: string | null;
  headerControl?: ReactNode;
  children?: ReactNode;
}) {
  const childArr = Array.isArray(children)
    ? children.flat(Infinity).filter(Boolean)
    : children
      ? [children]
      : [];
  const hasContent = childArr.length > 0;

  return (
    <div className="px-6 py-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-baseline gap-2 flex-wrap">
            <p className="text-sm font-semibold">{title}</p>
            {headerSummary && (
              <span className="text-[10px] font-bold tracking-[0.1em] text-primary uppercase px-1.5 py-0.5 rounded bg-primary/10">
                {headerSummary}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        </div>
        {headerControl && <div className="shrink-0 pt-0.5">{headerControl}</div>}
      </div>
      {hasContent && <div className="mt-4 space-y-3">{children}</div>}
    </div>
  );
}

function Row({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex-1">
        <Label className="text-sm font-medium">{label}</Label>
        {description && <p className="text-xs text-muted-foreground mt-1">{description}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function ActionSelect({
  value,
  onChange,
  options,
  t,
}: {
  value: Action;
  onChange: (a: Action) => void;
  options: Action[];
  t: (key: string) => string;
}) {
  return (
    <Select value={value} onValueChange={(v) => onChange(v as Action)}>
      <SelectTrigger className="w-[160px]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((a) => (
          <SelectItem key={a} value={a}>
            {t(`memoryDefenseAction_${a}`)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

interface DetectorCardProps {
  title: string;
  description: string;
  rule: PolicyRule | undefined;
  actions: Action[];
  onToggle: (v: boolean) => void;
  onActionChange: (a: Action) => void;
  masterEnabled: boolean;
}

function DetectorCard({
  title,
  description,
  rule,
  actions,
  onActionChange,
  onToggle,
  masterEnabled,
}: DetectorCardProps) {
  const t = useTranslations("bankConfig");
  const enabled = !!rule;
  const effectivelyEnabled = masterEnabled && enabled;
  const statusSummary = effectivelyEnabled ? t(`memoryDefenseAction_${rule!.action}`) : null;

  return (
    <SubSection
      title={title}
      description={description}
      headerSummary={statusSummary}
      headerControl={<Switch checked={effectivelyEnabled} onCheckedChange={onToggle} />}
    >
      {effectivelyEnabled && actions.length > 1 && (
        <Row label={t("memoryDefenseActionLabel")}>
          <ActionSelect value={rule!.action} onChange={onActionChange} options={actions} t={t} />
        </Row>
      )}
    </SubSection>
  );
}
