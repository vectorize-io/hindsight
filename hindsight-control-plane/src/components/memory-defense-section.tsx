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
import { Loader2, Sparkles } from "lucide-react";
import { client } from "@/lib/api";

const ENTERPRISE_DEMO_URL = "https://calendly.com/d/ctw6-byb-3kg";

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

      <EnterpriseDiscoveryPanel />

      {saving && (
        <div className="flex justify-end items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          {t("saving")}
        </div>
      )}
    </section>
  );
}

// Product-discovery surface. The OSS extension only ships sensitive_data
// scanning; everything below sits behind Hindsight Cloud entitlements. This
// panel is informational only — no fake disabled toggles, no entitlement
// fetch, no flag plumbing — just a list of what's available with a CTA. The
// panel sits outside the master-enabled disabled-wrapper above so it remains
// legible when the master switch is off (discovery isn't gated on policy).
function EnterpriseDiscoveryPanel() {
  const t = useTranslations("bankConfig");
  const features = [
    t("memoryDefenseEnterpriseFeatureDetectSecrets"),
    t("memoryDefenseEnterpriseFeatureBase64"),
    t("memoryDefenseEnterpriseFeatureLlm"),
    t("memoryDefenseEnterpriseFeaturePromptInjection"),
    t("memoryDefenseEnterpriseFeatureSizeAnomaly"),
    t("memoryDefenseEnterpriseFeatureProtectedKeys"),
  ];
  return (
    <div className="rounded-xl border border-emerald-600/15 bg-emerald-500/[0.04] dark:bg-emerald-400/[0.06] p-6 flex flex-col gap-4">
      <div className="flex items-start gap-3">
        <div className="rounded-full bg-emerald-500/10 p-2 shrink-0">
          <Sparkles className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
        </div>
        <div className="flex-1">
          <p className="text-base font-semibold text-foreground">
            {t("memoryDefenseEnterpriseTitle")}
          </p>
          <p className="text-sm text-muted-foreground mt-1">
            {t("memoryDefenseEnterpriseSubtitle")}
          </p>
        </div>
      </div>
      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5 text-sm text-foreground">
        {features.map((f) => (
          <li key={f} className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span>{f}</span>
          </li>
        ))}
      </ul>
      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
        <a
          href={ENTERPRISE_DEMO_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center justify-center gap-2 rounded-md bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold px-4 py-2.5 transition-colors w-full sm:w-auto"
        >
          {t("memoryDefenseEnterpriseCta")}
          <span aria-hidden>→</span>
        </a>
      </div>
    </div>
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
