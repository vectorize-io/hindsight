"use client";

import { useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useBank } from "@/lib/bank-context";
import { client } from "@/lib/api";
import { Loader2 } from "lucide-react";

type BackupSettings = {
  backup_enabled: boolean;
  backup_retention_days: number;
};

const DEFAULT_BACKUP_SETTINGS: BackupSettings = {
  backup_enabled: false,
  backup_retention_days: 7,
};

function backupSettingsFromConfig(config: Record<string, unknown>): BackupSettings {
  const retention = config.backup_retention_days;
  return {
    backup_enabled: config.backup_enabled === true,
    backup_retention_days: typeof retention === "number" ? Math.min(Math.max(retention, 1), 7) : 7,
  };
}

export function BankBackupSettingsView() {
  const t = useTranslations("bankBackup");
  const { currentBank: bankId } = useBank();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [baseSettings, setBaseSettings] = useState<BackupSettings>(DEFAULT_BACKUP_SETTINGS);
  const [settings, setSettings] = useState<BackupSettings>(DEFAULT_BACKUP_SETTINGS);

  const dirty = useMemo(
    () => JSON.stringify(settings) !== JSON.stringify(baseSettings),
    [settings, baseSettings]
  );

  useEffect(() => {
    if (!bankId) return;
    let cancelled = false;

    async function load() {
      if (!bankId) return;
      setLoading(true);
      try {
        const response = await client.getBankConfig(bankId);
        if (cancelled) return;
        const next = backupSettingsFromConfig(response.config as Record<string, unknown>);
        setBaseSettings(next);
        setSettings(next);
      } catch {
        if (!cancelled) toast.error(t("loadFailed"));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [bankId, t]);

  const save = async () => {
    if (!bankId) return;
    setSaving(true);
    try {
      const payload = {
        backup_enabled: settings.backup_enabled,
        backup_retention_days: Math.min(Math.max(settings.backup_retention_days, 1), 7),
      } satisfies BackupSettings;
      await client.updateBankConfig(bankId, payload);
      setBaseSettings(payload);
      setSettings(payload);
      toast.success(t("saved"));
    } catch {
      toast.error(t("saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  if (!bankId) {
    return <p className="text-sm text-muted-foreground">{t("noBankSelected")}</p>;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <Card className="p-6 space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-foreground">{t("title")}</h2>
        <p className="text-sm text-muted-foreground mt-1">{t("description")}</p>
      </div>

      <div className="grid gap-6 md:grid-cols-[1fr_320px] md:items-center">
        <div>
          <Label className="text-sm font-medium">{t("enableDailyLabel")}</Label>
          <p className="text-sm text-muted-foreground mt-1">{t("enableDailyDescription")}</p>
        </div>
        <div className="flex justify-end">
          <Switch
            checked={settings.backup_enabled}
            onCheckedChange={(checked) =>
              setSettings((prev) => ({ ...prev, backup_enabled: checked }))
            }
          />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-[1fr_320px] md:items-center">
        <div>
          <Label htmlFor="backup-retention-days" className="text-sm font-medium">
            {t("retentionLabel")}
          </Label>
          <p className="text-sm text-muted-foreground mt-1">{t("retentionDescription")}</p>
        </div>
        <Input
          id="backup-retention-days"
          type="number"
          min={1}
          max={7}
          value={settings.backup_retention_days}
          onChange={(event) => {
            const value = Number.parseInt(event.target.value, 10);
            setSettings((prev) => ({
              ...prev,
              backup_retention_days: Number.isFinite(value) ? Math.min(Math.max(value, 1), 7) : 7,
            }));
          }}
        />
      </div>

      <div className="flex justify-end">
        <Button onClick={save} disabled={!dirty || saving}>
          {saving ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              {t("saving")}
            </>
          ) : (
            t("save")
          )}
        </Button>
      </div>
    </Card>
  );
}
