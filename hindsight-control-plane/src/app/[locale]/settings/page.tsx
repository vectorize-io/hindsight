"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Settings,
  Save,
  Globe,
  Clock,
  Palette,
  Cpu,
  Bell,
  Shield,
  CheckCircle2,
} from "lucide-react";

export default function SettingsPage() {
  const t = useTranslations("operator.settings");
  const [activeTab, setActiveTab] = useState<"general" | "performance" | "notifications" | "security">("general");
  const [saved, setSaved] = useState(false);
  const [settings, setSettings] = useState({
    systemName: "CollabMind Operator",
    language: "en",
    timezone: "UTC",
    theme: "system",
    workerCount: "4",
    cacheTTL: "3600",
    batchSize: "50",
    pollingInterval: "30",
    systemAlerts: true,
    agentNotifications: true,
    errorReports: true,
    weeklyDigest: false,
    sessionTimeout: "30",
    maxLoginAttempts: "5",
    apiKeyRotation: "90",
    auditLogRetention: "365",
  });

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>

        <div className="flex items-center gap-4 border-b pb-2">
          <button onClick={() => setActiveTab("general")} className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "general" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
            {t("tabGeneral")}
          </button>
          <button onClick={() => setActiveTab("performance")} className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "performance" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
            {t("tabPerformance")}
          </button>
          <button onClick={() => setActiveTab("notifications")} className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "notifications" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
            {t("tabNotifications")}
          </button>
          <button onClick={() => setActiveTab("security")} className={`text-sm pb-2 px-1 -mb-2 border-b-2 transition-colors ${activeTab === "security" ? "border-primary font-medium" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
            {t("tabSecurity")}
          </button>
          <div className="ml-auto">
            <Button size="sm" onClick={handleSave} disabled={saved}>
              {saved ? (
                <><CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />{t("saved")}</>
              ) : (
                <><Save className="w-3.5 h-3.5 mr-1.5" />{t("save")}</>
              )}
            </Button>
          </div>
        </div>

        {activeTab === "general" && (
          <div className="space-y-4 max-w-2xl">
            <Card>
              <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Globe className="w-4 h-4" />{t("tabGeneral")}</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("systemName")}</label>
                    <Input className="h-9 text-xs" value={settings.systemName} onChange={e => setSettings({ ...settings, systemName: e.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("language")}</label>
                    <select className="w-full h-9 text-xs border rounded-md px-2 bg-background" value={settings.language} onChange={e => setSettings({ ...settings, language: e.target.value })}>
                      <option value="en">English</option>
                      <option value="zh">中文</option>
                      <option value="ja">日本語</option>
                      <option value="ko">한국어</option>
                      <option value="fr">Français</option>
                      <option value="de">Deutsch</option>
                      <option value="es">Español</option>
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("timezone")}</label>
                    <select className="w-full h-9 text-xs border rounded-md px-2 bg-background" value={settings.timezone} onChange={e => setSettings({ ...settings, timezone: e.target.value })}>
                      <option value="UTC">UTC</option>
                      <option value="America/New_York">Eastern</option>
                      <option value="America/Chicago">Central</option>
                      <option value="America/Denver">Mountain</option>
                      <option value="America/Los_Angeles">Pacific</option>
                      <option value="Europe/London">London</option>
                      <option value="Europe/Berlin">Berlin</option>
                      <option value="Asia/Tokyo">Tokyo</option>
                      <option value="Asia/Shanghai">Shanghai</option>
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("theme")}</label>
                    <select className="w-full h-9 text-xs border rounded-md px-2 bg-background" value={settings.theme} onChange={e => setSettings({ ...settings, theme: e.target.value })}>
                      <option value="system">System</option>
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                    </select>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "performance" && (
          <div className="space-y-4 max-w-2xl">
            <Card>
              <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Cpu className="w-4 h-4" />{t("tabPerformance")}</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("workerCount")}</label>
                    <Input type="number" className="h-9 text-xs" value={settings.workerCount} onChange={e => setSettings({ ...settings, workerCount: e.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("cacheTTL")} (s)</label>
                    <Input type="number" className="h-9 text-xs" value={settings.cacheTTL} onChange={e => setSettings({ ...settings, cacheTTL: e.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("batchSize")}</label>
                    <Input type="number" className="h-9 text-xs" value={settings.batchSize} onChange={e => setSettings({ ...settings, batchSize: e.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("pollingInterval")} (s)</label>
                    <Input type="number" className="h-9 text-xs" value={settings.pollingInterval} onChange={e => setSettings({ ...settings, pollingInterval: e.target.value })} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "notifications" && (
          <div className="space-y-4 max-w-2xl">
            <Card>
              <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Bell className="w-4 h-4" />{t("tabNotifications")}</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {[
                  { key: "systemAlerts", label: t("systemAlerts") },
                  { key: "agentNotifications", label: t("agentNotifications") },
                  { key: "errorReports", label: t("errorReports") },
                  { key: "weeklyDigest", label: t("weeklyDigest") },
                ].map(({ key, label }) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-sm">{label}</span>
                    <button
                      className={`w-10 h-5 rounded-full transition-colors ${settings[key as keyof typeof settings] ? "bg-primary" : "bg-muted"}`}
                      onClick={() => setSettings({ ...settings, [key]: !settings[key as keyof typeof settings] })}
                    >
                      <div className={`w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${settings[key as keyof typeof settings] ? "translate-x-5" : "translate-x-0.5"}`} />
                    </button>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "security" && (
          <div className="space-y-4 max-w-2xl">
            <Card>
              <CardHeader><CardTitle className="text-sm flex items-center gap-2"><Shield className="w-4 h-4" />{t("tabSecurity")}</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("sessionTimeout")} (min)</label>
                    <Input type="number" className="h-9 text-xs" value={settings.sessionTimeout} onChange={e => setSettings({ ...settings, sessionTimeout: e.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("maxLoginAttempts")}</label>
                    <Input type="number" className="h-9 text-xs" value={settings.maxLoginAttempts} onChange={e => setSettings({ ...settings, maxLoginAttempts: e.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("apiKeyRotation")} (days)</label>
                    <Input type="number" className="h-9 text-xs" value={settings.apiKeyRotation} onChange={e => setSettings({ ...settings, apiKeyRotation: e.target.value })} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium">{t("auditLogRetention")} (days)</label>
                    <Input type="number" className="h-9 text-xs" value={settings.auditLogRetention} onChange={e => setSettings({ ...settings, auditLogRetention: e.target.value })} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </OperatorShell>
  );
}
