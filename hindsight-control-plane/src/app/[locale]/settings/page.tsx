"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { useParams } from "next/navigation";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Settings2,
  Zap,
  Bell,
  Shield,
  Monitor,
  Globe,
  Clock,
  Palette,
  Loader2,
  Save,
  CheckCircle2,
} from "lucide-react";

export default function SettingsPage() {
  const t = useTranslations("operator");
  const params = useParams();
  const locale = (params?.locale as string) || "en";
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);

  const [systemName, setSystemName] = useState("CollabMind Hindsight");
  const [language, setLanguage] = useState("en");
  const [timezone, setTimezone] = useState("UTC");
  const [theme, setTheme] = useState("system");
  const [workerCount, setWorkerCount] = useState(4);
  const [cacheTTL, setCacheTTL] = useState(300);
  const [batchSize, setBatchSize] = useState(50);
  const [pollingInterval, setPollingInterval] = useState(30);
  const [systemAlerts, setSystemAlerts] = useState(true);
  const [agentNotifications, setAgentNotifications] = useState(true);
  const [errorReports, setErrorReports] = useState(true);
  const [weeklyDigest, setWeeklyDigest] = useState(false);
  const [sessionTimeout, setSessionTimeout] = useState(60);
  const [maxLoginAttempts, setMaxLoginAttempts] = useState(5);
  const [apiKeyRotation, setApiKeyRotation] = useState(90);
  const [auditLogRetention, setAuditLogRetention] = useState(365);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch("/api/system/config");
        if (res.ok) {
          const data = await res.json();
          if (data?.llm?.provider) setSystemName(`CollabMind ${data.llm.provider}`);
        }
      } catch {
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  if (loading) {
    return (
      <OperatorShell>
        <div className="p-6">
          <div className="flex items-center justify-center py-16 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading settings...
          </div>
        </div>
      </OperatorShell>
    );
  }

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <Settings2 className="h-6 w-6 text-primary" />
            {t("panels.settings")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{t("descriptions.settings")}</p>
        </div>

        <Tabs defaultValue="general" className="space-y-4">
          <TabsList>
            <TabsTrigger value="general" className="flex items-center gap-2">
              <Monitor className="h-4 w-4" /> General
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <Zap className="h-4 w-4" /> Performance
            </TabsTrigger>
            <TabsTrigger value="notifications" className="flex items-center gap-2">
              <Bell className="h-4 w-4" /> Notifications
            </TabsTrigger>
            <TabsTrigger value="security" className="flex items-center gap-2">
              <Shield className="h-4 w-4" /> Security
            </TabsTrigger>
          </TabsList>

          <TabsContent value="general" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">General Settings</CardTitle>
                <CardDescription>System-wide preferences and localization</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">System Name</label>
                    <Input value={systemName} onChange={(e) => setSystemName(e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Default Language</label>
                    <select
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                    >
                      <option value="en">English</option>
                      <option value="fr">Français</option>
                      <option value="de">Deutsch</option>
                      <option value="ja">日本語</option>
                      <option value="zh">中文</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Timezone</label>
                    <select
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={timezone}
                      onChange={(e) => setTimezone(e.target.value)}
                    >
                      <option value="UTC">UTC</option>
                      <option value="America/New_York">America/New_York</option>
                      <option value="America/Los_Angeles">America/Los_Angeles</option>
                      <option value="Europe/London">Europe/London</option>
                      <option value="Europe/Paris">Europe/Paris</option>
                      <option value="Asia/Tokyo">Asia/Tokyo</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Theme</label>
                    <select
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={theme}
                      onChange={(e) => setTheme(e.target.value)}
                    >
                      <option value="system">System</option>
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                    </select>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Performance Settings</CardTitle>
                <CardDescription>Worker pool, caching, and processing tuning</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Worker Count ({workerCount})</label>
                    <Input
                      type="range"
                      min={1}
                      max={16}
                      value={workerCount}
                      onChange={(e) => setWorkerCount(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Cache TTL (seconds)</label>
                    <Input
                      type="number"
                      value={cacheTTL}
                      onChange={(e) => setCacheTTL(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Batch Size</label>
                    <Input
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Polling Interval (seconds)</label>
                    <Input
                      type="number"
                      value={pollingInterval}
                      onChange={(e) => setPollingInterval(Number(e.target.value))}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="notifications" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Notification Preferences</CardTitle>
                <CardDescription>Control what alerts you receive</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">System Alerts</p>
                      <p className="text-xs text-muted-foreground">Critical system events and failures</p>
                    </div>
                    <Switch checked={systemAlerts} onCheckedChange={setSystemAlerts} />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">Agent Notifications</p>
                      <p className="text-xs text-muted-foreground">Agent task completions and failures</p>
                    </div>
                    <Switch checked={agentNotifications} onCheckedChange={setAgentNotifications} />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">Error Reports</p>
                      <p className="text-xs text-muted-foreground">Detailed error reports for debugging</p>
                    </div>
                    <Switch checked={errorReports} onCheckedChange={setErrorReports} />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">Weekly Digest</p>
                      <p className="text-xs text-muted-foreground">Weekly summary of system activity</p>
                    </div>
                    <Switch checked={weeklyDigest} onCheckedChange={setWeeklyDigest} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="security" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Security Settings</CardTitle>
                <CardDescription>Authentication, encryption, and audit configuration</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Session Timeout (minutes)</label>
                    <Input
                      type="number"
                      value={sessionTimeout}
                      onChange={(e) => setSessionTimeout(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Login Attempts</label>
                    <Input
                      type="number"
                      value={maxLoginAttempts}
                      onChange={(e) => setMaxLoginAttempts(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">API Key Rotation (days)</label>
                    <Input
                      type="number"
                      value={apiKeyRotation}
                      onChange={(e) => setApiKeyRotation(Number(e.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Audit Log Retention (days)</label>
                    <Input
                      type="number"
                      value={auditLogRetention}
                      onChange={(e) => setAuditLogRetention(Number(e.target.value))}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <div className="flex items-center gap-3">
          <Button onClick={handleSave} className="flex items-center gap-2">
            {saved ? (
              <>
                <CheckCircle2 className="h-4 w-4" /> Settings Saved
              </>
            ) : (
              <>
                <Save className="h-4 w-4" /> Save Settings
              </>
            )}
          </Button>
        </div>
      </div>
    </OperatorShell>
  );
}
