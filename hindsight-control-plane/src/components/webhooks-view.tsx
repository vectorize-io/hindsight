"use client";

import { useState, useEffect } from "react";
import { useBank } from "@/lib/bank-context";
import { client, Webhook, WebhookDelivery, WebhookHttpConfig } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  RefreshCw,
  Plus,
  Trash2,
  Eye,
  EyeOff,
  Loader2,
  CheckCircle,
  AlertCircle,
  Clock,
  X,
  ChevronDown,
  ChevronRight,
  Pencil,
} from "lucide-react";

const AVAILABLE_EVENT_TYPES = ["consolidation.completed", "retain.completed"];

interface KeyValuePair {
  key: string;
  value: string;
}

interface CreateWebhookForm {
  url: string;
  secret: string;
  event_types: string[];
  enabled: boolean;
  http_config: {
    method: string;
    timeout_seconds: number;
    headers: KeyValuePair[];
    params: KeyValuePair[];
  };
}

const DEFAULT_FORM: CreateWebhookForm = {
  url: "",
  secret: "",
  event_types: ["consolidation.completed"],
  enabled: true,
  http_config: {
    method: "POST",
    timeout_seconds: 30,
    headers: [],
    params: [],
  },
};

function kvPairsToRecord(pairs: KeyValuePair[]): Record<string, string> {
  return Object.fromEntries(pairs.filter((p) => p.key.trim()).map((p) => [p.key, p.value]));
}

function buildHttpConfig(cfg: CreateWebhookForm["http_config"]): WebhookHttpConfig {
  return {
    method: cfg.method,
    timeout_seconds: cfg.timeout_seconds,
    headers: kvPairsToRecord(cfg.headers),
    params: kvPairsToRecord(cfg.params),
  };
}

interface KVEditorProps {
  label: string;
  pairs: KeyValuePair[];
  onChange: (pairs: KeyValuePair[]) => void;
}

function KVEditor({ label, pairs, onChange }: KVEditorProps) {
  const addPair = () => onChange([...pairs, { key: "", value: "" }]);
  const removePair = (i: number) => onChange(pairs.filter((_, idx) => idx !== i));
  const updatePair = (i: number, field: "key" | "value", val: string) => {
    const next = pairs.map((p, idx) => (idx === i ? { ...p, [field]: val } : p));
    onChange(next);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-6 text-xs px-2"
          onClick={addPair}
        >
          <Plus className="w-3 h-3 mr-1" />
          Add
        </Button>
      </div>
      {pairs.length > 0 && (
        <div className="space-y-1.5">
          {pairs.map((pair, i) => (
            <div key={i} className="flex items-center gap-2">
              <Input
                placeholder="Key"
                value={pair.key}
                onChange={(e) => updatePair(i, "key", e.target.value)}
                className="h-8 text-sm flex-1"
              />
              <Input
                placeholder="Value"
                value={pair.value}
                onChange={(e) => updatePair(i, "value", e.target.value)}
                className="h-8 text-sm flex-1"
              />
              <button
                type="button"
                onClick={() => removePair(i)}
                className="text-muted-foreground hover:text-foreground shrink-0"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function statusBadge(status: string) {
  if (status === "completed")
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
        <CheckCircle className="w-3 h-3" />
        delivered
      </span>
    );
  if (status === "pending")
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
        <Clock className="w-3 h-3" />
        pending
      </span>
    );
  if (status === "failed")
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20">
        <AlertCircle className="w-3 h-3" />
        failed
      </span>
    );
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-muted text-muted-foreground border border-border">
      {status}
    </span>
  );
}

function DeliveryTableRow({
  delivery,
  formatDate,
}: {
  delivery: WebhookDelivery;
  formatDate: (d: string | null) => string;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasDetails = delivery.last_response_body || delivery.last_error;

  return (
    <>
      <TableRow
        className={hasDetails ? "cursor-pointer hover:bg-muted/40" : undefined}
        onClick={() => hasDetails && setExpanded((v) => !v)}
      >
        <TableCell className="w-8 pl-3 pr-0">
          {hasDetails ? (
            expanded ? (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            )
          ) : (
            <span className="w-4 h-4 inline-block" />
          )}
        </TableCell>
        <TableCell>{statusBadge(delivery.status)}</TableCell>
        <TableCell>
          {delivery.last_response_status != null ? (
            <span
              className={`font-mono text-xs px-1.5 py-0.5 rounded border ${
                delivery.last_response_status >= 200 && delivery.last_response_status < 300
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                  : "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20"
              }`}
            >
              {delivery.last_response_status}
            </span>
          ) : (
            <span className="text-muted-foreground text-xs">—</span>
          )}
        </TableCell>
        <TableCell className="text-center text-sm">{delivery.attempts}</TableCell>
        <TableCell className="font-mono text-xs text-muted-foreground">
          {delivery.event_type}
        </TableCell>
        <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
          {formatDate(delivery.created_at)}
        </TableCell>
      </TableRow>
      {expanded && hasDetails && (
        <TableRow className="bg-muted/20 hover:bg-muted/20">
          <TableCell colSpan={6} className="px-4 py-3">
            <div className="space-y-3">
              {delivery.last_error && (
                <div className="space-y-1">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Error
                  </p>
                  <p className="font-mono text-xs text-red-600 dark:text-red-400 break-all">
                    {delivery.last_error}
                  </p>
                </div>
              )}
              {delivery.last_response_body && (
                <div className="space-y-1">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Response Body
                    {delivery.last_attempt_at && (
                      <span className="ml-2 normal-case font-normal">
                        · {formatDate(delivery.last_attempt_at)}
                      </span>
                    )}
                  </p>
                  <pre className="font-mono text-xs bg-background rounded p-2 overflow-x-auto whitespace-pre-wrap break-all max-h-40 overflow-y-auto border border-border">
                    {delivery.last_response_body}
                  </pre>
                </div>
              )}
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

function webhookToForm(webhook: Webhook): CreateWebhookForm {
  const cfg = webhook.http_config ?? {
    method: "POST",
    timeout_seconds: 30,
    headers: {},
    params: {},
  };
  return {
    url: webhook.url,
    secret: "",
    event_types: webhook.event_types,
    enabled: webhook.enabled,
    http_config: {
      method: cfg.method,
      timeout_seconds: cfg.timeout_seconds,
      headers: Object.entries(cfg.headers ?? {}).map(([key, value]) => ({ key, value })),
      params: Object.entries(cfg.params ?? {}).map(([key, value]) => ({ key, value })),
    },
  };
}

export function WebhooksView() {
  const { currentBank } = useBank();
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [loading, setLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteConfirmWebhook, setDeleteConfirmWebhook] = useState<Webhook | null>(null);
  const [showSecret, setShowSecret] = useState(false);

  const [form, setForm] = useState<CreateWebhookForm>(DEFAULT_FORM);

  // Edit dialog state
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingWebhook, setEditingWebhook] = useState<Webhook | null>(null);
  const [editForm, setEditForm] = useState<CreateWebhookForm>(DEFAULT_FORM);
  const [showEditSecret, setShowEditSecret] = useState(false);
  const [clearSecret, setClearSecret] = useState(false);
  const [saving, setSaving] = useState(false);

  // Deliveries dialog state
  const [deliveriesDialogOpen, setDeliveriesDialogOpen] = useState(false);
  const [selectedWebhook, setSelectedWebhook] = useState<Webhook | null>(null);
  const [deliveries, setDeliveries] = useState<WebhookDelivery[]>([]);
  const [loadingDeliveries, setLoadingDeliveries] = useState(false);
  const [deliveriesCursor, setDeliveriesCursor] = useState<string | null>(null);
  const [loadingMoreDeliveries, setLoadingMoreDeliveries] = useState(false);

  const loadWebhooks = async () => {
    if (!currentBank) return;
    setLoading(true);
    try {
      const data = await client.listWebhooks(currentBank);
      setWebhooks(data.items || []);
    } catch (error) {
      console.error("Error loading webhooks:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (currentBank) {
      loadWebhooks();
    }
  }, [currentBank]);

  const handleCreate = async () => {
    if (!currentBank || !form.url) return;
    setCreating(true);
    try {
      await client.createWebhook(currentBank, {
        url: form.url,
        secret: form.secret || undefined,
        event_types: form.event_types.length > 0 ? form.event_types : undefined,
        enabled: form.enabled,
        http_config: buildHttpConfig(form.http_config),
      });
      setCreateDialogOpen(false);
      setForm(DEFAULT_FORM);
      setShowSecret(false);
      await loadWebhooks();
    } catch (error) {
      // Error toast shown by API client interceptor
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (webhookId: string) => {
    if (!currentBank) return;
    setDeletingId(webhookId);
    setDeleteConfirmWebhook(null);
    try {
      await client.deleteWebhook(currentBank, webhookId);
      await loadWebhooks();
    } catch (error) {
      // Error toast shown by API client interceptor
    } finally {
      setDeletingId(null);
    }
  };

  const handleOpenEdit = (webhook: Webhook) => {
    setEditingWebhook(webhook);
    setEditForm(webhookToForm(webhook));
    setShowEditSecret(false);
    setClearSecret(false);
    setEditDialogOpen(true);
  };

  const handleSave = async () => {
    if (!currentBank || !editingWebhook || !editForm.url) return;
    setSaving(true);
    try {
      const patch: Parameters<typeof client.updateWebhook>[2] = {
        url: editForm.url,
        event_types: editForm.event_types,
        enabled: editForm.enabled,
        http_config: buildHttpConfig(editForm.http_config),
      };
      if (clearSecret) {
        patch.secret = null;
      } else if (editForm.secret) {
        patch.secret = editForm.secret;
      }
      await client.updateWebhook(currentBank, editingWebhook.id, patch);
      setEditDialogOpen(false);
      await loadWebhooks();
    } catch {
      // Error toast shown by API client interceptor
    } finally {
      setSaving(false);
    }
  };

  const handleViewDeliveries = async (webhook: Webhook) => {
    if (!currentBank) return;
    setSelectedWebhook(webhook);
    setDeliveriesDialogOpen(true);
    setDeliveries([]);
    setDeliveriesCursor(null);
    setLoadingDeliveries(true);
    try {
      const data = await client.listWebhookDeliveries(currentBank, webhook.id, 50);
      setDeliveries(data.items || []);
      setDeliveriesCursor(data.next_cursor ?? null);
    } catch (error) {
      console.error("Error loading deliveries:", error);
    } finally {
      setLoadingDeliveries(false);
    }
  };

  const handleLoadMoreDeliveries = async () => {
    if (!currentBank || !selectedWebhook || !deliveriesCursor) return;
    setLoadingMoreDeliveries(true);
    try {
      const data = await client.listWebhookDeliveries(
        currentBank,
        selectedWebhook.id,
        50,
        deliveriesCursor
      );
      setDeliveries((prev) => [...prev, ...(data.items || [])]);
      setDeliveriesCursor(data.next_cursor ?? null);
    } catch (error) {
      console.error("Error loading more deliveries:", error);
    } finally {
      setLoadingMoreDeliveries(false);
    }
  };

  const toggleEventType = (eventType: string) => {
    setForm((prev) => ({
      ...prev,
      event_types: prev.event_types.includes(eventType)
        ? prev.event_types.filter((e) => e !== eventType)
        : [...prev.event_types, eventType],
    }));
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "N/A";
    return new Date(dateStr).toLocaleString();
  };

  if (!currentBank) return null;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">Webhooks</h3>
            <button
              onClick={() => loadWebhooks()}
              className="p-1 rounded hover:bg-muted transition-colors"
              title="Refresh webhooks"
              disabled={loading}
            >
              <RefreshCw
                className={`w-4 h-4 text-muted-foreground hover:text-foreground ${loading ? "animate-spin" : ""}`}
              />
            </button>
          </div>
          <p className="text-sm text-muted-foreground">
            {webhooks.length} webhook{webhooks.length !== 1 ? "s" : ""}
          </p>
        </div>
        <Button size="sm" onClick={() => setCreateDialogOpen(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Add Webhook
        </Button>
      </div>

      {/* Webhooks table */}
      {webhooks.length > 0 ? (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>URL</TableHead>
                <TableHead>Method</TableHead>
                <TableHead>Event Types</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created At</TableHead>
                <TableHead className="w-[120px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {webhooks.map((webhook) => (
                <TableRow key={webhook.id}>
                  <TableCell className="font-mono text-sm max-w-[300px] truncate">
                    <span title={webhook.url}>{webhook.url}</span>
                  </TableCell>
                  <TableCell>
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-muted text-muted-foreground border border-border">
                      {webhook.http_config?.method || "POST"}
                    </span>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {webhook.event_types.length > 0 ? (
                        webhook.event_types.map((et) => (
                          <span
                            key={et}
                            className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20"
                          >
                            {et}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground">All events</span>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    {webhook.enabled ? (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                        <CheckCircle className="w-3 h-3" />
                        Enabled
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-muted text-muted-foreground border border-border">
                        Disabled
                      </span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(webhook.created_at)}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => handleViewDeliveries(webhook)}
                        title="View deliveries"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        Deliveries
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs text-muted-foreground hover:text-foreground"
                        onClick={() => handleOpenEdit(webhook)}
                        title="Edit webhook"
                      >
                        <Pencil className="w-3 h-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                        onClick={() => setDeleteConfirmWebhook(webhook)}
                        disabled={deletingId === webhook.id}
                        title="Delete webhook"
                      >
                        {deletingId === webhook.id ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Trash2 className="w-3 h-3" />
                        )}
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ) : (
        <p className="text-muted-foreground text-center py-8 text-sm">
          No webhooks configured. Add a webhook to receive event notifications.
        </p>
      )}

      {/* Create Webhook Dialog */}
      <Dialog
        open={createDialogOpen}
        onOpenChange={(open) => {
          if (!open) {
            setForm(DEFAULT_FORM);
            setShowSecret(false);
          }
          setCreateDialogOpen(open);
        }}
      >
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Add Webhook</DialogTitle>
            <DialogDescription>
              Configure a webhook endpoint to receive event notifications.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            {/* Type (display only for now) */}
            <div className="space-y-1.5">
              <Label>Type</Label>
              <div className="flex items-center h-9 px-3 rounded-md border border-border bg-muted text-sm text-muted-foreground">
                HTTP
              </div>
            </div>

            {/* URL */}
            <div className="space-y-1.5">
              <Label htmlFor="webhook-url">
                URL <span className="text-red-500">*</span>
              </Label>
              <Input
                id="webhook-url"
                type="url"
                placeholder="https://example.com/webhook"
                value={form.url}
                onChange={(e) => setForm((prev) => ({ ...prev, url: e.target.value }))}
              />
            </div>

            {/* Method + Timeout row */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label>Method</Label>
                <Select
                  value={form.http_config.method}
                  onValueChange={(v) =>
                    setForm((prev) => ({
                      ...prev,
                      http_config: { ...prev.http_config, method: v },
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="POST">POST</SelectItem>
                    <SelectItem value="GET">GET</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="webhook-timeout">Timeout (seconds)</Label>
                <Input
                  id="webhook-timeout"
                  type="number"
                  min={1}
                  max={300}
                  value={form.http_config.timeout_seconds}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      http_config: {
                        ...prev.http_config,
                        timeout_seconds: parseInt(e.target.value) || 30,
                      },
                    }))
                  }
                />
              </div>
            </div>

            {/* Secret */}
            <div className="space-y-1.5">
              <Label htmlFor="webhook-secret">
                Secret <span className="text-muted-foreground text-xs">(optional)</span>
              </Label>
              <div className="relative">
                <Input
                  id="webhook-secret"
                  type={showSecret ? "text" : "password"}
                  placeholder="Signing secret"
                  value={form.secret}
                  onChange={(e) => setForm((prev) => ({ ...prev, secret: e.target.value }))}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowSecret((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  tabIndex={-1}
                >
                  {showSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Custom Headers */}
            <KVEditor
              label="Custom Headers"
              pairs={form.http_config.headers}
              onChange={(pairs) =>
                setForm((prev) => ({
                  ...prev,
                  http_config: { ...prev.http_config, headers: pairs },
                }))
              }
            />

            {/* Custom Query Params */}
            <KVEditor
              label="Query Parameters"
              pairs={form.http_config.params}
              onChange={(pairs) =>
                setForm((prev) => ({
                  ...prev,
                  http_config: { ...prev.http_config, params: pairs },
                }))
              }
            />

            {/* Event Types */}
            <div className="space-y-2">
              <Label>Event Types</Label>
              <div className="space-y-2">
                {AVAILABLE_EVENT_TYPES.map((eventType) => (
                  <div key={eventType} className="flex items-center gap-2">
                    <Checkbox
                      id={`event-${eventType}`}
                      checked={form.event_types.includes(eventType)}
                      onCheckedChange={() => toggleEventType(eventType)}
                    />
                    <Label
                      htmlFor={`event-${eventType}`}
                      className="font-mono text-sm cursor-pointer"
                    >
                      {eventType}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* Enabled toggle */}
            <div className="flex items-center gap-3">
              <Switch
                id="webhook-enabled"
                checked={form.enabled}
                onCheckedChange={(checked) => setForm((prev) => ({ ...prev, enabled: checked }))}
              />
              <Label htmlFor="webhook-enabled">Enabled</Label>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setCreateDialogOpen(false)}
              disabled={creating}
            >
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={creating || !form.url}>
              {creating ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create Webhook"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Webhook Dialog */}
      <Dialog
        open={editDialogOpen}
        onOpenChange={(open) => {
          if (!open) {
            setEditingWebhook(null);
            setEditForm(DEFAULT_FORM);
            setShowEditSecret(false);
            setClearSecret(false);
          }
          setEditDialogOpen(open);
        }}
      >
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Webhook</DialogTitle>
            <DialogDescription>Update the webhook configuration.</DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            {/* Type */}
            <div className="space-y-1.5">
              <Label>Type</Label>
              <div className="flex items-center h-9 px-3 rounded-md border border-border bg-muted text-sm text-muted-foreground">
                HTTP
              </div>
            </div>

            {/* URL */}
            <div className="space-y-1.5">
              <Label htmlFor="edit-webhook-url">
                URL <span className="text-red-500">*</span>
              </Label>
              <Input
                id="edit-webhook-url"
                type="url"
                placeholder="https://example.com/webhook"
                value={editForm.url}
                onChange={(e) => setEditForm((prev) => ({ ...prev, url: e.target.value }))}
              />
            </div>

            {/* Method + Timeout */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label>Method</Label>
                <Select
                  value={editForm.http_config.method}
                  onValueChange={(v) =>
                    setEditForm((prev) => ({
                      ...prev,
                      http_config: { ...prev.http_config, method: v },
                    }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="POST">POST</SelectItem>
                    <SelectItem value="GET">GET</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="edit-webhook-timeout">Timeout (seconds)</Label>
                <Input
                  id="edit-webhook-timeout"
                  type="number"
                  min={1}
                  max={300}
                  value={editForm.http_config.timeout_seconds}
                  onChange={(e) =>
                    setEditForm((prev) => ({
                      ...prev,
                      http_config: {
                        ...prev.http_config,
                        timeout_seconds: parseInt(e.target.value) || 30,
                      },
                    }))
                  }
                />
              </div>
            </div>

            {/* Secret */}
            <div className="space-y-1.5">
              <Label htmlFor="edit-webhook-secret">Secret</Label>
              <div className="relative">
                <Input
                  id="edit-webhook-secret"
                  type={showEditSecret ? "text" : "password"}
                  placeholder="Leave blank to keep existing"
                  value={editForm.secret}
                  disabled={clearSecret}
                  onChange={(e) => setEditForm((prev) => ({ ...prev, secret: e.target.value }))}
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowEditSecret((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  tabIndex={-1}
                >
                  {showEditSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <Checkbox
                  id="edit-clear-secret"
                  checked={clearSecret}
                  onCheckedChange={(v) => {
                    setClearSecret(!!v);
                    if (v) setEditForm((prev) => ({ ...prev, secret: "" }));
                  }}
                />
                <Label
                  htmlFor="edit-clear-secret"
                  className="text-xs text-muted-foreground cursor-pointer"
                >
                  Clear existing secret
                </Label>
              </div>
            </div>

            {/* Custom Headers */}
            <KVEditor
              label="Custom Headers"
              pairs={editForm.http_config.headers}
              onChange={(pairs) =>
                setEditForm((prev) => ({
                  ...prev,
                  http_config: { ...prev.http_config, headers: pairs },
                }))
              }
            />

            {/* Custom Query Params */}
            <KVEditor
              label="Query Parameters"
              pairs={editForm.http_config.params}
              onChange={(pairs) =>
                setEditForm((prev) => ({
                  ...prev,
                  http_config: { ...prev.http_config, params: pairs },
                }))
              }
            />

            {/* Event Types */}
            <div className="space-y-2">
              <Label>Event Types</Label>
              <div className="space-y-2">
                {AVAILABLE_EVENT_TYPES.map((eventType) => (
                  <div key={eventType} className="flex items-center gap-2">
                    <Checkbox
                      id={`edit-event-${eventType}`}
                      checked={editForm.event_types.includes(eventType)}
                      onCheckedChange={() =>
                        setEditForm((prev) => ({
                          ...prev,
                          event_types: prev.event_types.includes(eventType)
                            ? prev.event_types.filter((e) => e !== eventType)
                            : [...prev.event_types, eventType],
                        }))
                      }
                    />
                    <Label
                      htmlFor={`edit-event-${eventType}`}
                      className="font-mono text-sm cursor-pointer"
                    >
                      {eventType}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* Enabled toggle */}
            <div className="flex items-center gap-3">
              <Switch
                id="edit-webhook-enabled"
                checked={editForm.enabled}
                onCheckedChange={(checked) =>
                  setEditForm((prev) => ({ ...prev, enabled: checked }))
                }
              />
              <Label htmlFor="edit-webhook-enabled">Enabled</Label>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={saving || !editForm.url}>
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                "Save Changes"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={!!deleteConfirmWebhook}
        onOpenChange={(open) => {
          if (!open) setDeleteConfirmWebhook(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Webhook</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this webhook? This action cannot be undone and all
              pending deliveries will be cancelled.
            </DialogDescription>
          </DialogHeader>
          {deleteConfirmWebhook && (
            <div className="py-2">
              <p className="font-mono text-sm truncate text-muted-foreground bg-muted px-3 py-2 rounded-md border border-border">
                {deleteConfirmWebhook.url}
              </p>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteConfirmWebhook(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deleteConfirmWebhook && handleDelete(deleteConfirmWebhook.id)}
              disabled={!!deletingId}
            >
              {deletingId ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete Webhook"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Deliveries Dialog */}
      <Dialog
        open={deliveriesDialogOpen}
        onOpenChange={(open) => {
          if (!open) {
            setDeliveries([]);
            setDeliveriesCursor(null);
          }
          setDeliveriesDialogOpen(open);
        }}
      >
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Webhook Deliveries</DialogTitle>
            {selectedWebhook && (
              <DialogDescription className="font-mono text-xs truncate">
                {selectedWebhook.url}
              </DialogDescription>
            )}
          </DialogHeader>

          {loadingDeliveries ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : deliveries.length > 0 ? (
            <>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-8 pl-3 pr-0" />
                      <TableHead>Status</TableHead>
                      <TableHead>HTTP</TableHead>
                      <TableHead className="text-center">Attempts</TableHead>
                      <TableHead>Event</TableHead>
                      <TableHead>Created At</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {deliveries.map((delivery) => (
                      <DeliveryTableRow
                        key={delivery.id}
                        delivery={delivery}
                        formatDate={formatDate}
                      />
                    ))}
                  </TableBody>
                </Table>
              </div>
              {deliveriesCursor && (
                <div className="flex justify-center pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleLoadMoreDeliveries}
                    disabled={loadingMoreDeliveries}
                  >
                    {loadingMoreDeliveries && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Load more
                  </Button>
                </div>
              )}
            </>
          ) : (
            <p className="text-muted-foreground text-center py-8 text-sm">
              No deliveries yet for this webhook.
            </p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
