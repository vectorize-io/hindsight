"use client";

import { useState, useEffect } from "react";
import { useBank } from "@/lib/bank-context";
import { client } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Loader2, AlertCircle, CheckCircle2, Pencil, RotateCcw, MoreVertical } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

// Field metadata for UI rendering
const FIELD_CATEGORIES = {
  retention: {
    title: "Retention Settings",
    description: "Control how memories are extracted and stored",
    fields: {
      retain_chunk_size: {
        label: "Chunk Size",
        type: "number",
        description: "Size of text chunks for processing (tokens)",
        min: 500,
        max: 8000,
      },
      retain_extraction_mode: {
        label: "Extraction Mode",
        type: "select",
        description: "How to extract facts from content",
        options: ["concise", "verbose", "custom"],
      },
      retain_custom_instructions: {
        label: "Custom Instructions",
        type: "textarea",
        description:
          "Custom instructions for fact extraction (requires retain_extraction_mode='custom')",
        placeholder: "Focus on technical details and implementation specifics...",
        rows: 3,
      },
    },
  },
  consolidation: {
    title: "Consolidation Settings",
    description: "Control observation synthesis",
    fields: {
      enable_observations: {
        label: "Enable Observations",
        type: "boolean",
        description: "Enable automatic consolidation of facts into observations",
      },
    },
  },
};

export function BankConfigView() {
  const { currentBank: bankId } = useBank();
  const [loading, setLoading] = useState(true);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [overrides, setOverrides] = useState<Record<string, any>>({});
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    if (bankId) {
      loadConfig();
    }
  }, [bankId]);

  const loadConfig = async () => {
    if (!bankId) return;

    setLoading(true);
    try {
      const response = await client.getBankConfig(bankId);
      setConfig(response.config);
      setOverrides(response.overrides);
    } catch (err: any) {
      console.error("Failed to load config:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setShowResetDialog(true);
  };

  const confirmReset = async () => {
    if (!bankId) return;

    setResetting(true);
    try {
      await client.resetBankConfig(bankId);
      await loadConfig();
      setShowResetDialog(false);
    } catch (err: any) {
      console.error("Failed to reset config:", err);
      alert("Error resetting config: " + err.message);
    } finally {
      setResetting(false);
    }
  };

  const renderReadOnlyField = (fieldKey: string, fieldMeta: any) => {
    const value = config[fieldKey];

    return (
      <div
        key={fieldKey}
        className="flex items-start justify-between gap-4 p-3 border border-border rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors"
      >
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium font-mono">{fieldKey}</div>
          {fieldMeta.description && (
            <p className="text-xs text-muted-foreground mt-0.5">{fieldMeta.description}</p>
          )}
        </div>
        <div className="text-sm text-foreground font-mono flex-shrink-0">
          {fieldMeta.type === "boolean" ? (
            <span className={value ? "text-green-600" : "text-muted-foreground"}>
              {value ? "Enabled" : "Disabled"}
            </span>
          ) : fieldMeta.type === "textarea" ? (
            <span className="text-muted-foreground italic">
              {value ? `${value.substring(0, 50)}${value.length > 50 ? "..." : ""}` : "Not set"}
            </span>
          ) : (
            value || <span className="text-muted-foreground italic">Not set</span>
          )}
        </div>
      </div>
    );
  };

  if (!bankId) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-muted-foreground">No bank selected</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">Configuration Settings</CardTitle>
              <CardDescription className="text-xs">
                Behavioral parameters for this memory bank
              </CardDescription>
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" disabled={resetting}>
                  {resetting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <MoreVertical className="h-4 w-4" />
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => setShowEditDialog(true)}>
                  <Pencil className="h-4 w-4 mr-2" />
                  Edit
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleReset}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Reset to Defaults
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {Object.entries(FIELD_CATEGORIES).map(([catKey, category]) => (
            <div key={catKey}>
              <div className="mb-3">
                <h3 className="text-sm font-semibold">{category.title}</h3>
                <p className="text-xs text-muted-foreground">{category.description}</p>
              </div>
              <div className="grid grid-cols-2 gap-x-8 gap-y-4">
                {Object.entries(category.fields).map(([fieldKey, fieldMeta]) =>
                  renderReadOnlyField(fieldKey, fieldMeta)
                )}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {showEditDialog && (
        <ConfigEditDialog
          bankId={bankId}
          initialConfig={config}
          overrides={overrides}
          onClose={() => setShowEditDialog(false)}
          onSaved={() => {
            loadConfig();
            setShowEditDialog(false);
          }}
        />
      )}

      <AlertDialog open={showResetDialog} onOpenChange={setShowResetDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Reset Configuration</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to reset all configuration overrides to defaults? This action
              cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={resetting}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmReset} disabled={resetting}>
              {resetting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Resetting...
                </>
              ) : (
                "Reset to Defaults"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

// Edit dialog component
function ConfigEditDialog({
  bankId,
  initialConfig,
  overrides,
  onClose,
  onSaved,
}: {
  bankId: string;
  initialConfig: Record<string, any>;
  overrides: Record<string, any>;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState(initialConfig);

  const handleFieldChange = (field: string, value: any) => {
    setConfig({ ...config, [field]: value });
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const updates: Record<string, any> = {};
      Object.keys(config).forEach((key) => {
        const isConfigurable = Object.values(FIELD_CATEGORIES).some((cat) =>
          Object.keys(cat.fields).includes(key)
        );
        if (isConfigurable) {
          updates[key] = config[key];
        }
      });

      await client.updateBankConfig(bankId, updates);
      onSaved();
    } catch (err: any) {
      console.error("Failed to save config:", err);
      setError(err.message || "Failed to save configuration");
      setSaving(false);
    }
  };

  const renderField = (fieldKey: string, fieldMeta: any) => {
    const value = config[fieldKey];

    if (fieldMeta.type === "boolean") {
      return (
        <div key={fieldKey} className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor={fieldKey} className="font-mono">
                {fieldKey}
              </Label>
              {fieldMeta.description && (
                <p className="text-xs text-muted-foreground mt-1">{fieldMeta.description}</p>
              )}
            </div>
            <button
              onClick={() => handleFieldChange(fieldKey, !value)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                value ? "bg-primary" : "bg-muted"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  value ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>
        </div>
      );
    }

    if (fieldMeta.type === "select") {
      return (
        <div key={fieldKey} className="space-y-2">
          <Label htmlFor={fieldKey} className="font-mono">
            {fieldKey}
          </Label>
          {fieldMeta.description && (
            <p className="text-xs text-muted-foreground mt-1">{fieldMeta.description}</p>
          )}
          <Select
            value={value?.toString()}
            onValueChange={(val) => handleFieldChange(fieldKey, val)}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {fieldMeta.options.map((opt: string) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      );
    }

    if (fieldMeta.type === "textarea") {
      return (
        <div key={fieldKey} className="space-y-2">
          <Label htmlFor={fieldKey} className="font-mono">
            {fieldKey}
          </Label>
          {fieldMeta.description && (
            <p className="text-xs text-muted-foreground mt-1">{fieldMeta.description}</p>
          )}
          <Textarea
            id={fieldKey}
            value={value || ""}
            onChange={(e) => handleFieldChange(fieldKey, e.target.value || null)}
            placeholder={fieldMeta.placeholder}
            rows={fieldMeta.rows || 3}
            className="font-mono text-sm"
          />
        </div>
      );
    }

    // number or text
    return (
      <div key={fieldKey} className="space-y-2">
        <Label htmlFor={fieldKey} className="font-mono">
          {fieldKey}
        </Label>
        {fieldMeta.description && (
          <p className="text-xs text-muted-foreground mt-1">{fieldMeta.description}</p>
        )}
        <Input
          id={fieldKey}
          type={fieldMeta.type || "text"}
          value={value ?? ""}
          onChange={(e) =>
            handleFieldChange(
              fieldKey,
              fieldMeta.type === "number" ? parseFloat(e.target.value) : e.target.value
            )
          }
          min={fieldMeta.min}
          max={fieldMeta.max}
        />
      </div>
    );
  };

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit Configuration</DialogTitle>
          <DialogDescription>
            Customize behavioral settings for this bank. Changes only affect this bank and override
            global defaults.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {Object.entries(FIELD_CATEGORIES).map(([catKey, category]) => (
            <div key={catKey} className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold">{category.title}</h3>
                <p className="text-xs text-muted-foreground">{category.description}</p>
              </div>
              <div className="grid gap-4">
                {Object.entries(category.fields).map(([fieldKey, fieldMeta]) =>
                  renderField(fieldKey, fieldMeta)
                )}
              </div>
            </div>
          ))}
        </div>

        <DialogFooter>
          <Button onClick={onClose} variant="outline" disabled={saving}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              "Save Changes"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
