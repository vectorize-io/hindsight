"use client";

import * as React from "react";
import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useBank } from "@/lib/bank-context";
import { client } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Check,
  ChevronsUpDown,
  Plus,
  FileText,
  Moon,
  Sun,
  Github,
  Upload,
  X,
  Lock,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useTheme } from "@/lib/theme-context";
import Image from "next/image";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

function BankSelectorInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { currentBank, setCurrentBank, banks, loadBanks } = useBank();
  const { theme, toggleTheme } = useTheme();
  const [open, setOpen] = React.useState(false);
  const [createDialogOpen, setCreateDialogOpen] = React.useState(false);
  const [newBankId, setNewBankId] = React.useState("");
  const [isCreating, setIsCreating] = React.useState(false);
  const [createError, setCreateError] = React.useState<string | null>(null);
  const [useTemplate, setUseTemplate] = React.useState(false);
  const [templateJson, setTemplateJson] = React.useState("");
  const [templateError, setTemplateError] = React.useState<string | null>(null);

  // Document creation state
  const [docDialogOpen, setDocDialogOpen] = React.useState(false);
  const [docTab, setDocTab] = React.useState<"text" | "upload">("text");
  const [docContent, setDocContent] = React.useState("");
  const [docContext, setDocContext] = React.useState("");
  const [docEventDate, setDocEventDate] = React.useState("");
  const [docDocumentId, setDocDocumentId] = React.useState("");
  const [docTags, setDocTags] = React.useState("");
  const [docObservationScopes, setDocObservationScopes] = React.useState<
    "per_tag" | "combined" | "all_combinations" | "custom"
  >("combined");
  const [docObservationScopesCustom, setDocObservationScopesCustom] = React.useState("");
  const [docMetadata, setDocMetadata] = React.useState("");
  const [docEntities, setDocEntities] = React.useState("");
  const [docAdvancedTab, setDocAdvancedTab] = React.useState<"document" | "tags" | "source">(
    "document"
  );
  const [docAsync, setDocAsync] = React.useState(false);
  const [docStrategy, setDocStrategy] = React.useState("");
  const [isCreatingDoc, setIsCreatingDoc] = React.useState(false);

  // Available strategies for the current bank
  const [bankStrategies, setBankStrategies] = React.useState<string[]>([]);
  React.useEffect(() => {
    if (!docDialogOpen || !currentBank) return;
    client
      .getBankConfig(currentBank)
      .then((resp) => {
        const strategies = resp.config?.retain_strategies;
        setBankStrategies(strategies ? Object.keys(strategies) : []);
      })
      .catch(() => setBankStrategies([]));
  }, [docDialogOpen, currentBank]);

  // File upload state
  const [selectedFiles, setSelectedFiles] = React.useState<File[]>([]);
  const [filesMetadata, setFilesMetadata] = React.useState<
    {
      context: string;
      timestamp: string;
      document_id: string;
      tags: string;
      metadata: string;
      strategy: string;
      advancedTab: "document" | "tags" | "source";
      expanded: boolean;
    }[]
  >([]);
  const [uploadProgress, setUploadProgress] = React.useState<string>("");
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // Feature flags
  const [fileUploadEnabled, setFileUploadEnabled] = React.useState<boolean | null>(null);

  // Load feature flags
  React.useEffect(() => {
    client
      .getVersion()
      .then((version) => {
        setFileUploadEnabled(version.features.file_upload_api);
      })
      .catch(() => {
        setFileUploadEnabled(false);
      });
  }, []);

  const sortedBanks = React.useMemo(() => {
    return [...banks].sort((a, b) => a.localeCompare(b));
  }, [banks]);

  const handleCreateBank = async () => {
    if (!newBankId.trim()) return;

    setIsCreating(true);
    setCreateError(null);
    setTemplateError(null);

    try {
      // Create the bank first
      await client.createBank(newBankId.trim());

      // If template JSON is provided, import it
      if (templateJson.trim()) {
        let manifest: Record<string, unknown>;
        try {
          manifest = JSON.parse(templateJson.trim());
        } catch {
          setTemplateError("Invalid JSON. Please check the template syntax.");
          setIsCreating(false);
          return;
        }

        try {
          await client.importBankTemplate(newBankId.trim(), manifest);
        } catch (importError) {
          setTemplateError(
            importError instanceof Error ? importError.message : "Failed to import template"
          );
          setIsCreating(false);
          return;
        }
      }

      await loadBanks();
      setCreateDialogOpen(false);
      setNewBankId("");
      setTemplateJson("");
      setTemplateError(null);
      // Navigate to the new bank
      setCurrentBank(newBankId.trim());
      router.push(`/banks/${newBankId.trim()}?view=data`);
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : "Failed to create bank");
    } finally {
      setIsCreating(false);
    }
  };

  const parseMetadata = (s: string): Record<string, string> | undefined => {
    const result: Record<string, string> = {};
    for (const line of s.split("\n")) {
      const idx = line.indexOf(":");
      if (idx > 0) {
        const key = line.slice(0, idx).trim();
        const val = line.slice(idx + 1).trim();
        if (key) result[key] = val;
      }
    }
    return Object.keys(result).length > 0 ? result : undefined;
  };

  const parseEntities = (s: string) => {
    const items = s
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    if (items.length === 0) return undefined;
    return items.map((t) => ({ text: t }));
  };

  const scopeLabel = (tags: string[]) => tags.join(", ");

  const scopeQuestion = (tags: string[]): string => {
    if (tags.length === 1) return `What happened with ${tags[0]}?`;
    const allButLast = tags.slice(0, -1).join(", ");
    return `What happened with ${allButLast} and ${tags[tags.length - 1]}?`;
  };

  const computeScopes = (
    tags: string[],
    mode: "per_tag" | "combined" | "all_combinations"
  ): string[][] => {
    if (tags.length === 0) return [];
    if (mode === "per_tag") return tags.map((t) => [t]);
    if (mode === "combined") return [tags];
    // all_combinations: every non-empty subset
    const result: string[][] = [];
    for (let size = 1; size <= tags.length; size++) {
      const combine = (start: number, combo: string[]) => {
        if (combo.length === size) {
          result.push([...combo]);
          return;
        }
        for (let i = start; i < tags.length; i++) combine(i + 1, [...combo, tags[i]]);
      };
      combine(0, []);
    }
    return result;
  };

  const emptyFileMeta = () => ({
    context: "",
    timestamp: "",
    document_id: "",
    tags: "",
    metadata: "",
    strategy: "",
    advancedTab: "document" as "document" | "tags" | "source",
    expanded: false,
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setSelectedFiles((prev) => [...prev, ...files]);
    setFilesMetadata((prev) => [...prev, ...files.map(emptyFileMeta)]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setFilesMetadata((prev) => prev.filter((_, i) => i !== index));
  };

  const updateFileMeta = (
    index: number,
    field:
      | "context"
      | "timestamp"
      | "document_id"
      | "tags"
      | "metadata"
      | "strategy"
      | "advancedTab",
    value: string
  ) => {
    setFilesMetadata((prev) => prev.map((m, i) => (i === index ? { ...m, [field]: value } : m)));
  };

  const toggleFileExpanded = (index: number) => {
    setFilesMetadata((prev) =>
      prev.map((m, i) => (i === index ? { ...m, expanded: !m.expanded } : m))
    );
  };

  const handleUploadFiles = async () => {
    if (!currentBank || selectedFiles.length === 0) return;

    setIsCreatingDoc(true);
    setUploadProgress("");

    try {
      setUploadProgress(`Uploading ${selectedFiles.length} file(s)...`);

      const perFileMeta = filesMetadata.map((meta) => ({
        ...(meta.context && { context: meta.context }),
        ...(meta.timestamp && { timestamp: meta.timestamp + ":00" }),
        ...(meta.document_id && { document_id: meta.document_id }),
        ...(meta.tags && {
          tags: meta.tags
            .split(",")
            .map((t) => t.trim())
            .filter(Boolean),
        }),
        ...(meta.metadata && { metadata: parseMetadata(meta.metadata) }),
        ...(meta.strategy && { strategy: meta.strategy }),
      }));

      await client.uploadFiles({
        bank_id: currentBank,
        files: selectedFiles,
        async: true,
        files_metadata: perFileMeta,
      });

      // Reset form and close dialog
      setDocDialogOpen(false);
      setSelectedFiles([]);
      setFilesMetadata([]);
      setDocTags("");
      setDocAsync(false);
      setUploadProgress("");

      // Navigate to documents view
      router.push(`/banks/${currentBank}?view=documents`);
    } catch {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setIsCreatingDoc(false);
      setUploadProgress("");
    }
  };

  const handleCreateDocument = async () => {
    if (!currentBank || !docContent.trim()) return;

    setIsCreatingDoc(true);

    try {
      const parsedTags = docTags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);

      const item: {
        content: string;
        context?: string;
        timestamp?: string;
        document_id?: string;
        tags?: string[];
        observation_scopes?: "per_tag" | "combined" | "all_combinations" | string[][];
        metadata?: Record<string, string>;
        entities?: Array<{ text: string }>;
        strategy?: string;
      } = { content: docContent };
      if (docContext) item.context = docContext;
      if (docEventDate) item.timestamp = docEventDate + ":00";
      if (docDocumentId) item.document_id = docDocumentId;
      if (parsedTags.length > 0) item.tags = parsedTags;
      if (docObservationScopes === "per_tag") {
        item.observation_scopes = "per_tag";
      } else if (docObservationScopes === "combined") {
        item.observation_scopes = "combined";
      } else if (docObservationScopes === "all_combinations") {
        item.observation_scopes = "all_combinations";
      } else if (docObservationScopes === "custom") {
        const customScopes = docObservationScopesCustom
          .split("\n")
          .map((line) =>
            line
              .split(",")
              .map((t) => t.trim())
              .filter(Boolean)
          )
          .filter((scope) => scope.length > 0);
        if (customScopes.length > 0) item.observation_scopes = customScopes;
      }
      const parsedMeta = parseMetadata(docMetadata);
      if (parsedMeta) item.metadata = parsedMeta;
      const parsedEntities = parseEntities(docEntities);
      if (parsedEntities) item.entities = parsedEntities;
      if (docStrategy) item.strategy = docStrategy;

      await client.retain({
        bank_id: currentBank,
        items: [item],
        async: docAsync,
      });

      // Reset form and close dialog
      setDocDialogOpen(false);
      setDocContent("");
      setDocContext("");
      setDocEventDate("");
      setDocDocumentId("");
      setDocTags("");
      setDocObservationScopes("combined");
      setDocObservationScopesCustom("");
      setDocMetadata("");
      setDocEntities("");
      setDocAdvancedTab("document");
      setDocAsync(false);
      setDocStrategy("");

      // Navigate to documents view to see the new document
      router.push(`/banks/${currentBank}?view=documents`);
    } catch {
      // Error toast is shown automatically by the API client interceptor
    } finally {
      setIsCreatingDoc(false);
    }
  };

  return (
    <div className="bg-card text-card-foreground px-5 py-3 border-b-4 border-primary-gradient">
      <div className="flex items-center gap-4 text-sm">
        {/* Logo */}
        <Image
          src="/logo.png"
          alt="Hindsight"
          width={40}
          height={40}
          className="h-10 w-auto"
          unoptimized
        />

        {/* Separator */}
        <div className="h-8 w-px bg-border" />

        {/* Memory Bank Selector */}
        <Popover
          open={open}
          onOpenChange={(isOpen) => {
            setOpen(isOpen);
            if (isOpen) loadBanks();
          }}
        >
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              role="combobox"
              aria-expanded={open}
              className="w-[250px] justify-between font-bold border-2 border-primary hover:bg-accent"
            >
              <span className="truncate">{currentBank || "Select a memory bank..."}</span>
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-0">
            <Command>
              {sortedBanks.length > 0 && <CommandInput placeholder="Search memory banks..." />}
              <CommandList>
                <CommandEmpty>No memory banks yet.</CommandEmpty>
                <CommandGroup>
                  {sortedBanks.map((bank) => (
                    <CommandItem
                      key={bank}
                      value={bank}
                      onSelect={(value) => {
                        setCurrentBank(value);
                        setOpen(false);
                        // Preserve current view and subTab when switching banks
                        const view = searchParams.get("view") || "data";
                        const subTab = searchParams.get("subTab");
                        const queryString = subTab
                          ? `?view=${view}&subTab=${subTab}`
                          : `?view=${view}`;
                        router.push(`/banks/${value}${queryString}`);
                      }}
                    >
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          currentBank === bank ? "opacity-100" : "opacity-0"
                        )}
                      />
                      {bank}
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
              {/* Footer: Create new bank */}
              <div className="border-t border-border p-1">
                <button
                  className="w-full flex items-center gap-2 px-2 py-2 text-sm rounded-md hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
                  onClick={() => {
                    setOpen(false);
                    setCreateDialogOpen(true);
                  }}
                >
                  <Plus className="h-4 w-4" />
                  <span>Create new bank</span>
                </button>
              </div>
            </Command>
          </PopoverContent>
        </Popover>

        {/* Separator */}
        <div className="h-8 w-px bg-border" />

        {/* Add Document Button */}
        {currentBank && (
          <Button
            variant="outline"
            size="sm"
            className="h-9 gap-1.5"
            onClick={() => setDocDialogOpen(true)}
            title="Add document to current bank"
            data-add-document
          >
            <Plus className="h-4 w-4" />
            <span>Add Document</span>
          </Button>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* GitHub Link */}
        <a
          href="https://github.com/vectorize-io/hindsight"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
          title="View on GitHub"
        >
          <Github className="h-5 w-5" />
          <span className="text-sm font-medium">GitHub</span>
        </a>

        {/* Separator */}
        <div className="h-8 w-px bg-border" />

        {/* Dark Mode Toggle */}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          className="h-9 w-9"
          title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
        >
          {theme === "light" ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
        </Button>

        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogContent className="sm:max-w-[550px]">
            <DialogHeader>
              <DialogTitle>Create New Memory Bank</DialogTitle>
            </DialogHeader>
            <div className="py-4 space-y-4">
              <Input
                placeholder="Enter bank ID..."
                value={newBankId}
                onChange={(e) => setNewBankId(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isCreating && !useTemplate) {
                    handleCreateBank();
                  }
                }}
                autoFocus
              />
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Switch
                    checked={useTemplate}
                    onCheckedChange={(checked) => {
                      setUseTemplate(checked);
                      if (!checked) {
                        setTemplateJson("");
                        setTemplateError(null);
                      }
                    }}
                  />
                  <label className="text-sm font-medium">Import from template</label>
                </div>
                {useTemplate && (
                  <a
                    href="https://hindsight.vectorize.io/templates"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Browse templates &rarr;
                  </a>
                )}
              </div>
              {useTemplate && (
                <div>
                  <p className="text-xs text-muted-foreground mb-2">
                    Paste a template manifest JSON to pre-configure the bank with settings, mental
                    models, and directives.
                  </p>
                  <Textarea
                    placeholder='{"version": "1", "bank": {...}, "mental_models": [...]}'
                    value={templateJson}
                    onChange={(e) => {
                      setTemplateJson(e.target.value);
                      setTemplateError(null);
                    }}
                    className="font-mono text-xs min-h-[120px]"
                  />
                </div>
              )}
              {templateError && (
                <p className="text-sm text-destructive whitespace-pre-wrap">{templateError}</p>
              )}
              {createError && <p className="text-sm text-destructive">{createError}</p>}
            </div>
            <DialogFooter>
              <Button
                variant="secondary"
                onClick={() => {
                  setCreateDialogOpen(false);
                  setNewBankId("");
                  setUseTemplate(false);
                  setTemplateJson("");
                  setCreateError(null);
                  setTemplateError(null);
                }}
              >
                Cancel
              </Button>
              <Button onClick={handleCreateBank} disabled={isCreating || !newBankId.trim()}>
                {isCreating ? "Creating..." : "Create"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={docDialogOpen} onOpenChange={setDocDialogOpen}>
          <DialogContent className="sm:max-w-[750px] max-h-[90vh] flex flex-col">
            <DialogHeader>
              <DialogTitle>Add New Document</DialogTitle>
              <p className="text-sm text-muted-foreground">
                Add a new document to memory bank:{" "}
                <span className="font-semibold">{currentBank}</span>
              </p>
            </DialogHeader>

            <div className="space-y-4 overflow-y-auto flex-1 px-1 -mx-1">
              {/* Content — tab-switched input only */}
              <Tabs value={docTab} onValueChange={(v) => setDocTab(v as "text" | "upload")}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="text" className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    Text
                  </TabsTrigger>
                  <TabsTrigger
                    value="upload"
                    className="flex items-center gap-2"
                    disabled={fileUploadEnabled === false}
                  >
                    {fileUploadEnabled === false ? (
                      <Lock className="h-4 w-4" />
                    ) : (
                      <Upload className="h-4 w-4" />
                    )}
                    Upload Files
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="text" className="mt-3">
                  <label className="font-bold block mb-1 text-sm text-foreground">Content *</label>
                  <Textarea
                    value={docContent}
                    onChange={(e) => setDocContent(e.target.value)}
                    placeholder="Enter the document content..."
                    className="min-h-[150px] resize-y"
                    autoFocus
                  />
                </TabsContent>

                <TabsContent value="upload" className="mt-3">
                  {fileUploadEnabled === false ? (
                    <div className="flex flex-col items-center justify-center py-8 text-center space-y-3">
                      <Lock className="h-12 w-12 text-muted-foreground/50" />
                      <div>
                        <p className="font-semibold text-foreground">File Upload API Disabled</p>
                        <p className="text-sm text-muted-foreground mt-1">
                          File upload is not enabled on this server.
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          To enable, set{" "}
                          <code className="bg-muted px-1 py-0.5 rounded">
                            HINDSIGHT_API_ENABLE_FILE_UPLOAD_API=true
                          </code>
                        </p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        onChange={handleFileSelect}
                        className="hidden"
                        id="file-upload"
                      />
                      <label
                        htmlFor="file-upload"
                        className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-muted-foreground/25 rounded-lg cursor-pointer hover:border-primary/50 hover:bg-accent/50 transition-colors"
                      >
                        <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                        <span className="text-sm text-muted-foreground">
                          Click to select files or drag and drop
                        </span>
                      </label>

                      {selectedFiles.length > 0 && (
                        <div className="mt-3 space-y-1">
                          {selectedFiles.map((file, index) => {
                            const meta = filesMetadata[index];
                            const hasData =
                              meta &&
                              (meta.context ||
                                meta.timestamp ||
                                meta.document_id ||
                                meta.tags ||
                                meta.metadata);
                            return (
                              <div
                                key={`${file.name}-${index}`}
                                className="bg-muted rounded-md overflow-hidden"
                              >
                                {/* File row header */}
                                <div className="flex items-center gap-1 px-2 py-2">
                                  <button
                                    type="button"
                                    className="flex items-center gap-1.5 min-w-0 flex-1 text-left hover:opacity-75 transition-opacity"
                                    onClick={() => toggleFileExpanded(index)}
                                    title="Edit metadata for this file"
                                  >
                                    {meta?.expanded ? (
                                      <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
                                    ) : (
                                      <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                                    )}
                                    <FileText
                                      className={`h-4 w-4 shrink-0 ${hasData ? "text-primary" : "text-muted-foreground"}`}
                                    />
                                    <span className="text-sm truncate">{file.name}</span>
                                    <span className="text-xs text-muted-foreground shrink-0">
                                      ({(file.size / 1024).toFixed(1)} KB)
                                    </span>
                                  </button>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-6 w-6 p-0 shrink-0"
                                    onClick={() => removeFile(index)}
                                  >
                                    <X className="h-4 w-4" />
                                  </Button>
                                </div>

                                {/* Per-file metadata form */}
                                {meta?.expanded && (
                                  <div className="border-t border-border/50">
                                    <Tabs
                                      value={meta.advancedTab}
                                      onValueChange={(v) => updateFileMeta(index, "advancedTab", v)}
                                    >
                                      <TabsList className="w-full border-b border-border bg-transparent h-8 p-0 gap-0 justify-start rounded-none">
                                        {(["document", "tags", "source"] as const).map((t) => (
                                          <TabsTrigger
                                            key={t}
                                            value={t}
                                            className="rounded-none h-full px-4 text-xs font-medium bg-transparent shadow-none text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary -mb-px capitalize"
                                          >
                                            {t}
                                          </TabsTrigger>
                                        ))}
                                      </TabsList>
                                      <div className="px-3 py-3 space-y-2">
                                        <TabsContent value="document" className="mt-0 space-y-2">
                                          <div className="grid grid-cols-2 gap-2">
                                            <div>
                                              <label className="font-bold block mb-1 text-sm text-foreground">
                                                Event Date
                                              </label>
                                              <Input
                                                type="datetime-local"
                                                value={meta.timestamp}
                                                onChange={(e) =>
                                                  updateFileMeta(index, "timestamp", e.target.value)
                                                }
                                                className="h-8 text-sm text-foreground"
                                              />
                                            </div>
                                            <div>
                                              <label className="font-bold block mb-1 text-sm text-foreground">
                                                Document ID
                                              </label>
                                              <Input
                                                value={meta.document_id}
                                                onChange={(e) =>
                                                  updateFileMeta(
                                                    index,
                                                    "document_id",
                                                    e.target.value
                                                  )
                                                }
                                                placeholder="Optional ID..."
                                                className="h-8 text-sm"
                                              />
                                            </div>
                                          </div>
                                          <div>
                                            <label className="font-bold block mb-1 text-sm text-foreground">
                                              Strategy
                                            </label>
                                            {bankStrategies.length > 0 ? (
                                              <Select
                                                value={meta.strategy || "__none__"}
                                                onValueChange={(v) =>
                                                  updateFileMeta(
                                                    index,
                                                    "strategy",
                                                    v === "__none__" ? "" : v
                                                  )
                                                }
                                              >
                                                <SelectTrigger className="w-full h-8 text-sm">
                                                  <SelectValue />
                                                </SelectTrigger>
                                                <SelectContent>
                                                  <SelectItem value="__none__">
                                                    <span className="text-muted-foreground italic">
                                                      Default
                                                    </span>
                                                  </SelectItem>
                                                  {bankStrategies.map((name) => (
                                                    <SelectItem key={name} value={name}>
                                                      {name}
                                                    </SelectItem>
                                                  ))}
                                                </SelectContent>
                                              </Select>
                                            ) : (
                                              <Input
                                                value={meta.strategy}
                                                onChange={(e) =>
                                                  updateFileMeta(index, "strategy", e.target.value)
                                                }
                                                placeholder="Strategy name (optional)..."
                                                className="h-8 text-sm"
                                              />
                                            )}
                                          </div>
                                        </TabsContent>
                                        <TabsContent value="tags" className="mt-0 space-y-2">
                                          <div>
                                            <label className="font-bold block mb-1 text-sm text-foreground">
                                              Tags
                                            </label>
                                            <Input
                                              value={meta.tags}
                                              onChange={(e) =>
                                                updateFileMeta(index, "tags", e.target.value)
                                              }
                                              placeholder="tag1, tag2..."
                                              className="h-8 text-sm"
                                            />
                                            <p className="text-xs text-muted-foreground mt-1">
                                              Comma-separated — used to filter memories during
                                              recall/reflect
                                            </p>
                                          </div>
                                        </TabsContent>
                                        <TabsContent value="source" className="mt-0 space-y-2">
                                          <div>
                                            <label className="font-bold block mb-1 text-sm text-foreground">
                                              Context
                                            </label>
                                            <Input
                                              value={meta.context}
                                              onChange={(e) =>
                                                updateFileMeta(index, "context", e.target.value)
                                              }
                                              placeholder="Optional context..."
                                              className="h-8 text-sm"
                                            />
                                          </div>
                                          <div>
                                            <label className="font-bold block mb-1 text-sm text-foreground">
                                              Metadata
                                            </label>
                                            <Textarea
                                              value={meta.metadata}
                                              onChange={(e) =>
                                                updateFileMeta(index, "metadata", e.target.value)
                                              }
                                              placeholder={"source: slack\nchannel: engineering"}
                                              className="min-h-[52px] resize-y font-mono text-sm"
                                            />
                                          </div>
                                        </TabsContent>
                                      </div>
                                    </Tabs>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {uploadProgress && (
                        <p className="text-sm text-muted-foreground mt-2">{uploadProgress}</p>
                      )}
                    </>
                  )}
                </TabsContent>
              </Tabs>

              {/* Context — text tab only */}
              {docTab === "text" && (
                <div>
                  <label className="font-bold block mb-1 text-sm text-foreground">Context</label>
                  <Input
                    type="text"
                    value={docContext}
                    onChange={(e) => setDocContext(e.target.value)}
                    placeholder="Optional context about this document..."
                  />
                </div>
              )}

              {/* Advanced section — text only */}
              {docTab === "text" && (
                <div>
                  <Tabs
                    value={docAdvancedTab}
                    onValueChange={(v) => setDocAdvancedTab(v as "document" | "tags" | "source")}
                  >
                    <TabsList className="w-full border-b border-border bg-transparent h-8 p-0 gap-0 justify-start rounded-none">
                      <TabsTrigger
                        value="document"
                        className="rounded-none h-full px-4 text-xs font-medium bg-transparent shadow-none text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary -mb-px"
                      >
                        Document
                      </TabsTrigger>
                      <TabsTrigger
                        value="tags"
                        className="rounded-none h-full px-4 text-xs font-medium bg-transparent shadow-none text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary -mb-px"
                      >
                        Tags
                      </TabsTrigger>
                      <TabsTrigger
                        value="source"
                        className="rounded-none h-full px-4 text-xs font-medium bg-transparent shadow-none text-muted-foreground hover:text-foreground data-[state=active]:text-foreground data-[state=active]:shadow-none data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary -mb-px"
                      >
                        Source
                      </TabsTrigger>
                    </TabsList>

                    <div className="pt-3 space-y-3">
                      <TabsContent value="document" className="mt-0 space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="font-bold block mb-1 text-sm text-foreground">
                              Event Date
                            </label>
                            <Input
                              type="datetime-local"
                              value={docEventDate}
                              onChange={(e) => setDocEventDate(e.target.value)}
                              className="text-foreground"
                            />
                          </div>
                          <div>
                            <label className="font-bold block mb-1 text-sm text-foreground">
                              Document ID
                            </label>
                            <Input
                              type="text"
                              value={docDocumentId}
                              onChange={(e) => setDocDocumentId(e.target.value)}
                              placeholder="Optional document identifier..."
                            />
                          </div>
                        </div>
                        <div>
                          <label className="font-bold block mb-1 text-sm text-foreground">
                            Strategy
                          </label>
                          {bankStrategies.length > 0 ? (
                            <Select
                              value={docStrategy || "__none__"}
                              onValueChange={(v) => setDocStrategy(v === "__none__" ? "" : v)}
                            >
                              <SelectTrigger className="w-full">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="__none__">
                                  <span className="text-muted-foreground italic">Default</span>
                                </SelectItem>
                                {bankStrategies.map((name) => (
                                  <SelectItem key={name} value={name}>
                                    {name}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          ) : (
                            <Input
                              type="text"
                              value={docStrategy}
                              onChange={(e) => setDocStrategy(e.target.value)}
                              placeholder="Strategy name (optional)..."
                            />
                          )}
                          <p className="text-xs text-muted-foreground mt-1">
                            Override the bank&apos;s default extraction strategy for this document.
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Checkbox
                            id="async-doc"
                            checked={docAsync}
                            onCheckedChange={(checked) => setDocAsync(checked as boolean)}
                          />
                          <label
                            htmlFor="async-doc"
                            className="text-sm cursor-pointer text-foreground"
                          >
                            Process in background (async)
                          </label>
                        </div>
                      </TabsContent>

                      <TabsContent value="tags" className="mt-0 space-y-3">
                        <div>
                          <label className="font-bold block mb-1 text-sm text-foreground">
                            Tags
                          </label>
                          <Input
                            type="text"
                            value={docTags}
                            onChange={(e) => setDocTags(e.target.value)}
                            placeholder="user_alice, session_123, project_x"
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            Comma-separated — used to filter memories during recall/reflect
                          </p>
                        </div>
                        <div>
                          <label className="font-bold block mb-1 text-sm text-foreground">
                            Observation Scopes
                          </label>
                          <Select
                            value={docObservationScopes}
                            onValueChange={(v) =>
                              setDocObservationScopes(
                                v as "per_tag" | "combined" | "all_combinations" | "custom"
                              )
                            }
                          >
                            <SelectTrigger className="w-full">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="per_tag">Per tag</SelectItem>
                              <SelectItem value="combined">Combined</SelectItem>
                              <SelectItem value="all_combinations">All combinations</SelectItem>
                              <SelectItem value="custom">Custom</SelectItem>
                            </SelectContent>
                          </Select>
                          {docObservationScopes !== "custom" &&
                            (() => {
                              const tags = docTags
                                .split(",")
                                .map((t) => t.trim())
                                .filter(Boolean);
                              const scopes = computeScopes(tags, docObservationScopes);
                              const MAX = 6;
                              if (tags.length === 0) {
                                return (
                                  <p className="text-xs text-muted-foreground/60 mt-1.5 italic">
                                    Add tags above to preview observation scopes
                                  </p>
                                );
                              }
                              return (
                                <ul className="mt-2 space-y-1.5">
                                  {scopes.slice(0, MAX).map((scope, i) => (
                                    <li key={i} className="flex flex-col gap-0.5">
                                      <span className="text-xs font-mono text-foreground">
                                        {scopeLabel(scope)}
                                      </span>
                                      <span className="text-xs text-muted-foreground">
                                        {scopeQuestion(scope)}
                                      </span>
                                    </li>
                                  ))}
                                  {scopes.length > MAX && (
                                    <li className="text-xs text-muted-foreground">
                                      +{scopes.length - MAX} more scopes
                                    </li>
                                  )}
                                </ul>
                              );
                            })()}
                          {docObservationScopes === "custom" && (
                            <Textarea
                              value={docObservationScopesCustom}
                              onChange={(e) => setDocObservationScopesCustom(e.target.value)}
                              placeholder={"user:alice\nuser:alice, place:online"}
                              className="min-h-[72px] resize-y font-mono text-sm mt-2"
                            />
                          )}
                        </div>
                      </TabsContent>

                      <TabsContent value="source" className="mt-0 space-y-3">
                        <div>
                          <label className="font-bold block mb-1 text-sm text-foreground">
                            Metadata
                          </label>
                          <Textarea
                            value={docMetadata}
                            onChange={(e) => setDocMetadata(e.target.value)}
                            placeholder={"source: slack\nchannel: engineering"}
                            className="min-h-[72px] resize-y font-mono text-sm"
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            One <code className="bg-muted px-0.5 rounded">key: value</code> per line
                          </p>
                        </div>
                        <div>
                          <label className="font-bold block mb-1 text-sm text-foreground">
                            Entities
                          </label>
                          <Input
                            type="text"
                            value={docEntities}
                            onChange={(e) => setDocEntities(e.target.value)}
                            placeholder="Alice, Google, ML model"
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            Comma-separated hints merged with auto-extracted entities
                          </p>
                        </div>
                      </TabsContent>
                    </div>
                  </Tabs>
                </div>
              )}
            </div>

            <DialogFooter>
              <Button
                variant="secondary"
                onClick={() => {
                  setDocDialogOpen(false);
                  setDocContent("");
                  setDocContext("");
                  setDocEventDate("");
                  setDocDocumentId("");
                  setDocTags("");
                  setDocObservationScopes("combined");
                  setDocObservationScopesCustom("");
                  setDocMetadata("");
                  setDocEntities("");
                  setDocAdvancedTab("document");
                  setDocAsync(false);
                  setSelectedFiles([]);
                  setFilesMetadata([]);
                  setUploadProgress("");
                }}
              >
                Cancel
              </Button>
              {docTab === "text" ? (
                <Button
                  onClick={handleCreateDocument}
                  disabled={isCreatingDoc || !docContent.trim()}
                >
                  {isCreatingDoc ? "Adding..." : "Add Document"}
                </Button>
              ) : (
                <Button
                  onClick={handleUploadFiles}
                  disabled={isCreatingDoc || selectedFiles.length === 0}
                >
                  {isCreatingDoc
                    ? uploadProgress || "Uploading..."
                    : `Upload ${selectedFiles.length} File${selectedFiles.length !== 1 ? "s" : ""}`}
                </Button>
              )}
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}

export function BankSelector() {
  return (
    <Suspense
      fallback={
        <div className="bg-card text-card-foreground px-5 py-3 border-b-4 border-primary-gradient">
          <div className="flex items-center gap-4 text-sm">
            <Image
              src="/logo.png"
              alt="Hindsight"
              width={40}
              height={40}
              className="h-10 w-auto"
              unoptimized
            />
            <div className="h-8 w-px bg-border" />
            <Button
              variant="outline"
              className="w-[250px] justify-between font-bold border-2 border-primary"
              disabled
            >
              Loading...
              <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>
            <div className="flex-1" />
            <a
              href="https://github.com/vectorize-io/hindsight"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-accent transition-colors text-muted-foreground"
            >
              <Github className="h-5 w-5" />
              <span className="text-sm font-medium">GitHub</span>
            </a>
            <div className="h-8 w-px bg-border" />
            <Button variant="ghost" size="icon" className="h-9 w-9" disabled>
              <Moon className="h-5 w-5" />
            </Button>
          </div>
        </div>
      }
    >
      <BankSelectorInner />
    </Suspense>
  );
}
