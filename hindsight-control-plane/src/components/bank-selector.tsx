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
  Tag,
  Upload,
  X,
} from "lucide-react";
import { useTheme } from "@/lib/theme-context";
import Image from "next/image";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

// Supported text file extensions (matching CLI)
const TEXT_EXTENSIONS = [
  "txt",
  "md",
  "json",
  "yaml",
  "yml",
  "toml",
  "xml",
  "csv",
  "log",
  "rst",
  "adoc",
];
const ACCEPT_FILES = TEXT_EXTENSIONS.map((ext) => `.${ext}`).join(",");

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

  // Document creation state
  const [docDialogOpen, setDocDialogOpen] = React.useState(false);
  const [docTab, setDocTab] = React.useState<"text" | "upload">("text");
  const [docContent, setDocContent] = React.useState("");
  const [docContext, setDocContext] = React.useState("");
  const [docEventDate, setDocEventDate] = React.useState("");
  const [docDocumentId, setDocDocumentId] = React.useState("");
  const [docTags, setDocTags] = React.useState("");
  const [docAsync, setDocAsync] = React.useState(false);
  const [isCreatingDoc, setIsCreatingDoc] = React.useState(false);
  const [docError, setDocError] = React.useState<string | null>(null);

  // File upload state
  const [selectedFiles, setSelectedFiles] = React.useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = React.useState<string>("");
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const sortedBanks = React.useMemo(() => {
    return [...banks].sort((a, b) => a.localeCompare(b));
  }, [banks]);

  const handleCreateBank = async () => {
    if (!newBankId.trim()) return;

    setIsCreating(true);
    setCreateError(null);

    try {
      await client.createBank(newBankId.trim());
      await loadBanks();
      setCreateDialogOpen(false);
      setNewBankId("");
      // Navigate to the new bank
      setCurrentBank(newBankId.trim());
      router.push(`/banks/${newBankId.trim()}?view=data`);
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : "Failed to create bank");
    } finally {
      setIsCreating(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    // Filter to only supported extensions
    const validFiles = files.filter((file) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      return ext && TEXT_EXTENSIONS.includes(ext);
    });
    setSelectedFiles((prev) => [...prev, ...validFiles]);
    // Reset input to allow selecting same file again
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const readFileContent = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = (e) => reject(e);
      reader.readAsText(file);
    });
  };

  const handleUploadFiles = async () => {
    if (!currentBank || selectedFiles.length === 0) return;

    setIsCreatingDoc(true);
    setDocError(null);
    setUploadProgress("");

    try {
      const parsedTags = docTags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      // Read all files and create items
      const items: any[] = [];
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        setUploadProgress(`Reading file ${i + 1}/${selectedFiles.length}: ${file.name}`);
        const content = await readFileContent(file);
        const item: any = {
          content,
          context: `File: ${file.name}`,
        };
        if (parsedTags.length > 0) item.tags = parsedTags;
        items.push(item);
      }

      setUploadProgress(`Uploading ${items.length} file(s)...`);

      const params: any = {
        bank_id: currentBank,
        items,
      };
      if (parsedTags.length > 0) params.document_tags = parsedTags;

      if (docAsync) {
        await client.retain({ ...params, async: true });
      } else {
        await client.retain(params);
      }

      // Reset form and close dialog
      setDocDialogOpen(false);
      setSelectedFiles([]);
      setDocTags("");
      setDocAsync(false);
      setUploadProgress("");

      // Navigate to documents view
      router.push(`/banks/${currentBank}?view=documents`);
    } catch (error) {
      setDocError(error instanceof Error ? error.message : "Failed to upload files");
    } finally {
      setIsCreatingDoc(false);
      setUploadProgress("");
    }
  };

  const handleCreateDocument = async () => {
    if (!currentBank || !docContent.trim()) return;

    setIsCreatingDoc(true);
    setDocError(null);

    try {
      // Parse tags from comma-separated string
      const parsedTags = docTags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const item: any = { content: docContent };
      if (docContext) item.context = docContext;
      // datetime-local gives "2024-01-15T10:30", add seconds for proper ISO format
      if (docEventDate) item.timestamp = docEventDate + ":00";
      if (parsedTags.length > 0) item.tags = parsedTags;

      const params: any = {
        bank_id: currentBank,
        items: [item],
      };

      if (docDocumentId) params.document_id = docDocumentId;
      if (parsedTags.length > 0) params.document_tags = parsedTags;

      if (docAsync) {
        await client.retain({ ...params, async: true });
      } else {
        await client.retain(params);
      }

      // Reset form and close dialog
      setDocDialogOpen(false);
      setDocContent("");
      setDocContext("");
      setDocEventDate("");
      setDocDocumentId("");
      setDocTags("");
      setDocAsync(false);

      // Navigate to documents view to see the new document
      router.push(`/banks/${currentBank}?view=documents`);
    } catch (error) {
      setDocError(error instanceof Error ? error.message : "Failed to create document");
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
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              role="combobox"
              aria-expanded={open}
              className="w-[250px] justify-between font-bold border-2 border-primary hover:bg-accent"
            >
              {currentBank || "Select a memory bank..."}
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
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Create New Memory Bank</DialogTitle>
            </DialogHeader>
            <div className="py-4">
              <Input
                placeholder="Enter bank ID..."
                value={newBankId}
                onChange={(e) => setNewBankId(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isCreating) {
                    handleCreateBank();
                  }
                }}
                autoFocus
              />
              {createError && <p className="text-sm text-destructive mt-2">{createError}</p>}
            </div>
            <DialogFooter>
              <Button
                variant="secondary"
                onClick={() => {
                  setCreateDialogOpen(false);
                  setNewBankId("");
                  setCreateError(null);
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
          <DialogContent className="sm:max-w-[600px]">
            <DialogHeader>
              <DialogTitle>Add New Document</DialogTitle>
              <p className="text-sm text-muted-foreground">
                Add a new document to memory bank:{" "}
                <span className="font-semibold">{currentBank}</span>
              </p>
            </DialogHeader>

            <Tabs value={docTab} onValueChange={(v) => setDocTab(v as "text" | "upload")}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="text" className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Text
                </TabsTrigger>
                <TabsTrigger value="upload" className="flex items-center gap-2">
                  <Upload className="h-4 w-4" />
                  Upload Files
                </TabsTrigger>
              </TabsList>

              <TabsContent value="text" className="space-y-4 mt-4">
                <div>
                  <label className="font-bold block mb-1 text-sm text-foreground">Content *</label>
                  <Textarea
                    value={docContent}
                    onChange={(e) => setDocContent(e.target.value)}
                    placeholder="Enter the document content..."
                    className="min-h-[150px] resize-y"
                    autoFocus
                  />
                </div>

                <div>
                  <label className="font-bold block mb-1 text-sm text-foreground">Context</label>
                  <Input
                    type="text"
                    value={docContext}
                    onChange={(e) => setDocContext(e.target.value)}
                    placeholder="Optional context about this document..."
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
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
                  <label className="font-bold block mb-1 text-sm text-foreground flex items-center gap-2">
                    <Tag className="h-4 w-4" />
                    Tags
                  </label>
                  <Input
                    type="text"
                    value={docTags}
                    onChange={(e) => setDocTags(e.target.value)}
                    placeholder="user_alice, session_123, project_x"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Comma-separated tags for filtering during recall/reflect
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <Checkbox
                    id="async-doc"
                    checked={docAsync}
                    onCheckedChange={(checked) => setDocAsync(checked as boolean)}
                  />
                  <label htmlFor="async-doc" className="text-sm cursor-pointer text-foreground">
                    Process in background (async)
                  </label>
                </div>
              </TabsContent>

              <TabsContent value="upload" className="space-y-4 mt-4">
                <div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept={ACCEPT_FILES}
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
                    <span className="text-xs text-muted-foreground mt-1">
                      Supported: {TEXT_EXTENSIONS.join(", ")}
                    </span>
                  </label>
                </div>

                {selectedFiles.length > 0 && (
                  <div className="space-y-2">
                    <label className="font-bold block text-sm text-foreground">
                      Selected Files ({selectedFiles.length})
                    </label>
                    <div className="max-h-[150px] overflow-y-auto space-y-1">
                      {selectedFiles.map((file, index) => (
                        <div
                          key={`${file.name}-${index}`}
                          className="flex items-center justify-between p-2 bg-muted rounded-md"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <FileText className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                            <span className="text-sm truncate">{file.name}</span>
                            <span className="text-xs text-muted-foreground">
                              ({(file.size / 1024).toFixed(1)} KB)
                            </span>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => removeFile(index)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <label className="font-bold block mb-1 text-sm text-foreground flex items-center gap-2">
                    <Tag className="h-4 w-4" />
                    Tags (applied to all files)
                  </label>
                  <Input
                    type="text"
                    value={docTags}
                    onChange={(e) => setDocTags(e.target.value)}
                    placeholder="user_alice, session_123, project_x"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Comma-separated tags for filtering during recall/reflect
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <Checkbox
                    id="async-upload"
                    checked={docAsync}
                    onCheckedChange={(checked) => setDocAsync(checked as boolean)}
                  />
                  <label htmlFor="async-upload" className="text-sm cursor-pointer text-foreground">
                    Process in background (async)
                  </label>
                </div>

                {uploadProgress && (
                  <p className="text-sm text-muted-foreground">{uploadProgress}</p>
                )}
              </TabsContent>
            </Tabs>

            {docError && <p className="text-sm text-destructive mt-2">{docError}</p>}

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
                  setDocAsync(false);
                  setDocError(null);
                  setSelectedFiles([]);
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
