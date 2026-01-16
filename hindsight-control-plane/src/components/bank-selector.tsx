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
import { Check, ChevronsUpDown, Plus, FileText, Moon, Sun, Github, Tag } from "lucide-react";
import { useTheme } from "@/lib/theme-context";
import Image from "next/image";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
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

  // Document creation state
  const [docDialogOpen, setDocDialogOpen] = React.useState(false);
  const [docContent, setDocContent] = React.useState("");
  const [docContext, setDocContext] = React.useState("");
  const [docEventDate, setDocEventDate] = React.useState("");
  const [docDocumentId, setDocDocumentId] = React.useState("");
  const [docTags, setDocTags] = React.useState("");
  const [docAsync, setDocAsync] = React.useState(false);
  const [isCreatingDoc, setIsCreatingDoc] = React.useState(false);
  const [docError, setDocError] = React.useState<string | null>(null);

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
            <div className="py-4 space-y-4">
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
                  <label className="font-bold block mb-1 text-sm text-foreground">Event Date</label>
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

              {docError && <p className="text-sm text-destructive">{docError}</p>}
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
                  setDocAsync(false);
                  setDocError(null);
                }}
              >
                Cancel
              </Button>
              <Button onClick={handleCreateDocument} disabled={isCreatingDoc || !docContent.trim()}>
                {isCreatingDoc ? "Adding..." : "Add Document"}
              </Button>
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
