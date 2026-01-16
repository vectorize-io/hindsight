"use client";

import { useState } from "react";
import { client } from "@/lib/api";
import { useBank } from "@/lib/bank-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Tag } from "lucide-react";

export function AddMemoryView() {
  const { currentBank } = useBank();
  const [content, setContent] = useState("");
  const [context, setContext] = useState("");
  const [eventDate, setEventDate] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [tags, setTags] = useState("");
  const [async, setAsync] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const clearForm = () => {
    setContent("");
    setContext("");
    setEventDate("");
    setDocumentId("");
    setTags("");
    setAsync(false);
    setResult(null);
  };

  const submitMemory = async () => {
    if (!currentBank || !content) {
      alert("Please enter content");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      // Parse tags from comma-separated string
      const parsedTags = tags
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      const item: any = { content };
      if (context) item.context = context;
      // datetime-local gives "2024-01-15T10:30", add seconds for proper ISO format
      if (eventDate) item.timestamp = eventDate + ":00";
      if (parsedTags.length > 0) item.tags = parsedTags;

      const data: any = await client.retain({
        bank_id: currentBank,
        items: [item],
        document_id: documentId,
        async,
        ...(parsedTags.length > 0 && { document_tags: parsedTags }),
      });

      setResult(data.message as string);
      setContent("");
    } catch (error) {
      console.error("Error submitting memory:", error);
      setResult("Error: " + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl">
      <p className="text-muted-foreground mb-4">
        Retain memories to the selected memory bank. You can add one or multiple memories at once.
      </p>

      <div className="max-w-3xl">
        <div className="bg-card p-5 rounded-lg mb-5 border-2 border-primary">
          <h3 className="mt-0 text-card-foreground">Memory Entry</h3>

          <div className="mb-4">
            <label className="font-bold block mb-1 text-card-foreground">Content *</label>
            <Textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Enter the memory content..."
              className="min-h-[100px] resize-y"
            />
          </div>

          <div className="mb-4">
            <label className="font-bold block mb-1 text-card-foreground">Context</label>
            <Input
              type="text"
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="Optional context about this memory..."
            />
          </div>

          <div className="mb-4">
            <label className="font-bold block mb-1 text-card-foreground">Event Date</label>
            <Input
              type="datetime-local"
              value={eventDate}
              onChange={(e) => setEventDate(e.target.value)}
            />
          </div>

          <div className="mb-4">
            <label className="font-bold block mb-1 text-card-foreground">Document ID</label>
            <Input
              type="text"
              value={documentId}
              onChange={(e) => setDocumentId(e.target.value)}
              placeholder="Optional document identifier (automatically upserts if document exists)..."
            />
            <small className="text-muted-foreground text-xs mt-1 block">
              Note: If a document with this ID already exists, it will be automatically replaced
              with the new content.
            </small>
          </div>

          <div className="mb-4">
            <label className="font-bold block mb-1 text-card-foreground flex items-center gap-2">
              <Tag className="h-4 w-4" />
              Tags
            </label>
            <Input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="user_alice, session_123, project_x"
            />
            <small className="text-muted-foreground text-xs mt-1 block">
              Comma-separated tags for filtering during recall/reflect. Tags cannot contain commas.
            </small>
          </div>

          <div className="mb-4">
            <div className="flex items-center gap-2">
              <Checkbox
                id="async"
                checked={async}
                onCheckedChange={(checked) => setAsync(checked as boolean)}
              />
              <label htmlFor="async" className="font-bold text-card-foreground cursor-pointer">
                Async (process in background)
              </label>
            </div>
          </div>

          <div className="flex gap-2.5">
            <Button onClick={submitMemory} disabled={loading}>
              {loading ? "Retaining..." : "Retain Memory"}
            </Button>
            <Button onClick={clearForm} variant="secondary">
              Clear Form
            </Button>
          </div>
        </div>

        {result && (
          <div
            className={`mt-5 p-5 rounded-lg border-2 ${result.startsWith("Error") ? "bg-destructive/10 border-destructive text-destructive" : "bg-primary/10 border-primary text-primary"}`}
          >
            <div className="font-semibold">{result}</div>
          </div>
        )}

        {loading && (
          <div className="text-center py-10 text-muted-foreground">
            <div className="text-5xl mb-2.5">⏳</div>
            <div className="text-lg">Retaining memory...</div>
          </div>
        )}
      </div>
    </div>
  );
}
