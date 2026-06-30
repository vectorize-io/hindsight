"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import { useBank } from "@/lib/bank-context";
import { client } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Send,
  Sparkles,
  Loader2,
  Brain,
  Database,
  Cpu,
  AlertCircle,
  CheckCircle2,
  Bot,
  User,
  Clock,
  RefreshCw,
  Shield,
  Search,
  Zap,
  X,
  Maximize2,
  Minimize2,
  Plus,
  MessageSquare,
  Settings2,
  Trash2,
  SlidersHorizontal,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ── Types ──

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  tool_calls?: Array<{
    name: string;
    args?: any;
    result?: string;
    status: "pending" | "success" | "error";
  }>;
  based_on?: any;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

interface ModelInfo {
  id: string;
  provider_id?: string;
  model_id?: string;
  display_name?: string;
  name?: string;
}

interface ChatAssistantProps {
  /** If true, renders inside a full-height container for the standalone /chat page */
  standalone?: boolean;
  /** If true, shows compact header/title */
  compact?: boolean;
  /** Optional placeholder message */
  placeholder?: string;
  /** Initial messages to pre-populate */
  initialMessages?: Message[];
  /** External context to include in messages */
  context?: string;
}

// ── Helpers ──

function newConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    messages: [],
  };
}

function formatModelName(m: ModelInfo): string {
  return m.display_name || m.name || m.model_id || m.id || "Unknown";
}

function formatModelValue(m: ModelInfo): string {
  return `${m.provider_id || "unknown"}/${m.model_id || m.id || "unknown"}`;
}

// ── Component ──

export function ChatAssistant({
  standalone = false,
  compact = false,
  placeholder = "Ask anything — recall memories, reflect on patterns, manage banks...",
  initialMessages = [],
}: ChatAssistantProps) {
  const t = useTranslations("operator");
  const { currentBank } = useBank();

  // ── Core state (existing) ──
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  // ── Conversation state (new) ──
  const [conversations, setConversations] = useState<Conversation[]>([
    newConversation(),
  ]);
  const [activeConversationId, setActiveConversationId] = useState<string>(
    conversations[0].id
  );
  const [showSidebar, setShowSidebar] = useState(true);

  // ── Model & parameter state (new) ──
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [showSettings, setShowSettings] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Derive active conversation
  const activeConv = conversations.find((c) => c.id === activeConversationId);
  // Sync messages with active conversation
  const displayMessages = activeConv?.messages || messages;

  // ── Auto-scroll ──
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayMessages]);

  // ── Focus input on mount for standalone mode ──
  useEffect(() => {
    if (standalone) {
      inputRef.current?.focus();
    }
  }, [standalone]);

  // ── Fetch available models ──
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/chat/models");
        if (!res.ok) return;
        const data = await res.json();
        // Handle various response shapes
        const list: ModelInfo[] =
          data.data || data.models || data.model || [];
        setModels(list);
        if (list.length > 0 && !selectedModel) {
          setSelectedModel(formatModelValue(list[0]));
        }
      } catch {
        // Models unavailable — non-critical
      }
    })();
  }, []);

  // ── Conversation helpers ──
  const switchConversation = useCallback((id: string) => {
    setActiveConversationId(id);
    setInput("");
  }, []);

  const createNewConversation = useCallback(() => {
    const c = newConversation();
    setConversations((prev) => [c, ...prev]);
    setActiveConversationId(c.id);
  }, []);

  const deleteConversation = useCallback(
    (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setConversations((prev) => {
        const filtered = prev.filter((c) => c.id !== id);
        if (filtered.length === 0) {
          const fresh = newConversation();
          setActiveConversationId(fresh.id);
          return [fresh];
        }
        if (activeConversationId === id) {
          setActiveConversationId(filtered[0].id);
        }
        return filtered;
      });
    },
    [activeConversationId]
  );

  const updateConversationMessages = useCallback(
    (convId: string, updater: (msgs: Message[]) => Message[]) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId ? { ...c, messages: updater(c.messages) } : c
        )
      );
    },
    []
  );

  // ── Send message (streaming) ──
  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const convId = activeConversationId;
    setInput("");

    const userMsg: Message = {
      id: `msg-${Date.now()}-user`,
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    // Update conversation title if first message
    updateConversationMessages(convId, (msgs) => {
      const updated = [...msgs, userMsg];
      // Also update title in conversation list
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId && c.messages.length === 0
            ? { ...c, title: text.slice(0, 40) }
            : c
        )
      );
      return updated;
    });

    // Add empty assistant placeholder
    updateConversationMessages(convId, (msgs) => [
      ...msgs,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        tool_calls: [],
      },
    ]);

    setLoading(true);

    // Declare assistantId outside try/catch for scope access
    const assistantId = `msg-${Date.now()}-assistant`;

    try {
      // Step 1: Gather context from Hindsight reflect API (memory-aware)
      const bankContext = currentBank
        ? `\n[Active bank: ${currentBank}]`
        : "\n[No bank selected — operating globally]";

      let memoryContext = "";
      try {
        const reflectParams: any = {
          query: text,
          budget: "low" as any,
          max_tokens: 2048,
        };
        if (currentBank) {
          reflectParams.bank_id = currentBank;
        }
        const reflectData: any = await client.reflect(reflectParams);
        if (reflectData?.text) {
          memoryContext = `\n[Memory context from Hindsight]:\n${reflectData.text.slice(0, 1500)}`;
        }
      } catch {
        memoryContext = "\n[Memory context unavailable]";
      }

      // Step 2: Build messages for LLM proxy
      const systemMessage = {
        role: "system",
        content: `You are the CollabMind AI Assistant — the operator's direct interface to intelligence.
You have access to the user's memory banks via Hindsight.
Use the memory context provided to give personalized, informed responses.
Be concise, technical, and precise. When you don't know something, say so.
Current bank: ${currentBank || "none (global)"}`,
      };

      // Recent conversation (last 8 messages max) from this conversation
      const currentConvMessages =
        conversations.find((c) => c.id === convId)?.messages || [];
      const recentConvMessages = [...currentConvMessages, userMsg].slice(-8);
      const conversationMessages = recentConvMessages.map((m) => ({
        role: m.role === "user" ? "user" : ("assistant" as const),
        content: m.content,
      }));

      const llmMessages = [
        systemMessage,
        ...(memoryContext
          ? [{ role: "system" as const, content: memoryContext }]
          : []),
        ...conversationMessages,
      ];

      // Parse selected provider/model
      const [selectedProvider, selectedModelId] = selectedModel.split("/");

      // Step 3: Call AI proxy via our API route with streaming
      const proxyRes = await fetch("/api/chat/proxy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: llmMessages,
          model: selectedModelId || "auto",
          provider: selectedProvider || undefined,
          max_tokens: maxTokens,
          temperature,
          stream: true,
        }),
      });

      if (!proxyRes.ok) {
        const errorData = await proxyRes.json().catch(() => ({}));
        throw new Error(
          errorData.error || `Proxy returned ${proxyRes.status}`
        );
      }

      // Step 4: Handle SSE stream
      const reader = proxyRes.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim() || !line.startsWith("data: ")) continue;

          const data = line.slice(6); // Remove 'data: ' prefix
          if (data === "[DONE]") break;

          try {
            const parsed = JSON.parse(data);

            if (parsed.error) {
              throw new Error(parsed.error);
            }

            // OpenAI-compatible streaming: choices[0].delta.content
            const chunk =
              parsed.choices?.[0]?.delta?.content ||
              parsed.choices?.[0]?.text ||
              parsed.content ||
              "";

            if (chunk) {
              fullContent += chunk;
              // Progressively update assistant message
              updateConversationMessages(convId, (msgs) =>
                msgs.map((m) =>
                  m.id === assistantId ? { ...m, content: fullContent } : m
                )
              );
            }
          } catch (e) {
            // Skip unparseable chunks
            if (e instanceof Error && e.message !== "Unexpected end of JSON input") {
              console.error("SSE parse error:", e);
            }
          }
        }
      }
    } catch (error: any) {
      console.error("Chat error:", error);

      // Fallback to reflect-only mode if proxy fails
      try {
        const fallbackParams: any = {
          query: text,
          budget: "mid" as any,
          max_tokens: 4096,
        };
        if (currentBank) {
          fallbackParams.bank_id = currentBank;
        }
        const fallbackData: any = await client.reflect(fallbackParams);
        updateConversationMessages(convId, (msgs) =>
          msgs.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content:
                    fallbackData?.text ||
                    fallbackData?.answer ||
                    "I analyzed your request. (No response generated)",
                  based_on: fallbackData?.based_on,
                }
              : m
          )
        );
      } catch {
        updateConversationMessages(convId, (msgs) =>
          msgs.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: `**Error**: ${error?.message || "Failed to get response. Is the AI proxy or Hindsight API running?"}`,
                  role: "system",
                }
              : m
          )
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearConversation = () => {
    const convId = activeConversationId;
    updateConversationMessages(convId, () => []);
    setInput("");
    inputRef.current?.focus();
  };

  // ── Render helpers ──

  const containerClass = standalone
    ? "flex flex-col h-full"
    : "flex flex-col";

  const messagesClass = standalone
    ? "flex-1 overflow-y-auto px-4 py-4 space-y-4"
    : compact
      ? "overflow-y-auto px-2 py-2 space-y-2 max-h-[500px]"
      : "flex-1 overflow-y-auto px-4 py-4 space-y-4";

  // ── Render ──

  return (
    <div className={containerClass}>
      {/* Chat header (compact mode only) */}
      {compact && (
        <div className="flex items-center justify-between px-3 py-2 border-b border-border">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">AI Assistant</span>
            {currentBank && (
              <Badge variant="outline" className="text-[10px] h-5">
                {currentBank}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => setExpanded(!expanded)}
              title={expanded ? "Minimize" : "Expand"}
            >
              {expanded ? (
                <Minimize2 className="h-3 w-3" />
              ) : (
                <Maximize2 className="h-3 w-3" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={clearConversation}
              title="Clear conversation"
            >
              <RefreshCw className="h-3 w-3" />
            </Button>
          </div>
        </div>
      )}

      {/* Main layout — sidebar + chat area (standalone only) */}
      <div
        className={
          standalone
            ? "flex flex-1 overflow-hidden"
            : "flex flex-1 overflow-hidden"
        }
      >
        {/* ── Conversation Sidebar (standalone mode) ── */}
        {standalone && showSidebar && (
          <div className="w-[220px] flex-shrink-0 border-r border-border bg-card flex flex-col overflow-hidden">
            <div className="p-3 border-b border-border">
              <Button
                variant="outline"
                size="sm"
                className="w-full text-xs gap-1"
                onClick={createNewConversation}
              >
                <Plus className="h-3 w-3" />
                New conversation
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {conversations.map((c) => (
                <div
                  key={c.id}
                  onClick={() => switchConversation(c.id)}
                  className={`group flex items-center justify-between px-3 py-2 rounded-md cursor-pointer text-xs transition-colors ${
                    c.id === activeConversationId
                      ? "bg-primary/10 text-primary border-l-2 border-primary"
                      : "text-muted-foreground hover:bg-accent/50 border-l-2 border-transparent"
                  }`}
                >
                  <span className="truncate flex-1">{c.title}</span>
                  <button
                    onClick={(e) => deleteConversation(c.id, e)}
                    className="opacity-0 group-hover:opacity-100 hover:text-destructive transition-opacity"
                    title="Delete conversation"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Chat Area ── */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Messages */}
          <div className={messagesClass}>
            {displayMessages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center py-12 text-muted-foreground">
                <Brain className="h-12 w-12 mb-4 text-primary/30" />
                <h3 className="text-lg font-medium mb-2 text-foreground">
                  CollabMind AI Assistant
                </h3>
                <p className="text-sm max-w-md mb-6">{placeholder}</p>

                {/* Suggested queries */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full">
                  {[
                    {
                      label: "🧠 What patterns do you see in my work?",
                      query:
                        "Based on my memory banks, what patterns and preferences do you see in how I work?",
                    },
                    {
                      label: "📊 System health check",
                      query:
                        "What is the current health status of all system services?",
                    },
                    {
                      label: "🏦 Memory bank overview",
                      query:
                        "Summarize my memory banks and what knowledge is stored in each one",
                    },
                    {
                      label: "⚡ Recent activity",
                      query:
                        "What has been happening recently? Show me recent operations and changes",
                    },
                  ].map((suggestion) => (
                    <Button
                      key={suggestion.label}
                      variant="outline"
                      className="text-xs h-auto py-2 px-3 justify-start text-left leading-snug hover:bg-accent/50"
                      onClick={() => {
                        setInput(suggestion.query);
                        setTimeout(() => sendMessage(), 100);
                      }}
                    >
                      {suggestion.label}
                    </Button>
                  ))}
                </div>
              </div>
            ) : (
              displayMessages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-3 ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {/* Assistant avatar */}
                  {msg.role !== "user" && (
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  )}

                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-3 ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : msg.role === "system"
                          ? "bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800"
                          : "bg-muted"
                    }`}
                  >
                    {msg.content ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    ) : loading ? (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Thinking...</span>
                      </div>
                    ) : null}

                    {/* Tool calls */}
                    {msg.tool_calls && msg.tool_calls.length > 0 && (
                      <div className="mt-2 space-y-1 border-t border-border/50 pt-2">
                        {msg.tool_calls.map((tc, i) => (
                          <div
                            key={i}
                            className="flex items-center gap-2 text-xs text-muted-foreground"
                          >
                            {tc.status === "success" ? (
                              <CheckCircle2 className="h-3 w-3 text-green-500" />
                            ) : tc.status === "error" ? (
                              <AlertCircle className="h-3 w-3 text-red-500" />
                            ) : (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            )}
                            <code className="text-[10px] bg-background/50 px-1 rounded">
                              {tc.name}
                            </code>
                            {tc.args && (
                              <span className="truncate max-w-[200px]">
                                {JSON.stringify(tc.args).slice(0, 100)}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Timestamp */}
                    <div className="mt-1 flex items-center gap-2">
                      <span className="text-[10px] text-muted-foreground/60">
                        {msg.timestamp.toLocaleTimeString()}
                      </span>
                      {msg.based_on && (
                        <Badge variant="outline" className="text-[9px] h-4 px-1">
                          <Database className="w-2 h-2 mr-0.5" />
                          {msg.based_on.total_nodes || msg.based_on.total || 0}{" "}
                          nodes
                        </Badge>
                      )}
                    </div>
                  </div>

                  {/* User avatar */}
                  {msg.role === "user" && (
                    <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0">
                      <User className="h-4 w-4 text-primary-foreground" />
                    </div>
                  )}
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="border-t border-border bg-card">
            {/* Settings Panel (expandable) */}
            {showSettings && (
              <div className="px-4 pt-3 pb-2 border-b border-border">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {/* Model Selector */}
                  <div>
                    <label className="text-xs text-muted-foreground block mb-1.5">
                      Model
                    </label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full text-xs h-8 rounded-md border border-input bg-background px-2"
                    >
                      {models.length === 0 && (
                        <option value="">No models available</option>
                      )}
                      {models.map((m) => (
                        <option
                          key={formatModelValue(m)}
                          value={formatModelValue(m)}
                        >
                          {formatModelName(m)}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Temperature */}
                  <div>
                    <label className="text-xs text-muted-foreground block mb-1.5">
                      Temperature: {temperature.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={temperature}
                      onChange={(e) =>
                        setTemperature(parseFloat(e.target.value))
                      }
                      className="w-full"
                    />
                  </div>

                  {/* Max Tokens */}
                  <div>
                    <label className="text-xs text-muted-foreground block mb-1.5">
                      Max Tokens: {maxTokens}
                    </label>
                    <input
                      type="range"
                      min={128}
                      max={4096}
                      step={128}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Main Input Row */}
            <div className="p-3">
              <div className="flex gap-2">
                {/* Toggle sidebar (standalone only) */}
                {standalone && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-[44px] w-[44px] shrink-0"
                    onClick={() => setShowSidebar(!showSidebar)}
                    title={showSidebar ? "Hide sidebar" : "Show sidebar"}
                  >
                    {showSidebar ? (
                      <PanelLeftClose className="h-4 w-4" />
                    ) : (
                      <PanelLeft className="h-4 w-4" />
                    )}
                  </Button>
                )}

                {/* Model selector inline (when settings are hidden) */}
                {!showSettings && models.length > 0 && (
                  <div className="hidden sm:flex items-center">
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="text-xs h-[44px] rounded-md border border-input bg-background px-2 max-w-[160px]"
                      title="Select model"
                    >
                      {models.map((m) => (
                        <option
                          key={formatModelValue(m)}
                          value={formatModelValue(m)}
                        >
                          {formatModelName(m)}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <Textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={placeholder}
                  className="min-h-[44px] max-h-[120px] resize-none text-sm"
                  disabled={loading}
                  rows={1}
                />
                <div className="flex flex-col gap-1">
                  <Button
                    onClick={sendMessage}
                    disabled={!input.trim() || loading}
                    size="icon"
                    className="h-[44px] w-[44px]"
                  >
                    {loading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Send className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Footer row */}
              <div className="flex items-center justify-between mt-1">
                <div className="flex items-center gap-2">
                  {currentBank && (
                    <Badge variant="outline" className="text-[10px] h-5">
                      <Database className="w-2.5 h-2.5 mr-1" />
                      {currentBank}
                    </Badge>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-5 text-[10px] px-1 text-muted-foreground hover:text-foreground"
                    onClick={() => setShowSettings(!showSettings)}
                    title={showSettings ? "Hide settings" : "Show settings"}
                  >
                    <SlidersHorizontal className="h-3 w-3 mr-1" />
                    {showSettings ? "Hide params" : "Parameters"}
                  </Button>
                  {standalone && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-5 text-[10px] px-1 text-muted-foreground hover:text-foreground"
                      onClick={clearConversation}
                    >
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Clear
                    </Button>
                  )}
                </div>
                <span className="text-[10px] text-muted-foreground">
                  {loading ? "Streaming..." : `Powered by Hindsight · Shift+Enter for newline · ${temperature.toFixed(2)}t · ${maxTokens}tk`}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
