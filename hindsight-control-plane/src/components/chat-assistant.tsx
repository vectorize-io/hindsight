"use client";

import { useState, useRef, useEffect } from "react";
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
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

export function ChatAssistant({
  standalone = false,
  compact = false,
  placeholder = "Ask anything — recall memories, reflect on patterns, manage banks...",
  initialMessages = [],
}: ChatAssistantProps) {
  const t = useTranslations("operator");
  const { currentBank } = useBank();
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input on mount for standalone mode
  useEffect(() => {
    if (standalone) {
      inputRef.current?.focus();
    }
  }, [standalone]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput("");

    // Add user message
    const userMessage: Message = {
      id: `msg-${Date.now()}-user`,
      role: "user",
      content: userText,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Add placeholder assistant message
    const assistantId = `msg-${Date.now()}-assistant`;
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        tool_calls: [],
      },
    ]);

    setLoading(true);

    try {
      // Build conversation context from recent messages
      const recentMessages = [...messages, userMessage].slice(-10);
      const conversationContext = recentMessages
        .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
        .join("\n\n");

      const bankContext = currentBank
        ? `\n\n[Active bank: ${currentBank}]`
        : "\n\n[No bank selected — operating globally]";

      const fullQuery = `${userText}\n\n[Conversation so far]:\n${conversationContext}\n${bankContext}`;

      // Call reflect API
      const reflectParams: any = {
        query: fullQuery,
        budget: "mid" as any,
        max_tokens: 4096,
        include_facts: true,
        include_tool_calls: true,
      };
      if (currentBank) {
        reflectParams.bank_id = currentBank;
      }
      const data: any = await client.reflect(reflectParams);

      // Update assistant message with response
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content:
                  data.text ||
                  data.answer ||
                  data.response ||
                  "I analyzed your request. (No response generated)",
                tool_calls: data.tool_calls || [],
                based_on: data.based_on,
              }
            : m
        )
      );
    } catch (error: any) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: `**Error**: ${error?.message || "Failed to get response from AI. Is the Hindsight API running?"}`,
                role: "system",
              }
            : m
        )
      );
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
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  };

  const containerClass = standalone
    ? "flex flex-col h-full"
    : compact
      ? "flex flex-col"
      : "flex flex-col";

  const messagesClass = standalone
    ? "flex-1 overflow-y-auto px-4 py-4 space-y-4"
    : compact
      ? "overflow-y-auto px-2 py-2 space-y-2 max-h-[500px]"
      : "flex-1 overflow-y-auto px-4 py-4 space-y-4";

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

      {/* Messages */}
      <div className={messagesClass}>
        {messages.length === 0 ? (
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
                  query: "Based on my memory banks, what patterns and preferences do you see in how I work?",
                },
                {
                  label: "📊 System health check",
                  query: "What is the current health status of all system services?",
                },
                {
                  label: "🏦 Memory bank overview",
                  query: "Summarize my memory banks and what knowledge is stored in each one",
                },
                {
                  label: "⚡ Recent activity",
                  query: "What has been happening recently? Show me recent operations and changes",
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
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
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
                      <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground">
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
                      {msg.based_on.total_nodes || msg.based_on.total || 0} nodes
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
      <div className="border-t border-border p-3 bg-card">
        <div className="flex gap-2">
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
        <div className="flex items-center justify-between mt-1">
          {currentBank && (
            <Badge variant="outline" className="text-[10px] h-5">
              <Database className="w-2.5 h-2.5 mr-1" />
              {currentBank}
            </Badge>
          )}
          <span className="text-[10px] text-muted-foreground">
            Powered by Hindsight · Shift+Enter for newline
          </span>
        </div>
      </div>
    </div>
  );
}
