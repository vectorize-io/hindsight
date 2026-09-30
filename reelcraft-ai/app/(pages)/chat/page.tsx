"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Brain, MessageSquare, Sparkles, Copy, Check } from "lucide-react";
import { MarkdownRenderer } from "@/components/ui/MarkdownRenderer";
import { Badge } from "@/components/ui/Badge";
import { formatRelativeTime } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  memoriesUsed?: string[];
}

const QUICK_PROMPTS = [
  "Give me 5 food Reel ideas",
  "How do I do a speed ramp in CapCut?",
  "Write a hook for a travel Reel",
  "What equipment do I need to start?",
  "How do I sync cuts to music?",
  "Give me a 30-second Reel shot list",
  "How do I color grade food video?",
  "What hashtags should I use for food Reels?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hey! I'm your ReelCraft AI content coach 🎬\n\nI remember your preferences from previous sessions — so I can give you advice tailored to your gear, software, and style.\n\nAsk me anything: Reel ideas, shot lists, editing steps, transitions, color grading, captions, or anything about content creation.",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const history = [...messages, userMsg]
        .filter((m) => m.id !== "welcome")
        .map(({ role, content }) => ({ role, content }));

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-user-id": "reelcraft-user-default" },
        body: JSON.stringify({ messages: history }),
      });

      const data = await res.json();
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.success
          ? data.data.message
          : "Sorry, I ran into an issue. Please try again.",
        timestamp: new Date().toISOString(),
        memoriesUsed: data.data?.hasMemory ? data.data.memoriesUsed?.slice(0, 3) : undefined,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "Something went wrong. Please check your API configuration and try again.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const copy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-64px)]">
      {/* Header */}
      <div className="border-b border-white/[0.06] px-6 py-4 flex items-center justify-between"
        style={{ background: "rgba(10,10,15,0.9)" }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #c026d3, #ea580c)" }}>
            <MessageSquare className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white">ReelCraft AI Coach</h1>
            <p className="text-xs text-gray-500">Powered by Hindsight memory</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-gray-500">Online</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
              {/* Avatar */}
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                msg.role === "assistant"
                  ? "text-white"
                  : "bg-white/10 text-gray-400"
              }`} style={msg.role === "assistant" ? { background: "linear-gradient(135deg, #c026d3, #ea580c)" } : {}}>
                {msg.role === "assistant" ? <Sparkles className="w-4 h-4" /> : <span className="text-xs font-bold">U</span>}
              </div>

              <div className={`flex-1 max-w-2xl ${msg.role === "user" ? "flex flex-col items-end" : ""}`}>
                {/* Memory indicator */}
                {msg.memoriesUsed && msg.memoriesUsed.length > 0 && (
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Brain className="w-3 h-3 text-brand-400" />
                    <span className="text-xs text-brand-400 font-medium">Memory used</span>
                    <Badge variant="brand" className="text-[10px] px-1.5 py-0.5">{msg.memoriesUsed.length}</Badge>
                  </div>
                )}

                {/* Bubble */}
                <div className={`rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "text-white"
                    : "glass text-gray-200"
                } ${msg.role === "user" ? "rounded-tr-sm" : "rounded-tl-sm"}`}
                  style={msg.role === "user" ? { background: "linear-gradient(135deg, #c026d3, #ea580c)" } : {}}>
                  {msg.role === "assistant" ? (
                    <MarkdownRenderer content={msg.content} />
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  )}
                </div>

                {/* Footer */}
                <div className={`flex items-center gap-2 mt-1.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
                  <span className="text-xs text-gray-600">{formatRelativeTime(msg.timestamp)}</span>
                  {msg.role === "assistant" && (
                    <button onClick={() => copy(msg.id, msg.content)} className="text-gray-600 hover:text-gray-400 transition-colors">
                      {copiedId === msg.id ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full flex items-center justify-center text-white flex-shrink-0"
                style={{ background: "linear-gradient(135deg, #c026d3, #ea580c)" }}>
                <Sparkles className="w-4 h-4" />
              </div>
              <div className="glass rounded-2xl rounded-tl-sm px-4 py-3">
                <div className="flex gap-1.5 items-center h-5">
                  {[0, 1, 2].map((i) => (
                    <div key={i} className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }} />
                  ))}
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Quick prompts */}
      {messages.length <= 1 && (
        <div className="px-4 pb-3">
          <div className="max-w-3xl mx-auto">
            <p className="text-xs text-gray-600 mb-2">Quick prompts:</p>
            <div className="flex flex-wrap gap-2">
              {QUICK_PROMPTS.map((p) => (
                <button key={p} onClick={() => sendMessage(p)}
                  className="text-xs px-3 py-1.5 rounded-full glass-hover text-gray-400 hover:text-white transition-all">
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-white/[0.06] px-4 py-4" style={{ background: "rgba(10,10,15,0.95)" }}>
        <div className="max-w-3xl mx-auto">
          <div className="flex gap-2 items-end">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask anything about content creation… (Enter to send, Shift+Enter for new line)"
              className="input flex-1 resize-none min-h-[48px] max-h-36 py-3"
              rows={1}
              disabled={loading}
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="flex-shrink-0 w-11 h-11 rounded-xl flex items-center justify-center transition-all disabled:opacity-40"
              style={{ background: "linear-gradient(135deg, #c026d3, #ea580c)" }}
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>
          <p className="text-xs text-gray-600 mt-1.5 text-center">
            ReelCraft AI remembers your preferences · Powered by Hindsight
          </p>
        </div>
      </div>
    </div>
  );
}
