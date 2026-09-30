"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Star, Brain, Film, Sparkles, ArrowRight, CheckCircle } from "lucide-react";

const DEMO_STEPS = [
  { icon: "🎬", title: "Sample Reel Loaded", desc: "A complete viral food Reel (Pasta Chips) has been pre-analyzed for you — no upload needed." },
  { icon: "🧠", title: "Memory Pre-Seeded", desc: "Creator preferences stored: iPhone, CapCut, beginner level, fast-paced food Reels, 20-30 seconds." },
  { icon: "📋", title: "23-Section Guide", desc: "Browse the complete production blueprint: equipment, shot list, lighting, editing, export settings, and more." },
  { icon: "✨", title: "Personalization Demo", desc: "The second analysis will recall your preferences and automatically adapt the guide to your workflow." },
];

export default function DemoPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [seeded, setSeeded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/demo", { method: "POST" });
      const data = await res.json();
      if (!data.success) throw new Error(data.error);
      setSeeded(true);
      setTimeout(() => router.push(`/projects/${data.data.projectId}`), 1500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Demo setup failed");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-12">
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-yellow-500/30 bg-yellow-500/10 text-yellow-300 text-sm font-semibold mb-5">
          <Star className="w-4 h-4" />
          Hackathon Demo Mode
        </div>

        <h1 className="text-4xl font-display font-bold text-white mb-4">
          Experience ReelCraft AI
        </h1>
        <p className="text-gray-400 max-w-xl mx-auto text-lg">
          No video upload needed. See exactly how ReelCraft AI reverse-engineers a Reel and personalizes guides using Hindsight memory.
        </p>
      </div>

      {/* Demo steps */}
      <div className="glass rounded-2xl p-8 mb-8">
        <h2 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-brand-400" />
          What the demo shows
        </h2>
        <div className="space-y-4">
          {DEMO_STEPS.map((s, i) => (
            <div key={i} className="flex items-start gap-4">
              <span className="text-2xl flex-shrink-0">{s.icon}</span>
              <div>
                <p className="text-white font-semibold text-sm">{s.title}</p>
                <p className="text-gray-400 text-sm mt-0.5">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* The demo narrative */}
      <div className="glass rounded-2xl p-6 mb-8 border border-brand-500/20">
        <h2 className="text-sm font-semibold text-brand-300 mb-4 flex items-center gap-2">
          <Brain className="w-4 h-4" />
          The Hindsight Demo Story
        </h2>
        <div className="space-y-3 text-sm text-gray-300">
          <p className="flex items-start gap-2">
            <span className="font-bold text-white flex-shrink-0">Step 1:</span>
            Open the demo. A pre-analyzed food Reel loads immediately.
          </p>
          <p className="flex items-start gap-2">
            <span className="font-bold text-white flex-shrink-0">Step 2:</span>
            Browse the 23-section guide — this is what the AI generates for any Reel.
          </p>
          <p className="flex items-start gap-2">
            <span className="font-bold text-white flex-shrink-0">Step 3:</span>
            Visit the Memory page — see that the AI has already learned: iPhone, CapCut, beginner, fast-paced, food.
          </p>
          <p className="flex items-start gap-2">
            <span className="font-bold text-white flex-shrink-0">Step 4:</span>
            Go to Chat, type <em className="text-brand-300">"I also like warm color grades and I post mainly on Instagram"</em> — watch Hindsight store it.
          </p>
          <p className="flex items-start gap-2">
            <span className="font-bold text-white flex-shrink-0">Step 5:</span>
            Upload a new Reel — the new guide is automatically adapted to all stored preferences.
          </p>
          <p className="flex items-start gap-2">
            <span className="font-bold text-white flex-shrink-0">Key message:</span>
            <span className="text-white font-medium">ReelCraft AI doesn&apos;t just analyze videos. It learns how YOU create content.</span>
          </p>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm mb-6">
          {error}
        </div>
      )}

      {seeded ? (
        <div className="text-center">
          <div className="inline-flex items-center gap-2 text-green-400 font-semibold mb-2">
            <CheckCircle className="w-5 h-5" />
            Demo ready! Redirecting…
          </div>
        </div>
      ) : (
        <button onClick={startDemo} disabled={loading}
          className={`w-full flex items-center justify-center gap-2 py-4 rounded-xl font-semibold text-white text-lg transition-all ${loading ? "opacity-60 cursor-not-allowed bg-white/5" : "btn-primary"}`}>
          {loading ? (
            <><div className="w-5 h-5 rounded-full border-2 border-white border-t-transparent animate-spin" />Setting up demo…</>
          ) : (
            <><Film className="w-5 h-5" />Start the Demo<ArrowRight className="w-5 h-5" /></>
          )}
        </button>
      )}

      <p className="text-center text-xs text-gray-600 mt-4">
        Demo uses a pre-built guide — no OpenAI credits needed to view it.
        Real analysis requires an OpenAI API key.
      </p>
    </div>
  );
}
