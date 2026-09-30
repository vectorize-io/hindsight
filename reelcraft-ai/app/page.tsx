"use client";

import Link from "next/link";
import { Film, Brain, MessageSquare, Sparkles, ArrowRight, Zap, Target, BookOpen, Star } from "lucide-react";
import { useEffect, useState } from "react";
import type { Project } from "@/types";
import { formatRelativeTime } from "@/lib/utils";

const FEATURES = [
  {
    icon: Film,
    title: "Deep Video Analysis",
    desc: "Upload any Reel and get a scene-by-scene breakdown: camera angles, lighting, editing techniques, and pacing — all reverse-engineered.",
    color: "from-brand-600 to-brand-500",
  },
  {
    icon: Brain,
    title: "Hindsight Memory",
    desc: "ReelCraft AI remembers your skill level, equipment, software, and creative style. Every guide is personalized to YOUR workflow.",
    color: "from-purple-600 to-purple-500",
  },
  {
    icon: Target,
    title: "23-Section Blueprint",
    desc: "Not a summary — a full production manual. Equipment list, shot list, lighting diagram, editing workflow, color grade, and export settings.",
    color: "from-accent-600 to-accent-500",
  },
  {
    icon: Zap,
    title: "Create My Version",
    desc: "Take any Reel's production structure and create a 100% original guide for your own content, topic, and brand.",
    color: "from-emerald-600 to-emerald-500",
  },
  {
    icon: BookOpen,
    title: "Learning Mode",
    desc: "Every technique explained: what it is, why it works, how to do it, common mistakes, and a beginner exercise.",
    color: "from-blue-600 to-blue-500",
  },
  {
    icon: MessageSquare,
    title: "AI Content Coach",
    desc: "Chat with an AI that knows your preferences. Ask for Reel ideas, shot lists, scripts, transition help, or editing guidance.",
    color: "from-pink-600 to-pink-500",
  },
];

const DEMO_STEPS = [
  { step: "1", text: "Upload a Reel or paste a URL" },
  { step: "2", text: "AI analyzes every scene, technique, and creative decision" },
  { step: "3", text: "Get a complete production blueprint with 23 sections" },
  { step: "4", text: "AI remembers your gear, software, and style for next time" },
  { step: "5", text: "Generate your own original Reel inspired by the structure" },
];

export default function HomePage() {
  const [recentProjects, setRecentProjects] = useState<Project[]>([]);

  useEffect(() => {
    fetch("/api/projects", { headers: { "x-user-id": "reelcraft-user-default" } })
      .then((r) => r.json())
      .then((d) => { if (d.success) setRecentProjects(d.data.projects.slice(0, 3)); })
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen">
      {/* ── Hero ── */}
      <section className="relative overflow-hidden px-4 pt-20 pb-24 text-center">
        {/* Background glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] rounded-full opacity-20"
            style={{ background: "radial-gradient(ellipse, #c026d3, transparent 70%)", filter: "blur(60px)" }} />
        </div>

        <div className="relative max-w-4xl mx-auto">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-brand-500/30 bg-brand-500/10 text-brand-300 text-sm font-medium mb-6">
            <Sparkles className="w-3.5 h-3.5" />
            AI-Powered · Hindsight Memory · 23-Section Guide
          </div>

          <h1 className="text-5xl md:text-7xl font-display font-bold tracking-tight mb-6 leading-tight">
            <span className="text-white">How did they</span>
            <br />
            <span className="gradient-text">make that Reel?</span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            ReelCraft AI reverse-engineers any Reel into a complete, step-by-step production
            blueprint — personalized to your skill level, equipment, and editing software.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link href="/analyze" className="btn-primary text-base px-7 py-3.5">
              <Film className="w-5 h-5" />
              Analyze a Reel
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link href="/demo" className="btn-secondary text-base px-7 py-3.5">
              <Star className="w-4 h-4 text-yellow-400" />
              Try the Demo
            </Link>
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="max-w-4xl mx-auto px-4 pb-20">
        <div className="glass rounded-2xl p-8">
          <h2 className="text-xl font-bold text-white mb-6 text-center">How it works</h2>
          <div className="flex flex-col md:flex-row gap-0">
            {DEMO_STEPS.map((s, i) => (
              <div key={i} className="flex-1 flex flex-col md:flex-row items-center gap-2">
                <div className="flex md:flex-col items-center gap-3 flex-1 md:text-center">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white flex-shrink-0"
                    style={{ background: "linear-gradient(135deg, #c026d3, #ea580c)" }}>
                    {s.step}
                  </div>
                  <p className="text-sm text-gray-400 md:text-center">{s.text}</p>
                </div>
                {i < DEMO_STEPS.length - 1 && (
                  <ArrowRight className="w-4 h-4 text-gray-600 flex-shrink-0 hidden md:block" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="max-w-7xl mx-auto px-4 pb-24">
        <h2 className="text-3xl font-display font-bold text-center text-white mb-3">
          Everything a creator needs
        </h2>
        <p className="text-center text-gray-500 mb-12">From raw video to publishable Reel — one tool, end to end.</p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map(({ icon: Icon, title, desc, color }) => (
            <div key={title} className="glass-hover rounded-2xl p-6 group">
              <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center mb-4 group-hover:scale-105 transition-transform duration-200`}>
                <Icon className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-white font-semibold mb-2">{title}</h3>
              <p className="text-sm text-gray-400 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Memory highlight ── */}
      <section className="max-w-5xl mx-auto px-4 pb-24">
        <div className="gradient-border rounded-2xl p-px">
          <div className="rounded-2xl p-8 md:p-10" style={{ background: "linear-gradient(135deg, #0f0818, #180a28)" }}>
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/15 text-brand-300 text-xs font-semibold border border-brand-500/30 mb-4">
                  <Brain className="w-3 h-3" />
                  Powered by Hindsight
                </div>
                <h2 className="text-2xl md:text-3xl font-display font-bold text-white mb-4">
                  The AI that learns <span className="gradient-text">how YOU create</span>
                </h2>
                <p className="text-gray-400 leading-relaxed mb-6">
                  Tell ReelCraft AI once that you shoot on an iPhone and edit in Premiere Pro.
                  Every future guide automatically adapts — no need to repeat yourself.
                </p>
                <Link href="/memory" className="btn-secondary">
                  <Brain className="w-4 h-4" />
                  View Creative Memory
                </Link>
              </div>

              <div className="space-y-3">
                {[
                  { icon: "👤", label: "Remembered", text: "You edit using Premiere Pro" },
                  { icon: "🎨", label: "Remembered", text: "You prefer fast-paced food Reels" },
                  { icon: "📱", label: "Remembered", text: "You film with your iPhone" },
                  { icon: "⏱", label: "Remembered", text: "Target duration: 20-30 seconds" },
                ].map((m, i) => (
                  <div key={i} className="flex items-center gap-3 px-4 py-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <span className="text-xl">{m.icon}</span>
                    <div>
                      <span className="text-xs font-semibold text-brand-400 block">{m.label}</span>
                      <span className="text-sm text-white">{m.text}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Recent Projects ── */}
      {recentProjects.length > 0 && (
        <section className="max-w-7xl mx-auto px-4 pb-24">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white">Continue Creating</h2>
            <Link href="/projects" className="text-sm text-brand-400 hover:text-brand-300 transition-colors flex items-center gap-1">
              All projects <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {recentProjects.map((p) => (
              <Link key={p.id} href={`/projects/${p.id}`}
                className="glass-hover rounded-2xl p-5 block group">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-8 h-8 rounded-lg bg-brand-500/20 flex items-center justify-center">
                    <Film className="w-4 h-4 text-brand-400" />
                  </div>
                  {p.isDemo && (
                    <span className="badge bg-yellow-500/15 text-yellow-400 border-yellow-500/30 text-xs">Demo</span>
                  )}
                </div>
                <h3 className="text-sm font-semibold text-white mb-1 line-clamp-2 group-hover:text-brand-300 transition-colors">
                  {p.title}
                </h3>
                <p className="text-xs text-gray-500">{formatRelativeTime(p.createdAt)}</p>
              </Link>
            ))}
          </div>
        </section>
      )}

      {/* ── CTA ── */}
      <section className="max-w-3xl mx-auto px-4 pb-32 text-center">
        <h2 className="text-3xl font-display font-bold text-white mb-4">
          Ready to recreate any Reel?
        </h2>
        <p className="text-gray-400 mb-8">Upload your first video and get a complete production blueprint in minutes.</p>
        <Link href="/analyze" className="btn-primary text-base px-8 py-4 inline-flex">
          <Film className="w-5 h-5" />
          Analyze Your First Reel
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  );
}
