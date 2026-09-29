"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import type { Project } from "@/types";
import { PageLoader } from "@/components/ui/LoadingSpinner";
import { MemoryBanner } from "@/components/ui/MemoryBanner";
import { SectionCard } from "@/components/ui/SectionCard";
import { Badge } from "@/components/ui/Badge";
import { SceneTimeline } from "@/components/analysis/SceneTimeline";
import { ShotListTable } from "@/components/analysis/ShotListTable";
import { TechniqueCards } from "@/components/analysis/TechniqueCards";
import { EquipmentGrid } from "@/components/analysis/EquipmentGrid";
import { FinalChecklist } from "@/components/analysis/FinalChecklist";
import { MarkdownRenderer } from "@/components/ui/MarkdownRenderer";
import { skillLevelColor, formatDuration } from "@/lib/utils";
import {
  Film, Brain, ArrowRight, Clock, Maximize2, Gauge, Music,
  Camera, Layers, CheckSquare, Sparkles, BookOpen, Star,
  Palette, Type, ArrowLeftRight, Share2, Lightbulb, Trophy,
  DollarSign, List, Wand2, ChevronLeft
} from "lucide-react";

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [memories, setMemories] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<"guide" | "scenes" | "learn">("guide");

  useEffect(() => {
    fetch(`/api/projects/${id}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.success) setProject(d.data.project);
        else setError(d.error ?? "Project not found");
      })
      .catch(() => setError("Failed to load project"))
      .finally(() => setLoading(false));

    // Load recalled memories for this session
    const stored = sessionStorage.getItem(`memories-${id}`);
    if (stored) setMemories(JSON.parse(stored));
  }, [id]);

  if (loading) return <PageLoader message="Loading your recreation guide…" />;
  if (error || !project) return (
    <div className="max-w-2xl mx-auto px-4 py-20 text-center">
      <p className="text-red-400 mb-4">{error ?? "Project not found"}</p>
      <Link href="/analyze" className="btn-primary">Analyze a new Reel</Link>
    </div>
  );

  const g = project.guide;
  if (!g) return (
    <div className="max-w-2xl mx-auto px-4 py-20 text-center">
      <p className="text-gray-400 mb-4">This project is still being analyzed.</p>
      <button onClick={() => window.location.reload()} className="btn-secondary">Refresh</button>
    </div>
  );

  const tabs = [
    { id: "guide", label: "Recreation Guide", icon: BookOpen },
    { id: "scenes", label: "Scene Timeline", icon: Layers },
    { id: "learn", label: "Learn Mode", icon: Star },
  ] as const;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Back */}
      <button onClick={() => router.back()} className="btn-ghost mb-6 -ml-2">
        <ChevronLeft className="w-4 h-4" />
        Back
      </button>

      {/* Memory banner */}
      {memories.length > 0 && <MemoryBanner memories={memories} />}

      {/* Header */}
      <div className="glass rounded-2xl p-6 mb-6">
        <div className="flex flex-wrap items-start gap-4">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0"
            style={{ background: "linear-gradient(135deg, #c026d3, #ea580c)" }}>
            <Film className="w-7 h-7 text-white" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              {project.isDemo && <Badge variant="yellow">Demo</Badge>}
              <Badge variant="brand">{g.section1_overview.contentCategory}</Badge>
              <Badge variant={
                g.section2_skillLevel.level === "Beginner" ? "green" :
                g.section2_skillLevel.level === "Intermediate" ? "yellow" : "red"
              }>
                {g.section2_skillLevel.level}
              </Badge>
            </div>
            <h1 className="text-2xl font-display font-bold text-white mb-1">{project.title}</h1>
            <p className="text-gray-400 text-sm line-clamp-2">{g.section1_overview.about}</p>
          </div>

          {project.metadata && (
            <div className="flex flex-wrap gap-3 text-xs text-gray-400">
              <span className="flex items-center gap-1"><Clock className="w-3.5 h-3.5" />{formatDuration(project.metadata.duration)}</span>
              <span className="flex items-center gap-1"><Maximize2 className="w-3.5 h-3.5" />{project.metadata.aspectRatio}</span>
              <span className="flex items-center gap-1"><Gauge className="w-3.5 h-3.5" />{project.metadata.frameRate}fps</span>
            </div>
          )}
        </div>
      </div>

      {/* Create My Version CTA */}
      <div className="gradient-border rounded-2xl p-px mb-6">
        <div className="rounded-2xl p-5 flex flex-col sm:flex-row items-center gap-4 justify-between"
          style={{ background: "linear-gradient(135deg, rgba(192,38,211,0.08), rgba(234,88,12,0.08))" }}>
          <div>
            <p className="text-white font-semibold mb-1">Ready to make your own version?</p>
            <p className="text-sm text-gray-400">Take this production structure and create an original Reel for your own topic.</p>
          </div>
          <Link href={`/create?ref=${project.id}`} className="btn-primary flex-shrink-0">
            <Wand2 className="w-4 h-4" />
            Create My Version
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/[0.08] mb-8">
        {tabs.map(({ id: tabId, label, icon: Icon }) => (
          <button
            key={tabId}
            onClick={() => setActiveTab(tabId)}
            className={`flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all border-b-2 -mb-px ${
              activeTab === tabId
                ? "text-white border-brand-500"
                : "text-gray-500 border-transparent hover:text-gray-300"
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* ── GUIDE TAB ── */}
      {activeTab === "guide" && (
        <div className="space-y-4">
          {/* Section 1 — Overview */}
          <SectionCard number={1} title="Reel Overview" subtitle={g.section1_overview.overallStyle} defaultOpen icon={<Film className="w-4 h-4" />}>
            <div className="grid sm:grid-cols-2 gap-4 text-sm">
              <Field label="What it's about" value={g.section1_overview.about} />
              <Field label="What makes it effective" value={g.section1_overview.whatMakesItEffective} />
              <Field label="Target audience" value={g.section1_overview.targetAudience} />
              <Field label="Main creative idea" value={g.section1_overview.mainCreativeIdea} />
            </div>
            {project.creativeStrategy && (
              <div className="mt-4 pt-4 border-t border-white/[0.06]">
                <p className="text-xs font-semibold text-gray-500 mb-3 uppercase tracking-wide">Creative Strategy</p>
                <div className="grid sm:grid-cols-2 gap-3 text-sm">
                  <Field label="Hook" value={project.creativeStrategy.hook} />
                  <Field label="Structure" value={project.creativeStrategy.storytellingStructure} />
                  <Field label="Emotional tone" value={project.creativeStrategy.emotionalTone} />
                  <Field label="Pacing" value={project.creativeStrategy.pacing} />
                </div>
              </div>
            )}
          </SectionCard>

          {/* Section 2 — Skill Level */}
          <SectionCard number={2} title="Skill Level Required"
            badge={g.section2_skillLevel.level}
            badgeColor={skillLevelColor(g.section2_skillLevel.level)}>
            <p className="text-sm text-gray-300 mb-4">{g.section2_skillLevel.explanation}</p>
            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <p className="text-xs font-semibold text-red-400 mb-2 uppercase">Must Know</p>
                <ul className="space-y-1">{g.section2_skillLevel.mustKnow.map((s, i) => (
                  <li key={i} className="text-sm text-gray-300 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />{s}
                  </li>
                ))}</ul>
              </div>
              <div>
                <p className="text-xs font-semibold text-green-400 mb-2 uppercase">Optional</p>
                <ul className="space-y-1">{g.section2_skillLevel.optional.map((s, i) => (
                  <li key={i} className="text-sm text-gray-300 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-400 flex-shrink-0" />{s}
                  </li>
                ))}</ul>
              </div>
            </div>
          </SectionCard>

          {/* Section 3 — Skills to Learn */}
          <SectionCard number={3} title="Skills to Learn Before Starting" icon={<BookOpen className="w-4 h-4" />}>
            <div className="space-y-4">
              {g.section3_skillsToLearn.map((skill, i) => (
                <div key={i} className="glass rounded-xl p-4">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <h4 className="text-sm font-semibold text-white">{skill.name}</h4>
                    <div className="flex gap-2 flex-shrink-0">
                      <Badge variant={skill.difficulty === "Easy" ? "green" : skill.difficulty === "Medium" ? "yellow" : "red"}>
                        {skill.difficulty}
                      </Badge>
                      <Badge variant="default">{skill.timeToLearn}</Badge>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mb-2">{skill.whatItMeans}</p>
                  <p className="text-xs text-gray-500"><span className="text-brand-400">Why needed:</span> {skill.whyNeeded}</p>
                  {skill.practice && (
                    <div className="mt-2 px-3 py-2 rounded-lg bg-green-500/5 border border-green-500/15">
                      <p className="text-xs text-green-400"><span className="font-semibold">Practice: </span>{skill.practice}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Section 4 — Equipment */}
          <SectionCard number={4} title="Equipment Required" icon={<Camera className="w-4 h-4" />}>
            <EquipmentGrid equipment={g.section4_equipment} />
          </SectionCard>

          {/* Section 5 — Apps */}
          <SectionCard number={5} title="Apps & Software" icon={<Layers className="w-4 h-4" />}>
            <div className="grid sm:grid-cols-2 gap-3">
              {g.section5_apps.map((app, i) => (
                <div key={i} className="glass rounded-xl p-4">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <h4 className="text-sm font-semibold text-white">{app.name}</h4>
                    <Badge variant="default" className="text-xs">{app.purpose}</Badge>
                  </div>
                  <p className="text-xs text-gray-400 mb-2">{app.usedFor}</p>
                  <div className="flex gap-2 flex-wrap">
                    <Badge variant={app.difficulty === "Easy" ? "green" : app.difficulty === "Medium" ? "yellow" : "red"}>
                      {app.difficulty}
                    </Badge>
                    <Badge variant="default">{app.cost}</Badge>
                  </div>
                  {app.beginnerAlternative && (
                    <p className="text-xs text-brand-400 mt-2">Alt: {app.beginnerAlternative}</p>
                  )}
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Section 6 — Techniques */}
          <SectionCard number={6} title="Techniques Used" icon={<Sparkles className="w-4 h-4" />}>
            <TechniqueCards techniques={g.section6_techniques} />
          </SectionCard>

          {/* Section 7 — Pre-Production */}
          <SectionCard number={7} title="Pre-Production Plan" icon={<List className="w-4 h-4" />}>
            <div className="grid sm:grid-cols-2 gap-6">
              <div>
                <p className="text-xs font-semibold text-gray-500 mb-3 uppercase">Steps</p>
                <ol className="space-y-2">
                  {g.section7_preProduction.steps.map((s, i) => (
                    <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                      <span className="text-brand-400 font-semibold flex-shrink-0">{i + 1}.</span>
                      {s.replace(/^\d+\.\s*/, "")}
                    </li>
                  ))}
                </ol>
              </div>
              <div>
                <p className="text-xs font-semibold text-gray-500 mb-3 uppercase">Prep Checklist</p>
                <ul className="space-y-1.5">
                  {g.section7_preProduction.checklist.map((c, i) => (
                    <li key={i} className="text-sm text-gray-300 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-brand-400 flex-shrink-0" />
                      {c.replace(/^[☐□]\s*/, "")}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </SectionCard>

          {/* Section 8 — Script */}
          <SectionCard number={8} title="Script / Voiceover Structure" icon={<Type className="w-4 h-4" />}>
            <div className="space-y-3">
              {g.section8_script.map((s, i) => (
                <div key={i} className="glass rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-2.5 py-1 rounded-lg text-xs font-bold text-white"
                      style={{ background: "linear-gradient(135deg, #c026d3, #ea580c)" }}>
                      {s.label}
                    </span>
                    <span className="text-xs text-gray-500">{s.timeRange}</span>
                    {s.tone && <Badge variant="default">{s.tone}</Badge>}
                  </div>
                  <p className="text-sm text-white mb-2">{s.description}</p>
                  {s.voiceoverInstructions && (
                    <p className="text-xs text-brand-400">🎙 {s.voiceoverInstructions}</p>
                  )}
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Section 9 — Shot List */}
          <SectionCard number={9} title="Complete Shot List" icon={<Camera className="w-4 h-4" />} badge="Most Important">
            <ShotListTable shots={g.section9_shotList} />
          </SectionCard>

          {/* Section 10 — Filming */}
          <SectionCard number={10} title="Filming Instructions" icon={<Film className="w-4 h-4" />}>
            <MarkdownRenderer content={g.section10_filmingInstructions} />
          </SectionCard>

          {/* Section 11 — Lighting */}
          <SectionCard number={11} title="Lighting Setup" icon={<Lightbulb className="w-4 h-4" />}>
            <MarkdownRenderer content={g.section11_lightingSetup} />
          </SectionCard>

          {/* Section 12 — Audio */}
          <SectionCard number={12} title="Audio & Music" icon={<Music className="w-4 h-4" />}>
            <MarkdownRenderer content={g.section12_audioMusic} />
          </SectionCard>

          {/* Section 13 — Editing Timeline */}
          <SectionCard number={13} title="Editing Timeline" icon={<Layers className="w-4 h-4" />}>
            <MarkdownRenderer content={g.section13_editingTimeline} />
          </SectionCard>

          {/* Section 14 — App-Specific Guide */}
          <SectionCard number={14} title="App-Specific Editing Guide" subtitle="CapCut · Premiere Pro" icon={<Star className="w-4 h-4" />}>
            <MarkdownRenderer content={g.section14_appSpecificGuide} />
          </SectionCard>

          {/* Section 15 — Color Grade */}
          <SectionCard number={15} title="Color Grading" icon={<Palette className="w-4 h-4" />}
            badge={g.section15_colorGrading.style}>
            <div className="grid sm:grid-cols-2 gap-3 text-sm mb-4">
              <Field label="Exposure" value={g.section15_colorGrading.exposure} />
              <Field label="Contrast" value={g.section15_colorGrading.contrast} />
              <Field label="Highlights" value={g.section15_colorGrading.highlights} />
              <Field label="Shadows" value={g.section15_colorGrading.shadows} />
              <Field label="Saturation" value={g.section15_colorGrading.saturation} />
              <Field label="Temperature" value={g.section15_colorGrading.temperature} />
            </div>
            {g.section15_colorGrading.beginnerSettings && (
              <div className="p-4 rounded-xl border border-brand-500/20 bg-brand-500/5">
                <p className="text-xs font-semibold text-brand-300 mb-1">Beginner Settings</p>
                <p className="text-sm text-gray-300">{g.section15_colorGrading.beginnerSettings}</p>
              </div>
            )}
          </SectionCard>

          {/* Section 16 — Text & Graphics */}
          <SectionCard number={16} title="Text & Graphics" icon={<Type className="w-4 h-4" />}>
            <MarkdownRenderer content={g.section16_textGraphics} />
          </SectionCard>

          {/* Section 17 — Transitions */}
          <SectionCard number={17} title="Transitions" icon={<ArrowLeftRight className="w-4 h-4" />}>
            <div className="space-y-3">
              {g.section17_transitions.map((t, i) => (
                <div key={i} className="glass rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Badge variant="brand">{t.type}</Badge>
                    <span className="text-xs text-gray-500">{t.timestamp}</span>
                  </div>
                  <p className="text-sm text-gray-300 mb-2">{t.purpose}</p>
                  <div className="space-y-1.5 text-xs">
                    <p><span className="text-brand-400 font-semibold">How it works:</span> <span className="text-gray-400">{t.howItWorks}</span></p>
                    <p><span className="text-accent-400 font-semibold">How to recreate:</span> <span className="text-gray-400">{t.howToRecreate}</span></p>
                    {t.beginnerAlternative && (
                      <p><span className="text-green-400 font-semibold">Simpler option:</span> <span className="text-gray-400">{t.beginnerAlternative}</span></p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Section 18 — Export */}
          <SectionCard number={18} title="Export Settings" icon={<Share2 className="w-4 h-4" />}>
            <div className="grid sm:grid-cols-3 gap-3">
              {g.section18_exportSettings.map((s, i) => (
                <div key={i} className="glass rounded-xl p-4">
                  <p className="text-sm font-semibold text-white mb-3">{s.platform}</p>
                  <div className="space-y-1 text-xs">
                    <Row2 label="Resolution" value={s.resolution} />
                    <Row2 label="Aspect Ratio" value={s.aspectRatio} />
                    <Row2 label="Frame Rate" value={s.frameRate} />
                    <Row2 label="Codec" value={s.codec} />
                    <Row2 label="Audio" value={s.audioFormat} />
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          {/* Section 19 — Publishing */}
          <SectionCard number={19} title="Publishing Guide" icon={<Share2 className="w-4 h-4" />}>
            <div className="space-y-4">
              {g.section19_publishing.captionStructure && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 mb-2 uppercase">Caption Structure</p>
                  <p className="text-sm text-gray-300">{g.section19_publishing.captionStructure}</p>
                </div>
              )}
              {g.section19_publishing.cta && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 mb-2 uppercase">Call to Action</p>
                  <div className="px-4 py-3 rounded-xl bg-brand-500/5 border border-brand-500/20">
                    <p className="text-sm text-brand-300">{g.section19_publishing.cta}</p>
                  </div>
                </div>
              )}
              {g.section19_publishing.hashtags.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 mb-2 uppercase">Hashtags</p>
                  <div className="flex flex-wrap gap-2">
                    {g.section19_publishing.hashtags.map((h, i) => (
                      <span key={i} className="text-xs px-2 py-1 rounded-md bg-white/5 text-blue-400">{h}</span>
                    ))}
                  </div>
                </div>
              )}
              {g.section19_publishing.postingChecklist.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 mb-2 uppercase">Posting Checklist</p>
                  <ul className="space-y-1">
                    {g.section19_publishing.postingChecklist.map((c, i) => (
                      <li key={i} className="text-sm text-gray-300 flex items-center gap-2">
                        <span className="text-green-400">✓</span>
                        {c.replace(/^[✓☐]\s*/, "")}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </SectionCard>

          {/* Section 20 — Beginner Version */}
          <SectionCard number={20} title="Beginner Version" subtitle="Phone + free apps only" icon={<Sparkles className="w-4 h-4" />} badge="Simplified">
            <MarkdownRenderer content={g.section20_beginnerVersion} />
          </SectionCard>

          {/* Section 21 — Professional Version */}
          <SectionCard number={21} title="Professional Version" icon={<Trophy className="w-4 h-4" />} badge="Advanced">
            <MarkdownRenderer content={g.section21_professionalVersion} />
          </SectionCard>

          {/* Section 22 — Time & Budget */}
          <SectionCard number={22} title="Time & Budget Estimate" icon={<DollarSign className="w-4 h-4" />}>
            <MarkdownRenderer content={g.section22_timeBudget} />
          </SectionCard>

          {/* Section 23 — Final Checklist */}
          <SectionCard number={23} title="Final Checklist" icon={<CheckSquare className="w-4 h-4" />} defaultOpen badge="Interactive">
            <FinalChecklist items={g.section23_finalChecklist} />
          </SectionCard>
        </div>
      )}

      {/* ── SCENES TAB ── */}
      {activeTab === "scenes" && (
        <div className="glass rounded-2xl p-6">
          <h2 className="text-lg font-bold text-white mb-2">Scene-by-Scene Analysis</h2>
          <p className="text-sm text-gray-400 mb-6">
            Click any scene to see the detailed analysis. Click a bar in the timeline to jump to that scene.
          </p>
          {project.scenes?.length && project.metadata ? (
            <SceneTimeline scenes={project.scenes} metadata={project.metadata} />
          ) : (
            <p className="text-gray-500 text-sm">No scene data available for this project.</p>
          )}
        </div>
      )}

      {/* ── LEARN TAB ── */}
      {activeTab === "learn" && (
        <div className="space-y-4">
          <div className="glass rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-yellow-500/20 flex items-center justify-center">
                <Star className="w-5 h-5 text-yellow-400" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">Learn This Reel</h2>
                <p className="text-sm text-gray-400">Deep-dive into every technique — what it is, why it works, and how to master it.</p>
              </div>
            </div>

            <div className="space-y-4">
              {g.section6_techniques.map((tech, i) => (
                <div key={i} className="glass rounded-xl p-5">
                  <h3 className="text-base font-bold text-white mb-3">{tech.name}</h3>
                  <div className="grid sm:grid-cols-2 gap-4 text-sm">
                    <div className="space-y-3">
                      <Block label="WHAT IT IS" color="text-brand-400" text={tech.what} />
                      <Block label="WHY IT WORKS" color="text-accent-400" text={tech.why} />
                    </div>
                    <div className="space-y-3">
                      <Block label="HOW TO DO IT" color="text-green-400" text={tech.how} />
                      <Block label="SIMPLER ALTERNATIVE" color="text-blue-400" text={tech.alternative} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Bottom CTA */}
      <div className="mt-8 glass rounded-2xl p-6 text-center">
        <p className="text-white font-semibold mb-1">Ready to make YOUR version?</p>
        <p className="text-sm text-gray-400 mb-4">Use this structure to create a completely original Reel for your own topic.</p>
        <Link href={`/create?ref=${project.id}`} className="btn-primary">
          <Wand2 className="w-4 h-4" />
          Create My Version
        </Link>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  if (!value) return null;
  return (
    <div>
      <p className="text-xs font-semibold text-gray-500 mb-1 uppercase tracking-wide">{label}</p>
      <p className="text-sm text-gray-300">{value}</p>
    </div>
  );
}

function Row2({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-300 font-medium">{value}</span>
    </div>
  );
}

function Block({ label, color, text }: { label: string; color: string; text: string }) {
  if (!text) return null;
  return (
    <div>
      <p className={`text-xs font-bold mb-1 uppercase tracking-wide ${color}`}>{label}</p>
      <p className="text-gray-300 text-sm leading-relaxed">{text}</p>
    </div>
  );
}
