"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Film, Link2, ArrowRight, Sparkles, AlertCircle, Brain } from "lucide-react";
import { VideoUploader } from "@/components/ui/VideoUploader";
import { cn } from "@/lib/utils";

type Tab = "upload" | "url";

export default function AnalyzePage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("upload");
  const [videoUrl, setVideoUrl] = useState("");
  const [additionalContext, setAdditionalContext] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>("");

  const PROGRESS_STEPS = [
    "Extracting video metadata…",
    "Analyzing visual frames…",
    "Recalling your creative memory…",
    "Reverse-engineering production techniques…",
    "Building your personalized guide…",
  ];

  const animateProgress = () => {
    let i = 0;
    const interval = setInterval(() => {
      setProgress(PROGRESS_STEPS[i % PROGRESS_STEPS.length]);
      i++;
      if (i >= PROGRESS_STEPS.length) clearInterval(interval);
    }, 2800);
    return interval;
  };

  const handleAnalyze = async () => {
    setError(null);

    if (tab === "upload" && !selectedFile) {
      setError("Please select a video file to upload.");
      return;
    }
    if (tab === "url" && !videoUrl.trim()) {
      setError("Please enter a video URL.");
      return;
    }

    setLoading(true);
    setProgress(PROGRESS_STEPS[0]);
    const progressInterval = animateProgress();

    try {
      let response: Response;

      if (tab === "upload" && selectedFile) {
        const formData = new FormData();
        formData.append("video", selectedFile);
        response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "x-user-id": "reelcraft-user-default" },
          body: formData,
        });
      } else {
        response = await fetch("/api/analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-user-id": "reelcraft-user-default",
          },
          body: JSON.stringify({ videoUrl: videoUrl.trim(), additionalContext }),
        });
      }

      clearInterval(progressInterval);
      const data = await response.json();

      if (!data.success) {
        setError(data.error ?? "Analysis failed. Please try again.");
        return;
      }

      router.push(`/projects/${data.data.project.id}`);
    } catch (err) {
      clearInterval(progressInterval);
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
      setProgress("");
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-12">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-brand-500/30 bg-brand-500/10 text-brand-300 text-xs font-semibold mb-4">
          <Sparkles className="w-3 h-3" />
          AI Video Analysis
        </div>
        <h1 className="text-4xl font-display font-bold text-white mb-3">
          Analyze a Reel
        </h1>
        <p className="text-gray-400 max-w-md mx-auto">
          Upload a video or paste a URL. ReelCraft AI will reverse-engineer it into a complete production blueprint.
        </p>
      </div>

      {/* Memory reminder */}
      <div className="glass rounded-xl p-4 mb-8 flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-brand-500/20 flex items-center justify-center flex-shrink-0">
          <Brain className="w-4 h-4 text-brand-400" />
        </div>
        <p className="text-sm text-gray-400">
          <span className="text-brand-300 font-medium">Hindsight is active.</span>{" "}
          If you&apos;ve set your preferences before, this guide will be automatically personalized to your skill level, equipment, and software.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/[0.08] mb-8">
        {(["upload", "url"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "flex items-center gap-2 px-5 py-3 text-sm font-medium capitalize transition-all border-b-2 -mb-px",
              tab === t
                ? "text-white border-brand-500"
                : "text-gray-500 border-transparent hover:text-gray-300"
            )}
          >
            {t === "upload" ? <Film className="w-4 h-4" /> : <Link2 className="w-4 h-4" />}
            {t === "upload" ? "Upload Video" : "Paste URL"}
          </button>
        ))}
      </div>

      {/* Upload tab */}
      {tab === "upload" && (
        <div className="space-y-6">
          <VideoUploader
            onFileSelected={setSelectedFile}
            isLoading={loading}
            disabled={loading}
          />
        </div>
      )}

      {/* URL tab */}
      {tab === "url" && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Video URL</label>
            <input
              type="url"
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              placeholder="https://www.instagram.com/reel/..."
              className="input"
              disabled={loading}
            />
            <p className="text-xs text-gray-500 mt-2">
              Supported: Instagram, YouTube, TikTok, Twitter/X
            </p>
            <p className="text-xs text-yellow-500 mt-1">
              ⚠️ For best results, download the video and use the upload option. URLs are analyzed without direct video access.
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Additional context <span className="text-gray-600">(optional)</span>
            </label>
            <textarea
              value={additionalContext}
              onChange={(e) => setAdditionalContext(e.target.value)}
              placeholder="e.g. This is a food Reel about pasta, about 25 seconds long, uses fast cuts and ASMR sounds..."
              className="input resize-none h-24"
              disabled={loading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Describe what you remember about the video to improve the analysis.
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Progress */}
      {loading && progress && (
        <div className="mt-6 glass rounded-xl p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-5 h-5 rounded-full border-2 border-brand-500 border-t-transparent animate-spin flex-shrink-0" />
            <span className="text-sm font-medium text-white">{progress}</span>
          </div>
          <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div className="h-full rounded-full animate-pulse"
              style={{ width: "60%", background: "linear-gradient(90deg, #c026d3, #ea580c)" }} />
          </div>
          <p className="text-xs text-gray-500 mt-2">This can take 30–60 seconds for a real video.</p>
        </div>
      )}

      {/* Submit */}
      <button
        onClick={handleAnalyze}
        disabled={loading || (tab === "upload" && !selectedFile) || (tab === "url" && !videoUrl.trim())}
        className={cn(
          "w-full mt-8 flex items-center justify-center gap-2 py-4 rounded-xl font-semibold text-white transition-all duration-200",
          loading || (tab === "upload" && !selectedFile) || (tab === "url" && !videoUrl.trim())
            ? "opacity-40 cursor-not-allowed bg-white/5"
            : "btn-primary"
        )}
      >
        {loading ? (
          <>
            <div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
            Analyzing Reel…
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            Analyze & Generate Guide
            <ArrowRight className="w-4 h-4" />
          </>
        )}
      </button>
    </div>
  );
}
