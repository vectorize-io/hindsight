"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Film, Trash2, Plus, Clock, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import type { Project } from "@/types";
import { Badge } from "@/components/ui/Badge";
import { PageLoader } from "@/components/ui/LoadingSpinner";
import { formatRelativeTime, formatDuration } from "@/lib/utils";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);

  const fetchProjects = () => {
    fetch("/api/projects", { headers: { "x-user-id": "reelcraft-user-default" } })
      .then((r) => r.json())
      .then((d) => { if (d.success) setProjects(d.data.projects); })
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchProjects(); }, []);

  const deleteProject = async (id: string) => {
    if (!confirm("Delete this project? This cannot be undone.")) return;
    setDeleting(id);
    await fetch(`/api/projects?id=${id}`, { method: "DELETE" });
    fetchProjects();
    setDeleting(null);
  };

  if (loading) return <PageLoader message="Loading projects…" />;

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-display font-bold text-white mb-1">Projects</h1>
          <p className="text-gray-400">All your analyzed Reels and recreation guides.</p>
        </div>
        <Link href="/analyze" className="btn-primary">
          <Plus className="w-4 h-4" />
          Analyze Reel
        </Link>
      </div>

      {projects.length === 0 ? (
        <div className="glass rounded-2xl p-16 text-center">
          <Film className="w-14 h-14 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-white mb-2">No projects yet</h3>
          <p className="text-gray-400 mb-8 max-w-sm mx-auto">
            Upload your first Reel to get a complete production blueprint.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link href="/analyze" className="btn-primary">
              <Film className="w-4 h-4" />
              Analyze Your First Reel
            </Link>
            <Link href="/demo" className="btn-secondary">
              Try the Demo
            </Link>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {projects.map((project) => (
            <div key={project.id} className="glass-hover rounded-2xl p-5 relative group">
              {/* Delete button */}
              <button
                onClick={() => deleteProject(project.id)}
                disabled={deleting === project.id}
                className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg hover:bg-red-500/10 text-gray-600 hover:text-red-400"
              >
                {deleting === project.id
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <Trash2 className="w-4 h-4" />}
              </button>

              <Link href={`/projects/${project.id}`} className="block">
                {/* Status icon */}
                <div className="flex items-start gap-3 mb-3">
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ background: "linear-gradient(135deg, rgba(192,38,211,0.3), rgba(234,88,12,0.3))" }}>
                    {project.status === "complete" ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : project.status === "analyzing" ? (
                      <Loader2 className="w-5 h-5 text-brand-400 animate-spin" />
                    ) : project.status === "error" ? (
                      <AlertCircle className="w-5 h-5 text-red-400" />
                    ) : (
                      <Film className="w-5 h-5 text-gray-400" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap gap-1.5 mb-1">
                      {project.isDemo && <Badge variant="yellow">Demo</Badge>}
                      <Badge variant={
                        project.status === "complete" ? "green" :
                        project.status === "error" ? "red" : "default"
                      }>
                        {project.status}
                      </Badge>
                      {project.guide?.section1_overview.contentCategory && (
                        <Badge variant="brand">{project.guide.section1_overview.contentCategory}</Badge>
                      )}
                    </div>
                    <h2 className="text-sm font-semibold text-white line-clamp-2 group-hover:text-brand-300 transition-colors">
                      {project.title}
                    </h2>
                  </div>
                </div>

                {/* Meta */}
                <div className="flex flex-wrap gap-3 text-xs text-gray-500">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {formatRelativeTime(project.createdAt)}
                  </span>
                  {project.metadata && (
                    <span>{formatDuration(project.metadata.duration)}</span>
                  )}
                  {project.metadata && (
                    <span>{project.metadata.aspectRatio}</span>
                  )}
                  {project.guide?.section2_skillLevel && (
                    <span className={
                      project.guide.section2_skillLevel.level === "Beginner" ? "text-green-400" :
                      project.guide.section2_skillLevel.level === "Intermediate" ? "text-yellow-400" : "text-red-400"
                    }>
                      {project.guide.section2_skillLevel.level}
                    </span>
                  )}
                </div>

                {project.guide?.section1_overview.about && (
                  <p className="text-xs text-gray-500 mt-2 line-clamp-2">
                    {project.guide.section1_overview.about}
                  </p>
                )}

                {project.status === "complete" && (
                  <div className="mt-3 pt-3 border-t border-white/[0.05] flex items-center gap-1 text-xs text-brand-400">
                    <span>View recreation guide</span>
                    <span>→</span>
                  </div>
                )}
              </Link>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
