import type { Technique } from "@/types";
import { WhyButton } from "@/components/ui/WhyButton";

export function TechniqueCards({ techniques }: { techniques: Technique[] }) {
  if (!techniques.length) return <p className="text-gray-500 text-sm">No techniques identified.</p>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {techniques.map((tech, i) => (
        <div key={i} className="glass rounded-xl p-4 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <h4 className="text-sm font-bold text-white">{tech.name}</h4>
            <WhyButton explanation={tech.why} />
          </div>

          <div className="space-y-2 text-xs">
            <div>
              <span className="text-brand-400 font-semibold uppercase tracking-wide text-[10px]">What</span>
              <p className="text-gray-300 mt-0.5">{tech.what}</p>
            </div>
            <div>
              <span className="text-accent-400 font-semibold uppercase tracking-wide text-[10px]">How to do it</span>
              <p className="text-gray-300 mt-0.5">{tech.how}</p>
            </div>
            {tech.alternative && (
              <div>
                <span className="text-green-400 font-semibold uppercase tracking-wide text-[10px]">Simpler alternative</span>
                <p className="text-gray-400 mt-0.5">{tech.alternative}</p>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
