import type { EquipmentItem } from "@/types";
import { CheckCircle, Circle, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export function EquipmentGrid({ equipment }: { equipment: EquipmentItem[] }) {
  const must = equipment.filter((e) => e.category === "MUST HAVE");
  const nice = equipment.filter((e) => e.category === "NICE TO HAVE");
  const optional = equipment.filter((e) => e.category === "OPTIONAL / PROFESSIONAL");

  const Section = ({ title, items, icon, color }: {
    title: string; items: EquipmentItem[];
    icon: React.ReactNode; color: string;
  }) => {
    if (!items.length) return null;
    return (
      <div>
        <div className={cn("flex items-center gap-2 mb-3 text-sm font-semibold", color)}>
          {icon}
          {title}
          <span className="ml-1 px-2 py-0.5 rounded-full text-xs bg-white/5 text-gray-400 font-normal">
            {items.length}
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {items.map((item, i) => (
            <div key={i} className="glass rounded-xl p-3.5">
              <div className="flex items-start gap-2.5">
                <div className={cn("mt-0.5 flex-shrink-0", color)}>{icon}</div>
                <div>
                  <p className="text-sm font-semibold text-white">{item.name}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{item.whyNeeded}</p>
                  {item.cheaperAlternative && (
                    <p className="text-xs text-brand-400 mt-1.5">
                      💡 Alt: {item.cheaperAlternative}
                    </p>
                  )}
                  {item.canSkip && (
                    <span className="inline-block mt-1.5 text-xs text-green-400">Can skip</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <Section title="MUST HAVE" items={must} icon={<CheckCircle className="w-4 h-4" />} color="text-green-400" />
      <Section title="NICE TO HAVE" items={nice} icon={<Circle className="w-4 h-4" />} color="text-yellow-400" />
      <Section title="OPTIONAL / PROFESSIONAL" items={optional} icon={<AlertCircle className="w-4 h-4" />} color="text-gray-400" />
    </div>
  );
}
