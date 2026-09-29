import type { LucideIcon } from "lucide-react";
import { PowerOff } from "lucide-react";

interface FeatureNotEnabledProps {
  /** Headline, e.g. "Audit Logs Not Enabled". */
  title: string;
  /** Supporting copy. Accepts rich nodes so callers can embed an env-var <code> snippet. */
  description: React.ReactNode;
  /** Splash icon. Defaults to a "power off" glyph. */
  icon?: LucideIcon;
}

/**
 * Centered splash shown in place of a feature's view when that feature is
 * disabled on the server. Reused across observations, audit logs, LLM
 * requests, and any other server-gated feature.
 */
export function FeatureNotEnabled({
  title,
  description,
  icon: Icon = PowerOff,
}: FeatureNotEnabledProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="text-muted-foreground mb-2">
        <Icon width={48} height={48} strokeWidth={1.5} />
      </div>
      <h3 className="text-lg font-semibold text-foreground mb-1">{title}</h3>
      <p className="text-sm text-muted-foreground max-w-md">{description}</p>
    </div>
  );
}
