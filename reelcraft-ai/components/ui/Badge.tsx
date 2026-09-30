import { cn } from "@/lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "brand" | "accent" | "green" | "yellow" | "red" | "blue";
  className?: string;
}

const variants: Record<string, string> = {
  default: "bg-white/10 text-gray-300 border-white/15",
  brand: "bg-brand-500/15 text-brand-300 border-brand-500/30",
  accent: "bg-accent-500/15 text-accent-300 border-accent-500/30",
  green: "bg-green-500/15 text-green-300 border-green-500/30",
  yellow: "bg-yellow-500/15 text-yellow-300 border-yellow-500/30",
  red: "bg-red-500/15 text-red-300 border-red-500/30",
  blue: "bg-blue-500/15 text-blue-300 border-blue-500/30",
};

export function Badge({ children, variant = "default", className }: BadgeProps) {
  return (
    <span className={cn("badge", variants[variant], className)}>
      {children}
    </span>
  );
}
