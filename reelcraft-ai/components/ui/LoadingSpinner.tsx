import { cn } from "@/lib/utils";

export function LoadingSpinner({ className, size = "md" }: { className?: string; size?: "sm" | "md" | "lg" }) {
  const sizes = { sm: "w-4 h-4 border-2", md: "w-8 h-8 border-2", lg: "w-12 h-12 border-3" };
  return (
    <div className={cn(
      "rounded-full border-white/10 animate-spin",
      sizes[size],
      className
    )} style={{ borderTopColor: "#c026d3" }} />
  );
}

export function PageLoader({ message }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-24">
      <LoadingSpinner size="lg" />
      {message && <p className="text-gray-400 text-sm animate-pulse">{message}</p>}
    </div>
  );
}
