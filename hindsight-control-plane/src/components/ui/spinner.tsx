import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

// Shared loading spinner. Wraps lucide's Loader2 (the app's de-facto loader) so
// every loading state renders one consistent, accessible spinner instead of an
// ad-hoc animate-spin emoji. Pass `size` for the common sizes, or override the
// dimensions/colour via `className`.
const SIZES = {
  xs: "w-3 h-3",
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-7 h-7",
  xl: "w-10 h-10",
} as const;

export type SpinnerSize = keyof typeof SIZES;

export function Spinner({ size = "md", className }: { size?: SpinnerSize; className?: string }) {
  return (
    <Loader2
      className={cn(SIZES[size], "animate-spin text-muted-foreground", className)}
      role="status"
      aria-label="Loading"
    />
  );
}
