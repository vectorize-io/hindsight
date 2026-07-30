import { cn } from "@/lib/utils";
import { withBasePath } from "@/lib/base-path";

// Shared loading spinner — the Hindsight mark as a tumbling mascot. This replaces
// the old lucide Loader2 everywhere so every "please wait" state is on-brand and
// consistent. Two motions:
//   - variant "flip" (default): an in-place 360° flip with a squash pulse and only
//     a tiny vertical bob, so it's safe inside buttons and inline next to text.
//   - variant "jump": the full "tumble" (crouch → hop + flip → land squash). Use it
//     for prominent, centered loading states that have vertical room.
// The keyframes live in globals.css. /favicon.png is the mark alone (no wordmark),
// which reads best at these sizes.
const SIZES = {
  xs: "w-3.5 h-3.5",
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-8 h-8",
  xl: "w-14 h-14",
} as const;

export type SpinnerSize = keyof typeof SIZES;

export function Spinner({
  size = "md",
  variant = "flip",
  className,
}: {
  size?: SpinnerSize;
  variant?: "flip" | "jump";
  className?: string;
}) {
  return (
    // Raw <img>, not next/image: next/image wraps the tag in a positioned span
    // that fights the CSS transform animation (and adds no value for a tiny
    // static, already-optimized PNG served from /public).
    <img
      src={withBasePath("/favicon.png")}
      alt=""
      role="status"
      aria-label="Loading"
      className={cn(
        SIZES[size],
        variant === "jump" ? "animate-logo-tumble" : "animate-logo-flip",
        "inline-block select-none object-contain",
        className
      )}
    />
  );
}
