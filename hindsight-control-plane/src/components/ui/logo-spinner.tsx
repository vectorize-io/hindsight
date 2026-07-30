import { cn } from "@/lib/utils";
import { withBasePath } from "@/lib/base-path";

// Branded loading spinner: the Hindsight logo mark doing a looping "tumble"
// (crouch → hop + 360° flip → land squash). Use it for prominent, centered
// loading states where the mascot has room to bounce; keep the compact lucide
// `Spinner` for tiny inline spots (buttons, table rows) where a hopping logo
// would be too busy. The animation lives in globals.css (`.animate-logo-tumble`).
// /favicon.png is the mark alone (no wordmark), which reads best at these sizes.
const SIZES = {
  sm: "w-6 h-6",
  md: "w-9 h-9",
  lg: "w-12 h-12",
  xl: "w-16 h-16",
} as const;

export type LogoSpinnerSize = keyof typeof SIZES;

export function LogoSpinner({
  size = "md",
  className,
}: {
  size?: LogoSpinnerSize;
  className?: string;
}) {
  return (
    // eslint-disable-next-line @next/next/no-img-element -- animated CSS transforms on the raw <img>; next/image wraps it in a positioned span that fights the tumble.
    <img
      src={withBasePath("/favicon.png")}
      alt=""
      role="status"
      aria-label="Loading"
      className={cn(SIZES[size], "animate-logo-tumble select-none object-contain", className)}
    />
  );
}
