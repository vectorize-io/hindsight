"use client";

import * as React from "react";
import * as SwitchPrimitives from "@radix-ui/react-switch";

import { cn } from "@/lib/utils";

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => (
  // The checked track is the brand gradient, which is a plain CSS class rather
  // than a generated utility, so `data-[state=...]` variants can't reach it.
  // The track colors for both states are wired in globals.css off this
  // data-slot; don't set them here.
  <SwitchPrimitives.Root
    data-slot="hs-switch"
    className={cn(
      "peer inline-flex h-[22px] w-[40px] shrink-0 cursor-pointer items-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/40 disabled:cursor-not-allowed disabled:opacity-50",
      className
    )}
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb
      className={cn(
        "pointer-events-none block h-[18px] w-[18px] rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.35)] ring-0 transition-transform data-[state=checked]:translate-x-[20px] data-[state=unchecked]:translate-x-[2px]"
      )}
    />
  </SwitchPrimitives.Root>
));
Switch.displayName = SwitchPrimitives.Root.displayName;

export { Switch };
