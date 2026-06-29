"use client";

import { useEffect } from "react";
import { whitelabelConfig } from "@/lib/whitelabel-config";

/**
 * BrandStyleInjector
 *
 * Injects brand colors as CSS custom properties into the document root.
 * This allows the whitelabel configuration to override default Tailwind colors
 * and enables dynamic theming based on environment variables.
 *
 * Mount this component once in the root layout.
 */
export function BrandStyleInjector() {
  useEffect(() => {
    const root = document.documentElement;

    // Inject brand color CSS variables
    root.style.setProperty("--brand-primary", whitelabelConfig.colors.primary);
    root.style.setProperty("--brand-secondary", whitelabelConfig.colors.secondary);
    root.style.setProperty("--brand-accent", whitelabelConfig.colors.accent);

    // Update the primary gradient to use brand colors
    root.style.setProperty(
      "--primary-gradient",
      `linear-gradient(135deg, ${whitelabelConfig.colors.primary} 0%, ${whitelabelConfig.colors.secondary} 100%)`
    );
  }, []);

  return null;
}
