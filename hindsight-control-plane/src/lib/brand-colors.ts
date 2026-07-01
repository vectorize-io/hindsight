/**
 * Brand Colors
 *
 * Centralized color constants for components that use hardcoded colors.
 * Import these instead of using hex codes directly.
 */

import { whitelabelConfig } from "./whitelabel-config";

/**
 * Primary brand colors from whitelabel configuration
 */
export const BRAND_COLORS = {
  primary: whitelabelConfig.colors.primary,
  secondary: whitelabelConfig.colors.secondary,
  accent: whitelabelConfig.colors.accent,
} as const;

/**
 * Chart colors for data visualizations.
 * These maintain visual distinction while respecting the brand palette.
 */
export const CHART_COLORS = {
  primary: whitelabelConfig.colors.primary,
  secondary: whitelabelConfig.colors.secondary,
  accent: whitelabelConfig.colors.accent,
  // Additional semantic colors
  success: "#10b981",
  warning: "#f59e0b",
  error: "#ef4444",
  info: "#3b82f6",
} as const;

/**
 * Graph visualization colors for different node/link types
 */
export const GRAPH_COLORS = {
  // Link types
  semantic: whitelabelConfig.colors.primary,
  temporal: whitelabelConfig.colors.secondary,
  causal: whitelabelConfig.colors.accent,
  reference: "#8b5cf6",

  // Node defaults
  defaultNode: whitelabelConfig.colors.primary,
  highlightNode: whitelabelConfig.colors.accent,

  // Background/UI
  background: "#ffffff",
  backgroundDark: "#1f2937",
} as const;

/**
 * Get a color palette array for multi-series charts
 */
export function getChartPalette(): string[] {
  return [
    whitelabelConfig.colors.primary,
    whitelabelConfig.colors.secondary,
    whitelabelConfig.colors.accent,
    "#10b981", // success green
    "#8b5cf6", // purple
    "#ec4899", // pink
    "#f59e0b", // amber
    "#06b6d4", // cyan
  ];
}

/**
 * Primary gradient used for backgrounds and highlights
 */
export const PRIMARY_GRADIENT = `linear-gradient(135deg, ${whitelabelConfig.colors.primary} 0%, ${whitelabelConfig.colors.secondary} 100%)`;
