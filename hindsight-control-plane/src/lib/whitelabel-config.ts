/**
 * Whitelabel Configuration
 * 
 * Centralized branding and configuration for the control plane.
 * All brand-specific values should be sourced from environment variables
 * to support multiple whitelabel deployments (e.g., Hindsight, CollabMinds).
 */

export interface WhitelabelConfig {
  // Brand Identity
  brandName: string;
  brandTagline?: string;
  
  // Visual Assets
  logoUrl: string;
  faviconUrl: string;
  
  // Color Palette (hex codes)
  colors: {
    primary: string;
    secondary: string;
    accent: string;
  };
  
  // External Links
  links: {
    github?: string;
    documentation?: string;
    support?: string;
    website?: string;
  };
  
  // Metadata
  metadata: {
    title: string;
    description: string;
    ogImage?: string;
  };
}

/**
 * Get the current whitelabel configuration from environment variables.
 * Falls back to default Hindsight branding if not configured.
 */
export function getWhitelabelConfig(): WhitelabelConfig {
  const brandName = process.env.NEXT_PUBLIC_BRAND_NAME || 'Hindsight';
  
  return {
    brandName,
    brandTagline: process.env.NEXT_PUBLIC_BRAND_TAGLINE,
    
    logoUrl: process.env.NEXT_PUBLIC_BRAND_LOGO_URL || '/logo.png',
    faviconUrl: process.env.NEXT_PUBLIC_BRAND_FAVICON_URL || '/favicon.png',
    
    colors: {
      primary: process.env.NEXT_PUBLIC_BRAND_PRIMARY_COLOR || '#0074d9',
      secondary: process.env.NEXT_PUBLIC_BRAND_SECONDARY_COLOR || '#009296',
      accent: process.env.NEXT_PUBLIC_BRAND_ACCENT_COLOR || '#f59e0b',
    },
    
    links: {
      github: process.env.NEXT_PUBLIC_BRAND_GITHUB_URL,
      documentation: process.env.NEXT_PUBLIC_BRAND_DOCS_URL,
      support: process.env.NEXT_PUBLIC_BRAND_SUPPORT_URL,
      website: process.env.NEXT_PUBLIC_BRAND_WEBSITE_URL,
    },
    
    metadata: {
      title: process.env.NEXT_PUBLIC_BRAND_META_TITLE || `${brandName} - Memory Management Control Plane`,
      description: process.env.NEXT_PUBLIC_BRAND_META_DESCRIPTION || 
        `Manage and monitor your ${brandName} memory infrastructure`,
      ogImage: process.env.NEXT_PUBLIC_BRAND_OG_IMAGE,
    },
  };
}

/**
 * Get whitelabel config for client-side usage.
 * This function can be called in both server and client components.
 */
export const whitelabelConfig = getWhitelabelConfig();

/**
 * Generate CSS custom properties for brand colors.
 * Use this in global CSS or styled-components.
 */
export function getBrandColorVars(): Record<string, string> {
  const config = getWhitelabelConfig();
  
  return {
    '--brand-primary': config.colors.primary,
    '--brand-secondary': config.colors.secondary,
    '--brand-accent': config.colors.accent,
  };
}

/**
 * Helper to get the primary gradient used throughout the UI.
 */
export function getPrimaryGradient(): string {
  const config = getWhitelabelConfig();
  return `linear-gradient(135deg, ${config.colors.primary} 0%, ${config.colors.secondary} 100%)`;
}
