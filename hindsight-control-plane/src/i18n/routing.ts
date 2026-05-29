import { defineRouting } from "next-intl/routing";
import { locales, defaultLocale } from "./config";

export const routing = defineRouting({
  locales,
  defaultLocale,
  // Don't add locale prefix for the default locale (English)
  // /dashboard instead of /en/dashboard
  localePrefix: "as-needed",
});
