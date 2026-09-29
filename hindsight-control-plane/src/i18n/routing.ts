import { defineRouting } from "next-intl/routing";
import { locales, defaultLocale } from "./config";

export const routing = defineRouting({
  locales,
  defaultLocale,
  // Never expose the locale in the URL. The active locale is resolved from the
  // NEXT_LOCALE cookie (Accept-Language as fallback) and next-intl rewrites
  // internally to the [locale] segment, so paths stay clean (/banks/x, never
  // /es/banks/x). Keeps the control plane locale-agnostic in the address bar.
  localePrefix: "never",
});
