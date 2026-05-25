import en from "./en";
import zhCN from "./zh-CN";

export const defaultLocale = "en";
export const languageCookieName = "hindsight_cp_locale";
export const languageStorageKey = "hindsight_cp_locale";

export const supportedLocales = [
  {
    code: "en",
    label: "English",
    nativeLabel: "English",
  },
  {
    code: "zh-CN",
    label: "Chinese (Simplified)",
    nativeLabel: "简体中文",
  },
] as const;

export type SupportedLocale = (typeof supportedLocales)[number]["code"];

export const resources = {
  en: {
    translation: en,
  },
  "zh-CN": {
    translation: zhCN,
  },
} satisfies Record<SupportedLocale, { translation: typeof en }>;

const localeAliases = {
  zh: "zh-CN",
  "zh-Hans": "zh-CN",
  "zh-SG": "zh-CN",
} satisfies Record<string, SupportedLocale>;

function resolveLocaleAlias(value: string): SupportedLocale | undefined {
  return localeAliases[value as keyof typeof localeAliases];
}

export function isSupportedLocale(value: string | null | undefined): value is SupportedLocale {
  return supportedLocales.some((locale) => locale.code === value);
}

export function resolveSupportedLocale(values: Array<string | null | undefined>): SupportedLocale {
  for (const value of values) {
    if (!value) continue;
    if (isSupportedLocale(value)) return value;
    const alias = resolveLocaleAlias(value);
    if (alias) return alias;

    const baseLocale = value.split("-")[0];
    if (isSupportedLocale(baseLocale)) return baseLocale;
    const baseAlias = resolveLocaleAlias(baseLocale);
    if (baseAlias) return baseAlias;
  }

  return defaultLocale;
}

export function parseAcceptLanguage(header: string | null | undefined): string[] {
  if (!header) return [];

  return header
    .split(",")
    .map((entry) => {
      const [locale = "", ...params] = entry.trim().split(";");
      const quality = params.reduce((currentQuality, param) => {
        const [key, value] = param.trim().split("=");
        const nextQuality = Number.parseFloat(value ?? "");
        return key === "q" && Number.isFinite(nextQuality) ? nextQuality : currentQuality;
      }, 1);

      return {
        locale: locale.trim(),
        quality,
      };
    })
    .filter(({ locale, quality }) => locale.length > 0 && quality > 0)
    .sort((left, right) => right.quality - left.quality)
    .map(({ locale }) => locale);
}
