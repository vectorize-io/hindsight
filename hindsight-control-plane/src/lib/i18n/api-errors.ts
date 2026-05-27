import { defaultLocale, locales, type Locale } from "@/i18n/config";

import de from "@/messages/de.json";
import en from "@/messages/en.json";
import es from "@/messages/es.json";
import fr from "@/messages/fr.json";
import ja from "@/messages/ja.json";
import ko from "@/messages/ko.json";
import pt from "@/messages/pt.json";
import yueHant from "@/messages/yue-Hant.json";
import zhCN from "@/messages/zh-CN.json";
import zhTW from "@/messages/zh-TW.json";

type CookieValue = string | { value?: string } | undefined;
export type RequestLike = {
  headers?: Headers;
  cookies?: {
    get: (name: string) => CookieValue;
  };
};

type ApiErrorPayload = Record<string, unknown> & {
  errorKey: string;
};

const resources = {
  de,
  en,
  es,
  fr,
  ja,
  ko,
  pt,
  "zh-CN": zhCN,
  "zh-TW": zhTW,
  "yue-Hant": yueHant,
} satisfies Record<Locale, Record<string, unknown>>;

const localeSet = new Set<string>(locales);
const localizedErrorFields = ["error", "message", "detail", "details"] as const;
const localeCookieNames = ["NEXT_LOCALE", "hindsight_cp_locale"] as const;

const localeAliases: Partial<Record<string, Locale>> = {
  "zh-Hans": "zh-CN",
  "zh-Hans-CN": "zh-CN",
  "zh-SG": "zh-CN",
  "zh-Hant": "zh-TW",
  "zh-Hant-TW": "zh-TW",
  yue: "yue-Hant",
  "yue-Hant-HK": "yue-Hant",
  "yue-Hant-MO": "yue-Hant",
  "yue-HK": "yue-Hant",
  "yue-MO": "yue-Hant",
  "zh-HK": "yue-Hant",
  "zh-Hant-HK": "yue-Hant",
  "zh-MO": "yue-Hant",
  "zh-Hant-MO": "yue-Hant",
} satisfies Record<string, Locale>;

function parseCookieHeader(cookieHeader: string | null | undefined): Map<string, string> {
  const cookies = new Map<string, string>();
  if (!cookieHeader) return cookies;

  for (const cookie of cookieHeader.split(";")) {
    const separatorIndex = cookie.indexOf("=");
    if (separatorIndex === -1) continue;

    const key = cookie.slice(0, separatorIndex).trim();
    const value = cookie.slice(separatorIndex + 1).trim();
    if (!key) continue;

    try {
      cookies.set(key, decodeURIComponent(value));
    } catch {
      cookies.set(key, value);
    }
  }

  return cookies;
}

function getCookieLocale(request: RequestLike | undefined): string | undefined {
  const parsedCookies = parseCookieHeader(request?.headers?.get("cookie"));

  for (const cookieName of localeCookieNames) {
    const cookieValue = request?.cookies?.get(cookieName);
    if (typeof cookieValue === "string") return cookieValue;
    if (cookieValue?.value) return cookieValue.value;

    const parsedValue = parsedCookies.get(cookieName);
    if (parsedValue) return parsedValue;
  }

  return undefined;
}

function parseAcceptLanguage(header: string | null | undefined): string[] {
  if (!header) return [];

  return header
    .split(",")
    .map((entry) => {
      const [tag, ...parameters] = entry.trim().split(";");
      const quality = parameters
        .map((parameter) => parameter.trim())
        .find((parameter) => parameter.startsWith("q="));
      const parsedQuality = quality ? Number.parseFloat(quality.slice(2)) : 1;

      return {
        tag,
        quality: Number.isFinite(parsedQuality) ? parsedQuality : 0,
      };
    })
    .filter(({ tag }) => tag.length > 0)
    .sort((a, b) => b.quality - a.quality)
    .map(({ tag }) => tag);
}

function resolveLocaleTag(value: string | undefined): Locale | undefined {
  if (!value) return undefined;
  if (localeSet.has(value)) return value as Locale;

  const alias = localeAliases[value];
  if (alias) return alias;

  const normalized = value.toLowerCase();
  const exact = locales.find((locale) => locale.toLowerCase() === normalized);
  if (exact) return exact;

  const normalizedAlias = Object.entries(localeAliases).find(
    ([aliasValue]) => aliasValue.toLowerCase() === normalized
  )?.[1];
  if (normalizedAlias) return normalizedAlias;

  const language = value.split("-")[0];
  if (language === "yue") return "yue-Hant";
  if (language === "zh") return "zh-CN";

  return locales.find((locale) => locale.split("-")[0] === language);
}

export function resolveApiErrorLocale(request?: RequestLike): Locale {
  const candidates = [
    getCookieLocale(request),
    ...parseAcceptLanguage(request?.headers?.get("accept-language")),
  ];

  for (const candidate of candidates) {
    const locale = resolveLocaleTag(candidate);
    if (locale) return locale;
  }

  return defaultLocale;
}

function getTranslationValue(locale: Locale, key: string): string | undefined {
  let current: unknown = resources[locale];

  for (const part of key.split(".")) {
    if (!current || typeof current !== "object" || !(part in current)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[part];
  }

  return typeof current === "string" ? current : undefined;
}

function getFallbackMessage(payload: ApiErrorPayload): string | undefined {
  for (const field of localizedErrorFields) {
    const value = payload[field];
    if (typeof value === "string" && value.trim().length > 0) {
      return value;
    }
  }

  return undefined;
}

export function translateApiError(
  errorKey: string,
  fallbackMessage?: string,
  request?: RequestLike
): string {
  const locale = resolveApiErrorLocale(request);

  return (
    getTranslationValue(locale, errorKey) ??
    getTranslationValue(defaultLocale, errorKey) ??
    fallbackMessage ??
    errorKey
  );
}

export function localizeApiErrorPayload<TPayload extends ApiErrorPayload>(
  request: RequestLike | undefined,
  payload: TPayload
): TPayload {
  const message = translateApiError(payload.errorKey, getFallbackMessage(payload), request);
  const localizedPayload: Record<string, unknown> = { ...payload };
  delete localizedPayload.errorKey;
  let updatedExistingField = false;

  for (const field of localizedErrorFields) {
    if (field in localizedPayload && typeof localizedPayload[field] === "string") {
      localizedPayload[field] = message;
      updatedExistingField = true;
    }
  }

  if (!updatedExistingField) {
    localizedPayload.error = message;
  }

  return localizedPayload as TPayload;
}
