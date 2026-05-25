"use client";

import * as React from "react";
import { createInstance, type i18n as I18nInstance } from "i18next";
import { I18nextProvider, initReactI18next } from "react-i18next";
import {
  defaultLocale,
  isSupportedLocale,
  languageCookieName,
  languageStorageKey,
  resources,
  supportedLocales,
  type SupportedLocale,
} from "./resources";

interface LocaleContextValue {
  locale: SupportedLocale;
  setLocale: (locale: SupportedLocale) => void;
  supportedLocales: typeof supportedLocales;
}

const LocaleContext = React.createContext<LocaleContextValue | undefined>(undefined);
const languageCookieMaxAgeSeconds = 60 * 60 * 24 * 365;

function createI18n(initialLocale: SupportedLocale): I18nInstance {
  const instance = createInstance();

  void instance.use(initReactI18next).init({
    resources,
    lng: initialLocale,
    fallbackLng: defaultLocale,
    supportedLngs: supportedLocales.map((locale) => locale.code),
    initAsync: false,
    interpolation: {
      escapeValue: false,
    },
    react: {
      useSuspense: false,
    },
  });
  return instance;
}

function writeDocumentLocale(locale: SupportedLocale) {
  if (typeof document === "undefined") return;
  document.documentElement.lang = locale;
}

function writeStoredLocale(locale: SupportedLocale) {
  if (typeof window === "undefined") return;

  try {
    window.localStorage.setItem(languageStorageKey, locale);
  } catch {
    // Storage may be unavailable in private browsing or locked-down environments.
  }
}

function writeLocaleCookie(locale: SupportedLocale) {
  if (typeof document === "undefined") return;
  document.cookie = `${languageCookieName}=${encodeURIComponent(
    locale
  )}; Path=/; Max-Age=${languageCookieMaxAgeSeconds}; SameSite=Lax`;
}

interface I18nProviderProps {
  children: React.ReactNode;
  initialLocale: SupportedLocale;
}

export function I18nProvider({ children, initialLocale }: I18nProviderProps) {
  const [i18n] = React.useState(() => createI18n(initialLocale));
  const [locale, setLocaleState] = React.useState<SupportedLocale>(initialLocale);

  const setLocale = React.useCallback(
    (nextLocale: SupportedLocale) => {
      setLocaleState(nextLocale);
      writeDocumentLocale(nextLocale);
      writeStoredLocale(nextLocale);
      writeLocaleCookie(nextLocale);

      void i18n.changeLanguage(nextLocale);
    },
    [i18n]
  );

  React.useEffect(() => {
    const handleLanguageChanged = (language: string) => {
      const nextLocale = isSupportedLocale(language) ? language : defaultLocale;
      setLocaleState(nextLocale);
      writeDocumentLocale(nextLocale);
    };

    i18n.on("languageChanged", handleLanguageChanged);
    return () => {
      i18n.off("languageChanged", handleLanguageChanged);
    };
  }, [i18n]);

  const value = React.useMemo(
    () => ({
      locale,
      setLocale,
      supportedLocales,
    }),
    [locale, setLocale]
  );

  return (
    <LocaleContext.Provider value={value}>
      <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  const context = React.useContext(LocaleContext);
  if (!context) {
    throw new Error("useLocale must be used within an I18nProvider");
  }
  return context;
}
