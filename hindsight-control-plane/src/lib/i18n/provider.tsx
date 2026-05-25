"use client";

import * as React from "react";
import i18next from "i18next";
import { I18nextProvider, initReactI18next } from "react-i18next";
import {
  defaultLocale,
  isSupportedLocale,
  languageStorageKey,
  resolveSupportedLocale,
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
const localeHydrationDelayMs = 750;

function ensureI18nInitialized() {
  if (i18next.isInitialized) {
    if (i18next.language !== defaultLocale) {
      void i18next.changeLanguage(defaultLocale);
    }
    return;
  }

  void i18next.use(initReactI18next).init({
    resources,
    lng: defaultLocale,
    fallbackLng: defaultLocale,
    supportedLngs: supportedLocales.map((locale) => locale.code),
    interpolation: {
      escapeValue: false,
    },
    react: {
      useSuspense: false,
    },
  });
}

function readPreferredLocale(): SupportedLocale {
  if (typeof window === "undefined") return defaultLocale;

  const storedLocale = window.localStorage.getItem(languageStorageKey);
  if (isSupportedLocale(storedLocale)) return storedLocale;

  return resolveSupportedLocale([...(window.navigator.languages ?? []), window.navigator.language]);
}

function writeDocumentLocale(locale: SupportedLocale) {
  if (typeof document === "undefined") return;
  document.documentElement.lang = locale;
}

ensureI18nInitialized();

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = React.useState<SupportedLocale>(defaultLocale);

  const setLocale = React.useCallback((nextLocale: SupportedLocale) => {
    setLocaleState(nextLocale);
    writeDocumentLocale(nextLocale);

    if (typeof window !== "undefined") {
      window.localStorage.setItem(languageStorageKey, nextLocale);
    }

    void i18next.changeLanguage(nextLocale);
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    let timeoutId: number | undefined;
    let frameId: number | undefined;
    let secondFrameId: number | undefined;

    const applyPreferredLocale = () => {
      // Keep the first client pass aligned with SSR output before applying
      // local/browser preferences, otherwise late-hydrating client components
      // can see translated text before React has attached to the English HTML.
      timeoutId = window.setTimeout(() => {
        if (!cancelled) {
          setLocale(readPreferredLocale());
        }
      }, localeHydrationDelayMs);
    };

    const scheduleAfterFrame = () => {
      frameId = window.requestAnimationFrame(() => {
        secondFrameId = window.requestAnimationFrame(applyPreferredLocale);
      });
    };

    if (document.readyState === "complete") {
      scheduleAfterFrame();
    } else {
      window.addEventListener("load", scheduleAfterFrame, { once: true });
    }

    return () => {
      cancelled = true;
      window.removeEventListener("load", scheduleAfterFrame);
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
      if (frameId !== undefined) window.cancelAnimationFrame(frameId);
      if (secondFrameId !== undefined) window.cancelAnimationFrame(secondFrameId);
    };
  }, [setLocale]);

  React.useEffect(() => {
    const handleLanguageChanged = (language: string) => {
      const nextLocale = isSupportedLocale(language) ? language : defaultLocale;
      setLocaleState(nextLocale);
      writeDocumentLocale(nextLocale);
    };

    i18next.on("languageChanged", handleLanguageChanged);
    return () => {
      i18next.off("languageChanged", handleLanguageChanged);
    };
  }, []);

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
      <I18nextProvider i18n={i18next}>{children}</I18nextProvider>
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
