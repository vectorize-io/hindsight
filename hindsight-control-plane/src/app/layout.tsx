import type { Metadata } from "next";
import { cookies, headers } from "next/headers";
import "./globals.css";
import { BankProvider } from "@/lib/bank-context";
import { FeaturesProvider } from "@/lib/features-context";
import {
  defaultLocale,
  isSupportedLocale,
  languageCookieName,
  parseAcceptLanguage,
  resolveSupportedLocale,
} from "@/lib/i18n/resources";
import { I18nProvider } from "@/lib/i18n/provider";
import { ThemeProvider } from "@/lib/theme-context";
import { Toaster } from "@/components/ui/sonner";

export const metadata: Metadata = {
  title: "Hindsight Control Plane",
  description: "Control plane for the temporal semantic memory system",
  icons: {
    icon: "/favicon.png",
  },
};

async function getInitialLocale() {
  const [cookieStore, headerStore] = await Promise.all([cookies(), headers()]);
  const storedLocale = cookieStore.get(languageCookieName)?.value;

  if (isSupportedLocale(storedLocale)) {
    return storedLocale;
  }

  const acceptedLocales = parseAcceptLanguage(headerStore.get("accept-language"));
  return acceptedLocales.length > 0 ? resolveSupportedLocale(acceptedLocales) : defaultLocale;
}

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const initialLocale = await getInitialLocale();

  return (
    <html lang={initialLocale} suppressHydrationWarning>
      <body className="bg-background text-foreground">
        <ThemeProvider>
          <I18nProvider initialLocale={initialLocale}>
            <FeaturesProvider>
              <BankProvider>{children}</BankProvider>
            </FeaturesProvider>
          </I18nProvider>
        </ThemeProvider>
        <Toaster />
      </body>
    </html>
  );
}
