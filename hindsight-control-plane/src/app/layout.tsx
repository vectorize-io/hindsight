import type { Metadata } from "next";
import "./globals.css";
import { BankProvider } from "@/lib/bank-context";
import { FeaturesProvider } from "@/lib/features-context";
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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="bg-background text-foreground">
        <ThemeProvider>
          <I18nProvider>
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
