import type { Metadata } from "next";
import "./globals.css";
import { whitelabelConfig } from "@/lib/whitelabel-config";

export const metadata: Metadata = {
  title: whitelabelConfig.metadata.title,
  description: whitelabelConfig.metadata.description,
  icons: {
    icon: "/favicon.png",
  },
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 5,
    userScalable: true,
  },
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // The root layout is a minimal shell. Locale-aware content,
  // providers, and <html lang> are handled in [locale]/layout.tsx.
  return children;
}
