import type { Metadata } from "next";
import "./globals.css";
import { BankProvider } from "@/lib/bank-context";
import { ThemeProvider } from "@/lib/theme-context";

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
          <BankProvider>
            {children}
          </BankProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
