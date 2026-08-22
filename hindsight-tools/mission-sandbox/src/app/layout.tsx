import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Mission Sandbox",
  description: "Iterate on Hindsight observation missions with a fast feedback loop.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
