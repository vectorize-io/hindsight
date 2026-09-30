import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/layout/Navbar";

export const metadata: Metadata = {
  title: "ReelCraft AI — AI-Powered Reel Recreation Guide",
  description:
    "Upload any Reel and get a complete, step-by-step production blueprint to recreate it. Powered by AI and personalized memory.",
  keywords: ["reel creation", "content creator", "AI video analysis", "social media", "reels guide"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="min-h-screen bg-[#0a0a0f] text-white antialiased">
        <Navbar />
        <main className="pt-16">{children}</main>
      </body>
    </html>
  );
}
