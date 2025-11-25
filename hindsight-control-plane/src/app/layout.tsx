import type { Metadata } from "next";
import "./globals.css";
import { AgentProvider } from "@/lib/agent-context";

export const metadata: Metadata = {
  title: "Hindsight Control Plane",
  description: "Control plane for the temporal semantic memory system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <AgentProvider>
          {children}
        </AgentProvider>
      </body>
    </html>
  );
}
