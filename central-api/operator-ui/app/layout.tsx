import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "CollabMind Operator",
  description: "Operator GUI v0.1 — governed control plane",
};

const NAV = [
  ["/", "Dashboard"],
  ["/login", "Login"],
  ["/connection", "Connection"],
  ["/workspaces", "Workspaces"],
  ["/connectors", "Connectors"],
  ["/google-drive", "Google Drive"],
  ["/files", "Indexed Files"],
  ["/ingestion", "Ingestion Queue"],
  ["/audit", "Audit Log"],
  ["/agent-activity", "Agent Activity"],
  ["/permissions", "Permissions View"],
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="layout">
          <aside className="sidebar">
            <h1>CollabMind</h1>
            <div className="sub">Operator · v0.1</div>
            <nav>
              {NAV.map(([href, label]) => (
                <Link key={href} href={href}>
                  {label}
                </Link>
              ))}
            </nav>
          </aside>
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
