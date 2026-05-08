"use client";

import { useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { BankSelector } from "@/components/bank-selector";
import { useBank } from "@/lib/bank-context";
import { bankRoute } from "@/lib/bank-url";
import { Button } from "@/components/ui/button";
import { LogOut } from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const { currentBank } = useBank();

  const handleLogout = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } finally {
      window.location.href = "/login";
    }
  }, []);

  // Redirect to bank page if a bank is selected
  useEffect(() => {
    if (currentBank) {
      router.push(bankRoute(currentBank, "?view=data"));
    }
  }, [currentBank, router]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <div className="text-sm text-muted-foreground">Hindsight Control Plane</div>
        <Button variant="ghost" size="sm" onClick={handleLogout}>
          <LogOut className="w-4 h-4" />
        </Button>
      </div>

      <div className="flex-1 flex flex-col">
        <BankSelector />

        <div className="flex items-center justify-center h-[calc(100vh-80px)] bg-muted/20">
          <div className="text-center p-10 bg-card rounded-lg border-2 border-border shadow-lg max-w-md">
            <h3 className="text-2xl font-bold mb-3 text-card-foreground">Welcome to Hindsight</h3>
            <p className="text-muted-foreground mb-4">
              Select a memory bank from the dropdown above to get started.
            </p>
            <div className="text-6xl mb-4">🧠</div>
            <p className="text-sm text-muted-foreground">
              The sidebar will appear once you select a memory bank.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
