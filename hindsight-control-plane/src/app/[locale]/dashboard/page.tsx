"use client";

import { useEffect } from "react";
import { BankSelector } from "@/components/bank-selector";
import { BanksOverview } from "@/components/banks-overview";
import { useBank } from "@/lib/bank-context";

export default function DashboardPage() {
  const { setCurrentBank } = useBank();

  // This page is the "all banks" view, so nothing is selected while it is open.
  // The selection survives navigating back here from a bank page otherwise, and the
  // header would keep offering bank-scoped actions for a bank that is not on screen.
  useEffect(() => {
    setCurrentBank(null);
  }, [setCurrentBank]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <div className="flex-1 flex flex-col">
        <BankSelector />
        <BanksOverview />
      </div>
    </div>
  );
}
