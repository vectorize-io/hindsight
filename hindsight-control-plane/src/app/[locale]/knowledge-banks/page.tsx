"use client";

import { useEffect } from "react";
import { BankSelector } from "@/components/bank-selector";
import { KnowledgeBanksList } from "@/components/knowledge-banks-list";
import { useBank } from "@/lib/bank-context";

export default function KnowledgeBanksPage() {
  const { setCurrentBank } = useBank();

  // No memory bank is on screen here, so none stays selected in the header state.
  useEffect(() => {
    setCurrentBank(null);
  }, [setCurrentBank]);

  return (
    <div className="min-h-dvh bg-background flex flex-col">
      <BankSelector kind="knowledge" />
      <KnowledgeBanksList />
    </div>
  );
}
