"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import { usePathname } from "next/navigation";
import { client } from "./api";
import { useTenant } from "./tenant-context";

interface BankContextType {
  currentBank: string | null;
  setCurrentBank: (bank: string | null) => void;
  banks: string[];
  loadBanks: () => Promise<void>;
}

const BankContext = createContext<BankContextType | undefined>(undefined);

export function BankProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { currentTenant } = useTenant();
  const [currentBank, setCurrentBank] = useState<string | null>(null);
  const [banks, setBanks] = useState<string[]>([]);
  // Guard against stale responses when tenant switches rapidly
  const loadIdRef = useRef(0);

  const loadBanks = useCallback(async () => {
    const id = ++loadIdRef.current;
    try {
      const response = await client.listBanks();
      // Drop the result if a newer load was started while we were waiting
      if (id !== loadIdRef.current) return;
      const bankIds = response.banks?.map((bank: any) => bank.bank_id) || [];
      setBanks(bankIds);
    } catch (error) {
      if (id !== loadIdRef.current) return;
      console.error("Error loading banks:", error);
    }
  }, []);

  // Initialize bank from URL on mount
  useEffect(() => {
    const bankMatch = pathname?.match(/^\/banks\/([^/?]+)/);
    if (bankMatch) {
      setCurrentBank(decodeURIComponent(bankMatch[1]));
    }
  }, [pathname]);

  useEffect(() => {
    if (currentTenant === null) return;
    setCurrentBank(null);
    loadBanks();
  }, [currentTenant, loadBanks]);

  return (
    <BankContext.Provider value={{ currentBank, setCurrentBank, banks, loadBanks }}>
      {children}
    </BankContext.Provider>
  );
}

export function useBank() {
  const context = useContext(BankContext);
  if (context === undefined) {
    throw new Error("useBank must be used within a BankProvider");
  }
  return context;
}
