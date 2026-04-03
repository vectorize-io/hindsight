"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
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

  const loadBanks = async () => {
    try {
      const response = await client.listBanks();
      // Extract bank_id from each bank object
      const bankIds = response.banks?.map((bank: any) => bank.bank_id) || [];
      setBanks(bankIds);
    } catch (error) {
      console.error("Error loading banks:", error);
    }
  };

  // Initialize bank from URL on mount
  useEffect(() => {
    const bankMatch = pathname?.match(/^\/banks\/([^/?]+)/);
    if (bankMatch) {
      setCurrentBank(decodeURIComponent(bankMatch[1]));
    }
  }, [pathname]);

  useEffect(() => {
    setCurrentBank(null);
    loadBanks();
  }, [currentTenant]);

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
