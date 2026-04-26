"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import { usePathname, useRouter } from "next/navigation";
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
  const router = useRouter();
  const { currentTenant } = useTenant();
  const [currentBank, setCurrentBank] = useState<string | null>(null);
  const [banks, setBanks] = useState<string[]>([]);
  const [banksLoaded, setBanksLoaded] = useState(false);
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
      setBanksLoaded(true);
    } catch (error) {
      if (id !== loadIdRef.current) return;
      console.error("Error loading banks:", error);
    }
  }, []);

  // Keep currentBank in sync with the URL. Routing is the source of truth —
  // navigating to /dashboard or another non-bank page clears it; navigating to
  // /banks/<id> sets it. Driving this from the URL avoids racing with the
  // tenant-switch effect below.
  useEffect(() => {
    const bankMatch = pathname?.match(/^\/banks\/([^/?]+)/);
    setCurrentBank(bankMatch ? decodeURIComponent(bankMatch[1]) : null);
  }, [pathname]);

  // Reload the bank list when the active tenant changes.
  useEffect(() => {
    if (currentTenant === null) return;
    setBanksLoaded(false);
    loadBanks();
  }, [currentTenant, loadBanks]);

  // Direct-link guard: a bookmarked /banks/<id> URL may target a bank from a
  // different tenant than the one restored from localStorage. Once the current
  // tenant's banks have loaded, redirect away from any URL bank that isn't in
  // that tenant — otherwise the page would render against an inaccessible bank.
  useEffect(() => {
    if (!banksLoaded) return;
    const bankMatch = pathname?.match(/^\/banks\/([^/?]+)/);
    if (!bankMatch) return;
    const urlBank = decodeURIComponent(bankMatch[1]);
    if (!banks.includes(urlBank)) {
      router.push("/dashboard");
    }
  }, [banksLoaded, banks, pathname, router]);

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
