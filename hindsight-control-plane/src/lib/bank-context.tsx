"use client";

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { client } from "./api";

/** Banks are fetched a page at a time; the selector pages as it is scrolled. */
export const BANKS_PAGE_SIZE = 50;

export interface BankInfo {
  bank_id: string;
  name: string | null;
  mission: string | null;
  created_at: string | null;
  updated_at: string | null;
  fact_count: number;
  last_document_at: string | null;
  last_write_at: string | null;
}

interface BankContextType {
  currentBank: string | null;
  setCurrentBank: (bank: string | null) => void;
  banks: string[];
  bankInfos: BankInfo[];
  /** A first page (or a new search) is in flight. */
  banksLoading: boolean;
  /** A follow-up page is in flight. */
  banksLoadingMore: boolean;
  hasMoreBanks: boolean;
  /** Banks matching the active search, including the ones not fetched yet. */
  totalBanks: number;
  bankSearch: string;
  /** Display name of the selected bank, even when it is not on the loaded page. */
  currentBankName: string | null;
  /** Runs a server-side search and resets to the first page. */
  searchBanks: (query: string) => Promise<void>;
  /** Reloads the first page of the active search. */
  loadBanks: () => Promise<void>;
  loadMoreBanks: () => Promise<void>;
}

const BankContext = createContext<BankContextType | undefined>(undefined);

function toBankInfo(bank: any): BankInfo {
  return {
    bank_id: bank.bank_id,
    name: bank.name ?? null,
    mission: bank.mission ?? null,
    created_at: bank.created_at ?? null,
    updated_at: bank.updated_at ?? null,
    fact_count: bank.fact_count ?? 0,
    last_document_at: bank.last_document_at ?? null,
    last_write_at: bank.last_write_at ?? null,
  };
}

export function BankProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [currentBank, setCurrentBank] = useState<string | null>(null);
  const [bankInfos, setBankInfos] = useState<BankInfo[]>([]);
  const [banksLoading, setBanksLoading] = useState(true);
  const [banksLoadingMore, setBanksLoadingMore] = useState(false);
  const [totalBanks, setTotalBanks] = useState(0);
  const [bankSearch, setBankSearch] = useState("");

  // Every first-page fetch bumps this; a response whose stamp is stale is dropped, so
  // typing quickly can't leave an earlier query's results on screen, and an in-flight
  // "load more" from the previous query can't append onto the new one.
  const requestSeq = useRef(0);
  const searchRef = useRef("");
  const loadedRef = useRef<BankInfo[]>([]);
  const totalRef = useRef(0);
  const loadingMoreRef = useRef(false);

  useEffect(() => {
    loadedRef.current = bankInfos;
  }, [bankInfos]);
  useEffect(() => {
    totalRef.current = totalBanks;
  }, [totalBanks]);

  const loadFirstPage = useCallback(async (query: string) => {
    const seq = ++requestSeq.current;
    searchRef.current = query;
    setBankSearch(query);
    setBanksLoading(true);
    try {
      const response = await client.listBanks({
        q: query || undefined,
        limit: BANKS_PAGE_SIZE,
        offset: 0,
      });
      if (seq !== requestSeq.current) return;
      setBankInfos((response.banks || []).map(toBankInfo));
      setTotalBanks(response.total ?? 0);
    } catch (error) {
      if (seq !== requestSeq.current) return;
      console.error("Error loading banks:", error);
    } finally {
      if (seq === requestSeq.current) setBanksLoading(false);
    }
  }, []);

  const searchBanks = useCallback((query: string) => loadFirstPage(query), [loadFirstPage]);
  const loadBanks = useCallback(() => loadFirstPage(searchRef.current), [loadFirstPage]);

  const loadMoreBanks = useCallback(async () => {
    if (loadingMoreRef.current) return;
    const offset = loadedRef.current.length;
    if (offset >= totalRef.current) return;
    const seq = requestSeq.current;
    loadingMoreRef.current = true;
    setBanksLoadingMore(true);
    try {
      const response = await client.listBanks({
        q: searchRef.current || undefined,
        limit: BANKS_PAGE_SIZE,
        offset,
      });
      if (seq !== requestSeq.current) return;
      setBankInfos((prev) => {
        // A bank created (or bumped to the front) between pages would otherwise come
        // back on two offsets, so append only ids we don't already hold.
        const seen = new Set(prev.map((b) => b.bank_id));
        const next = (response.banks || [])
          .map(toBankInfo)
          .filter((bank) => !seen.has(bank.bank_id));
        return next.length > 0 ? [...prev, ...next] : prev;
      });
      setTotalBanks(response.total ?? 0);
    } catch (error) {
      console.error("Error loading more banks:", error);
    } finally {
      loadingMoreRef.current = false;
      setBanksLoadingMore(false);
    }
  }, []);

  // Derive bank IDs for backwards compatibility
  const banks = bankInfos.map((b) => b.bank_id);

  // The selected bank is only in `bankInfos` if it happens to be on a loaded page, so
  // its name is fetched directly — otherwise the header would fall back to the raw id
  // for any bank sitting past the first page.
  const [fetchedBankName, setFetchedBankName] = useState<string | null>(null);
  const loadedBankName = bankInfos.find((b) => b.bank_id === currentBank)?.name ?? null;
  useEffect(() => {
    if (!currentBank || loadedBankName) return;
    let cancelled = false;
    client
      .getBankProfile(currentBank)
      .then((profile) => {
        if (!cancelled) setFetchedBankName(profile.name || null);
      })
      .catch(() => {
        if (!cancelled) setFetchedBankName(null);
      });
    return () => {
      cancelled = true;
    };
  }, [currentBank, loadedBankName]);

  // Initialize bank from URL on mount
  useEffect(() => {
    const bankMatch = pathname?.match(/^\/banks\/([^/?]+)/);
    if (bankMatch) {
      setCurrentBank(decodeURIComponent(bankMatch[1]));
    }
  }, [pathname]);

  useEffect(() => {
    loadBanks();
  }, [loadBanks]);

  return (
    <BankContext.Provider
      value={{
        currentBank,
        setCurrentBank,
        banks,
        bankInfos,
        banksLoading,
        banksLoadingMore,
        hasMoreBanks: bankInfos.length < totalBanks,
        totalBanks,
        bankSearch,
        currentBankName: loadedBankName ?? fetchedBankName,
        searchBanks,
        loadBanks,
        loadMoreBanks,
      }}
    >
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
