"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { client } from "./api";

interface TenantContextType {
  /** Currently selected tenant name, or null if not yet loaded */
  currentTenant: string | null;
  /** Switch to a different tenant */
  setCurrentTenant: (tenant: string) => void;
  /** All available tenant names */
  tenants: string[];
  /** Whether multi-tenant mode is active */
  isMultiTenant: boolean;
  /** Whether tenants are still loading */
  loading: boolean;
}

const TenantContext = createContext<TenantContextType | undefined>(undefined);

const STORAGE_KEY = "hindsight-cp-tenant";

export function TenantProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [tenants, setTenants] = useState<string[]>([]);
  const [isMultiTenant, setIsMultiTenant] = useState(false);
  const [currentTenant, setCurrentTenantState] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const setCurrentTenant = useCallback(
    (tenant: string) => {
      setCurrentTenantState(tenant);
      client.setTenant(tenant);
      try {
        localStorage.setItem(STORAGE_KEY, tenant);
      } catch {
        // localStorage may be unavailable
      }
      // The previous tenant's bank can't exist in the new schema, and per-bank
      // views cache fetched data in component state. Returning to /dashboard
      // unmounts those components so they don't render stale results.
      router.push("/dashboard");
    },
    [router]
  );

  useEffect(() => {
    async function loadTenants() {
      try {
        const data = await client.listTenants();
        setTenants(data.tenants);
        setIsMultiTenant(data.multi_tenant);

        // Restore saved tenant or use first available
        let saved: string | null = null;
        try {
          saved = localStorage.getItem(STORAGE_KEY);
        } catch {
          // localStorage may be unavailable
        }

        const initial =
          saved && data.tenants.includes(saved)
            ? saved
            : data.tenants[0] ?? null;

        if (initial) {
          setCurrentTenantState(initial);
          client.setTenant(initial);
        }
      } catch (error) {
        console.error("Failed to load tenants:", error);
      } finally {
        setLoading(false);
      }
    }

    loadTenants();
  }, []);

  return (
    <TenantContext.Provider
      value={{ currentTenant, setCurrentTenant, tenants, isMultiTenant, loading }}
    >
      {children}
    </TenantContext.Provider>
  );
}

export function useTenant() {
  const context = useContext(TenantContext);
  if (context === undefined) {
    throw new Error("useTenant must be used within a TenantProvider");
  }
  return context;
}
