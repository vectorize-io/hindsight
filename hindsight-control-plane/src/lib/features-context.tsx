"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { client } from "./api";

interface Features {
  observations: boolean;
  mcp: boolean;
  worker: boolean;
}

interface FeaturesContextType {
  features: Features | null;
  loading: boolean;
  error: string | null;
}

const defaultFeatures: Features = {
  observations: false,
  mcp: false,
  worker: false,
};

const FeaturesContext = createContext<FeaturesContextType | undefined>(undefined);

export function FeaturesProvider({ children }: { children: React.ReactNode }) {
  const [features, setFeatures] = useState<Features | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadFeatures = async () => {
      try {
        const response = await client.getVersion();
        setFeatures(response.features);
        setError(null);
      } catch (err) {
        console.error("Error loading features:", err);
        setError("Failed to load feature flags");
        // Use defaults on error
        setFeatures(defaultFeatures);
      } finally {
        setLoading(false);
      }
    };

    loadFeatures();
  }, []);

  return (
    <FeaturesContext.Provider value={{ features, loading, error }}>
      {children}
    </FeaturesContext.Provider>
  );
}

export function useFeatures() {
  const context = useContext(FeaturesContext);
  if (context === undefined) {
    throw new Error("useFeatures must be used within a FeaturesProvider");
  }
  return context;
}
