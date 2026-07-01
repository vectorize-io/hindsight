"use client";

import { useEffect, useState } from "react";

// Shared workspace selection (localStorage). Every data page reads the active
// workspace id from here so the operator picks it once.
const KEY = "cm_workspace_id";

export function useWorkspace(): [string, (id: string) => void] {
  const [wid, setWid] = useState<string>("");

  useEffect(() => {
    setWid(localStorage.getItem(KEY) ?? "");
  }, []);

  const set = (id: string) => {
    localStorage.setItem(KEY, id);
    setWid(id);
  };

  return [wid, set];
}
