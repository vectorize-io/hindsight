"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useFeatures } from "@/lib/features-context";

/** Knowledge bank pages exist only on a server with knowledge banks on
 *  (HINDSIGHT_API_ENABLE_KNOWLEDGE_BANKS); otherwise a direct link lands on the home page. */
export default function KnowledgeBanksLayout({ children }: { children: React.ReactNode }) {
  const { features, loading } = useFeatures();
  const router = useRouter();
  const enabled = features?.knowledge_banks ?? false;

  useEffect(() => {
    if (!loading && !enabled) router.replace("/");
  }, [loading, enabled, router]);

  return loading || !enabled ? null : <>{children}</>;
}
