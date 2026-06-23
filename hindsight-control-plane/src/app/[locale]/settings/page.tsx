"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { useFeatures } from "@/lib/features-context";

export default function SettingsPage() {
  const router = useRouter();
  const { features, loading } = useFeatures();

  useEffect(() => {
    if (loading) return;
    router.replace(features?.auth_settings_path || "/dashboard");
  }, [features?.auth_settings_path, loading, router]);

  return null;
}
