"use client";

import { OperatorShell } from "@/components/operator-shell";
import { ProviderManager } from "@/components/provider-manager";

export default function ProvidersPage() {
  return (
    <OperatorShell>
      <div className="p-6">
        <ProviderManager />
      </div>
    </OperatorShell>
  );
}
