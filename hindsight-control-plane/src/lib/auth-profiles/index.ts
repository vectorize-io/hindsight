import { registerControlPlaneAuthProvider } from "@/lib/auth/registry";
import { supabaseOrgAuthProvider } from "@/lib/auth-profiles/supabase-org/provider";

export function registerBundledAuthProfiles(): void {
  registerControlPlaneAuthProvider(supabaseOrgAuthProvider);
}
