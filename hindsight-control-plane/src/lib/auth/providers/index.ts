import { registerControlPlaneAuthProvider } from "@/lib/auth/registry";
import { accessKeyAuthProvider } from "@/lib/auth/providers/access-key";
import { disabledAuthProvider } from "@/lib/auth/providers/disabled";

export function registerBuiltinAuthProviders(): void {
  registerControlPlaneAuthProvider(disabledAuthProvider);
  registerControlPlaneAuthProvider(accessKeyAuthProvider);
}
