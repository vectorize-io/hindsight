import type {
  ControlPlaneAuthProvider,
  ControlPlaneAuthProviderAdapter,
} from "@/lib/auth/provider";

const providers = new Map<ControlPlaneAuthProvider, ControlPlaneAuthProviderAdapter>();

export function registerControlPlaneAuthProvider(adapter: ControlPlaneAuthProviderAdapter): void {
  providers.set(adapter.id, adapter);
}

export function getRegisteredControlPlaneAuthProvider(
  id: ControlPlaneAuthProvider
): ControlPlaneAuthProviderAdapter | undefined {
  return providers.get(id);
}

export function hasRegisteredControlPlaneAuthProvider(id: string): id is ControlPlaneAuthProvider {
  return providers.has(id);
}
