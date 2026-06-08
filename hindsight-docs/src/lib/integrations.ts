import integrationsData from '@site/src/data/integrations.json';

export interface IntegrationEntry {
  id: string;
  name: string;
  description: string;
  type: string;
  by: string;
  category: string;
  link: string;
  icon: string;
}

const byName = (a: IntegrationEntry, b: IntegrationEntry): number =>
  a.name.toLowerCase().localeCompare(b.name.toLowerCase());

// src/data/integrations.json is the single source of truth. Display order is
// alphabetical by name, shared by the Integrations Hub gallery and the docs
// sidebars (the JSON array order is no longer significant for display).
export const integrationsSorted: IntegrationEntry[] = [
  ...(integrationsData.integrations as IntegrationEntry[]),
].sort(byName);

// Only entries with an on-site doc page (external/http entries are gallery-only).
export const internalIntegrationsSorted: IntegrationEntry[] = integrationsSorted.filter((entry) =>
  entry.link.startsWith('/sdks/integrations/'),
);
