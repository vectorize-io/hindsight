#!/usr/bin/env node
/**
 * Bank Templates API examples for Hindsight (Node.js)
 * Run: node examples/api/bank-templates.mjs
 */

const HINDSIGHT_URL = process.env.HINDSIGHT_API_URL || 'http://localhost:8888';

// =============================================================================
// Doc Examples
// =============================================================================

// [docs:import-template]
const template = {
  version: '1',
  bank: {
    retain_mission: 'Extract customer issues, resolutions, and sentiment.',
    enable_observations: true,
    observations_mission: 'Track recurring customer pain points.',
  },
  mental_models: [
    {
      id: 'sentiment-overview',
      name: 'Customer Sentiment Overview',
      source_query: 'What is the overall sentiment trend?',
      trigger: { refresh_after_consolidation: true },
    },
  ],
  directives: [
    {
      name: 'Acknowledge frustration',
      content: 'Always acknowledge frustration before offering solutions.',
      priority: 10,
    },
  ],
};

const importResponse = await fetch(
  `${HINDSIGHT_URL}/v1/default/banks/my-bank/import`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(template),
  },
);
const result = await importResponse.json();
console.log('Config applied:', result.config_applied);
console.log('Mental models created:', result.mental_models_created);
console.log('Directives created:', result.directives_created);
// [/docs:import-template]


// [docs:import-dry-run]
const dryRunResponse = await fetch(
  `${HINDSIGHT_URL}/v1/default/banks/my-bank/import?dry_run=true`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(template),
  },
);
const dryRunResult = await dryRunResponse.json();
console.log('Dry run:', dryRunResult.dry_run);
console.log('Would apply config:', dryRunResult.config_applied);
// [/docs:import-dry-run]


// [docs:export-template]
const exportResponse = await fetch(
  `${HINDSIGHT_URL}/v1/default/banks/my-bank/export`,
);
const exported = await exportResponse.json();
console.log(JSON.stringify(exported, null, 2));
// [/docs:export-template]


// [docs:export-reimport]
// Export from source bank
const srcResponse = await fetch(
  `${HINDSIGHT_URL}/v1/default/banks/source-bank/export`,
);
const srcExported = await srcResponse.json();

// Import into a new bank
await fetch(`${HINDSIGHT_URL}/v1/default/banks/new-bank/import`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(srcExported),
});
// [/docs:export-reimport]


// [docs:get-schema]
const schemaResponse = await fetch(
  `${HINDSIGHT_URL}/v1/bank-template-schema`,
);
const schema = await schemaResponse.json();
console.log(JSON.stringify(schema, null, 2));
// [/docs:get-schema]
