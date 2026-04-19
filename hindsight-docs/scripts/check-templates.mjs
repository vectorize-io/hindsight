#!/usr/bin/env node
/**
 * Validates that every manifest referenced from templates.json conforms
 * to the bank template JSON Schema (static/bank-template-schema.json).
 * Each catalog entry's manifest_file is loaded off disk.
 *
 * Run: node scripts/check-templates.mjs
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv from 'ajv';

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsRoot = join(__dirname, '..');
const dataRoot = join(docsRoot, 'src/data');

const catalog = JSON.parse(
  readFileSync(join(dataRoot, 'templates.json'), 'utf-8'),
);
const schema = JSON.parse(
  readFileSync(join(docsRoot, 'static/bank-template-schema.json'), 'utf-8'),
);

const ajv = new Ajv({ allErrors: true, strict: false });
const validate = ajv.compile(schema);

let failed = 0;

for (const entry of catalog.templates) {
  const manifest = JSON.parse(
    readFileSync(join(dataRoot, entry.manifest_file), 'utf-8'),
  );
  const valid = validate(manifest);
  if (!valid) {
    failed++;
    console.error(`\x1b[31m✗\x1b[0m Template "${entry.id}" (${entry.manifest_file}) has invalid manifest:`);
    // Filter out noisy anyOf wrapper errors, keep only leaf errors with paths
    const meaningful = validate.errors.filter(
      (e) => e.keyword !== 'anyOf' && e.keyword !== 'if' && e.keyword !== 'then',
    );
    const shown = meaningful.length > 0 ? meaningful : validate.errors;
    for (const err of shown) {
      const path = err.instancePath || err.schemaPath || '(root)';
      console.error(`    ${path}: ${err.message} ${err.params ? JSON.stringify(err.params) : ''}`);
    }
  } else {
    console.log(`\x1b[32m✓\x1b[0m Template "${entry.id}" — valid`);
  }
}

if (failed > 0) {
  console.error(`\n\x1b[31m${failed} template(s) failed schema validation.\x1b[0m`);
  process.exit(1);
} else {
  console.log(`\n\x1b[32mAll ${catalog.templates.length} templates are valid.\x1b[0m`);
}
