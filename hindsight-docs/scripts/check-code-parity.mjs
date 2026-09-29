#!/usr/bin/env node
/**
 * Validates that every "language" Tabs block in MDX docs has all 4 required variants:
 * Python, Node.js, CLI, Go.
 *
 * A Tabs block is considered a "language" block if it contains at least one TabItem
 * with value "python", "node", "cli", or "go".
 *
 * Run: node scripts/check-code-parity.mjs
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsRoot = join(__dirname, '..');
const REQUIRED_TABS = new Set(['python', 'node', 'cli', 'go']);

const IGNORED_PATHS = [
  'node_modules',
  'build',
  '.docusaurus',
  'versioned_docs', // skip versioned docs
];

/**
 * Recursively find all .mdx files under a directory.
 */
function findMdxFiles(dir) {
  const results = [];
  for (const entry of readdirSync(dir)) {
    if (IGNORED_PATHS.includes(entry)) continue;
    const full = join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      results.push(...findMdxFiles(full));
    } else if (entry.endsWith('.mdx') || entry.endsWith('.md')) {
      results.push(full);
    }
  }
  return results;
}

/**
 * Parse a single MDX file and return all violations.
 * A violation is a Tabs block that has at least one language tab but is missing
 * one or more of the 4 required language variants.
 */
function checkFile(filePath) {
  const content = readFileSync(filePath, 'utf8');
  const violations = [];

  // Split content into Tabs blocks.
  // Strategy: find <Tabs> ... </Tabs> sections and scan for TabItem values.
  // We use a simple line-by-line state machine.
  const lines = content.split('\n');
  let inTabs = false;
  let tabsStartLine = -1;
  let currentTabValues = new Set();

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (!inTabs) {
      // Look for opening <Tabs> tag (not <TabItem>)
      if (/^\s*<Tabs[\s>]/.test(line) && !/<\/Tabs/.test(line)) {
        inTabs = true;
        tabsStartLine = i + 1; // 1-indexed
        currentTabValues = new Set();
      }
    } else {
      // Inside a Tabs block — look for </Tabs> or nested TabItem values
      if (/^\s*<\/Tabs\s*>/.test(line)) {
        // End of Tabs block — check if it's a language block
        const hasLanguageTab = [...currentTabValues].some(v => REQUIRED_TABS.has(v));
        if (hasLanguageTab) {
          const missing = [...REQUIRED_TABS].filter(t => !currentTabValues.has(t));
          if (missing.length > 0) {
            violations.push({
              line: tabsStartLine,
              found: [...currentTabValues].filter(v => REQUIRED_TABS.has(v)),
              missing,
            });
          }
        }
        inTabs = false;
        currentTabValues = new Set();
      } else {
        // Look for TabItem value attributes
        // Matches: <TabItem value="python" or <TabItem value='cli'
        const match = line.match(/TabItem[^>]*value=["']([^"']+)["']/);
        if (match) {
          currentTabValues.add(match[1]);
        }
      }
    }
  }

  return violations;
}

// ─── Main ────────────────────────────────────────────────────────────────────

const mdxFiles = findMdxFiles(docsRoot);
let totalViolations = 0;

for (const filePath of mdxFiles) {
  const violations = checkFile(filePath);
  if (violations.length > 0) {
    const rel = relative(docsRoot, filePath);
    for (const v of violations) {
      console.error(
        `[code-parity] ${rel}:${v.line} — Tabs block missing language tabs: ${v.missing.join(', ')} (found: ${v.found.join(', ')})`
      );
    }
    totalViolations += violations.length;
  }
}

if (totalViolations > 0) {
  console.error(`\n[code-parity] ❌ Found ${totalViolations} Tabs block(s) missing required language variants.`);
  console.error('[code-parity] Every Tabs block with language tabs must include: python, node, cli, go');
  process.exit(1);
} else {
  console.log(`[code-parity] ✅ All ${mdxFiles.length} docs files pass 4-tab parity check.`);
}
