#!/usr/bin/env node

/**
 * Generates llms-full.txt by concatenating all documentation markdown files.
 * This file is used by LLMs to understand the full documentation.
 *
 * Usage: node scripts/generate-llms-full.js
 *
 * Output: static/llms-full.txt (served at /llms-full.txt)
 */

const fs = require('fs');
const path = require('path');

const DOCS_DIR = path.join(__dirname, '..', 'docs');
const OUTPUT_FILE = path.join(__dirname, '..', 'static', 'llms-full.txt');

// Order matters - more important docs first
const DOC_ORDER = [
  'developer/index.md',
  'developer/api/quickstart.md',
  'developer/api/main-methods.md',
  'developer/retain.md',
  'developer/retrieval.md',
  'developer/reflect.md',
  'developer/api/retain.md',
  'developer/api/recall.md',
  'developer/api/reflect.md',
  'developer/api/memory-banks.md',
  'developer/api/entities.md',
  'developer/api/documents.md',
  'developer/api/operations.md',
  'developer/installation.md',
  'developer/configuration.md',
  'developer/models.md',
  'developer/rag-vs-hindsight.md',
  'sdks/python.md',
  'sdks/nodejs.md',
  'sdks/cli.md',
  'sdks/mcp.md',
  'cookbook/index.md',
  'cookbook/per-user-memory.md',
  'cookbook/support-agent-with-shared-knowledge.md',
];

function getAllMarkdownFiles(dir, baseDir = dir) {
  const files = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...getAllMarkdownFiles(fullPath, baseDir));
    } else if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
      const relativePath = path.relative(baseDir, fullPath);
      files.push(relativePath);
    }
  }

  return files;
}

function stripFrontmatter(content) {
  // Remove YAML frontmatter (between --- markers)
  const frontmatterRegex = /^---\n[\s\S]*?\n---\n/;
  return content.replace(frontmatterRegex, '');
}

function cleanMarkdown(content) {
  let cleaned = stripFrontmatter(content);

  // Remove import statements
  cleaned = cleaned.replace(/^import\s+.*$/gm, '');

  // Remove empty lines at start
  cleaned = cleaned.replace(/^\n+/, '');

  return cleaned;
}

function generateLlmsFullTxt() {
  console.log('Generating llms-full.txt...');

  // Get all markdown files
  const allFiles = getAllMarkdownFiles(DOCS_DIR);

  // Create ordered list: prioritized files first, then remaining files
  const orderedFiles = [];
  const remainingFiles = new Set(allFiles);

  // Add prioritized files in order
  for (const file of DOC_ORDER) {
    if (remainingFiles.has(file)) {
      orderedFiles.push(file);
      remainingFiles.delete(file);
    }
  }

  // Add remaining files (sorted alphabetically)
  const sortedRemaining = Array.from(remainingFiles).sort();
  orderedFiles.push(...sortedRemaining);

  // Build the output
  const sections = [];

  // Header
  sections.push(`# Hindsight Documentation

> Agent Memory that Works Like Human Memory

This file contains the complete Hindsight documentation for LLM consumption.
Generated: ${new Date().toISOString()}

---
`);

  // Process each file
  for (const file of orderedFiles) {
    const filePath = path.join(DOCS_DIR, file);

    if (!fs.existsSync(filePath)) {
      console.warn(`  Warning: ${file} not found, skipping`);
      continue;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const cleanedContent = cleanMarkdown(content);

    if (cleanedContent.trim()) {
      // Add file path as context
      sections.push(`\n## File: ${file}\n`);
      sections.push(cleanedContent);
      sections.push('\n---\n');
      console.log(`  Added: ${file}`);
    }
  }

  // Write output
  const output = sections.join('\n');
  fs.writeFileSync(OUTPUT_FILE, output);

  const stats = fs.statSync(OUTPUT_FILE);
  const sizeKb = (stats.size / 1024).toFixed(1);

  console.log(`\nGenerated: ${OUTPUT_FILE}`);
  console.log(`Size: ${sizeKb} KB`);
  console.log(`Files included: ${orderedFiles.length}`);
}

generateLlmsFullTxt();
