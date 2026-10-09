---
title: "What's New in Hindsight Cloud: Document File Upload"
description: "Hindsight Cloud now takes PDFs, Word, PowerPoint, Excel, images and text files, and turns them into memories your agents can recall."
authors: [benfrank241]
date: 2026-03-09T12:00
tags: [hindsight-cloud, release, memory]
hide_table_of_contents: true
---

[Hindsight Cloud](https://ui.hindsight.vectorize.io) now lets you upload files directly to any memory bank. PDFs, Word documents, PowerPoint presentations, Excel spreadsheets, images, and plain text files are all supported.

<!-- truncate -->

## Two Extraction Methods

When you upload a file, you choose how Hindsight processes it:

**Standard extraction** uses [Markitdown](https://github.com/microsoft/markitdown) to pull text from your document. It is free and works well for text-heavy files like PDFs, Word docs, and plain text.

**Enhanced extraction (Iris)** uses AI-powered processing to understand the structure and meaning of your document. This is useful for complex layouts, scanned documents, and images. Enhanced extraction incurs per-token charges.

Both methods extract the content and store it as structured memories in your bank, ready for recall and reflect queries.

## How It Works

1. Open any memory bank in [Hindsight Cloud](https://ui.hindsight.vectorize.io)
2. Click the upload button and select your files
3. Choose Standard or Enhanced (Iris) extraction
4. The Document Operations panel tracks progress with status indicators

Once processing completes, the extracted content becomes part of your memory bank. Your agents can recall and reflect on the document contents just like any other memory.

## Supported File Types

- **PDFs** -- reports, whitepapers, research papers
- **Word documents** (.docx) -- meeting notes, specs, proposals
- **PowerPoint presentations** (.pptx) -- slide decks, training materials
- **Excel spreadsheets** (.xlsx) -- data tables, financial reports
- **Images** (.png, .jpg) -- screenshots, diagrams (Enhanced extraction recommended)
- **Plain text** (.txt, .md) -- logs, notes, documentation

## What Else Is New

This update also follows two other recent additions:

**Bank-scoped API keys** (March 3) let you restrict API keys to specific memory banks. This is useful for multi-tenant setups where each customer or agent should only access its own memories. Unauthorized access attempts return a 403 error.

**MCP support** (February 13) added [Model Context Protocol integration](/blog/2026/03/04/mcp-agent-memory) for AI clients like Claude, Cursor, and VS Code. You can connect in single-bank mode for a dedicated agent memory or multi-bank mode for cross-bank operations, with 30 tools available out of the box.

## Get Started

File upload is available now in [Hindsight Cloud](https://ui.hindsight.vectorize.io). Upload your first document and try a recall query against it to see the results.
