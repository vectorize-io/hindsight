# Agent Guide: Hindsight Control Plane

This document provides essential context for AI agents working in the Hindsight Control Plane repository. It focuses on non-obvious architectural patterns, commands, and conventions to minimize trial-and-error.

## 🚀 Essential Commands

### Development
- `npm run dev`: Starts the Next.js development server on port `8999` (or the specified `PORT`). Uses `--turbopack`.
- `npm run build`: Builds the application for production and generates a standalone deployment package in the `standalone/` directory.
- `npm run start`: Starts the Next.js production server.

### Testing & Quality
- `npm run test`: Runs the Vitest test suite.
- `npm run test:watch`: Runs Vitest in watch mode.
- `npm run lint`: Runs Next.js linting.
- `npm run i18n:check`: Runs a custom script to find untranslated messages.

## 🏗️ Architecture & Data Flow

The Control Plane is a Next.js application that serves as a management interface for the Hindsight semantic memory system.

### High-Level Pattern
- **Frontend**: Next.js (App Router) using React 19, Tailwind CSS, and Radix UI components.
- **API Architecture**: The application uses Next.js Route Handlers (`app/api/...`) as a proxy/gateway.
- **Data Source**: Most API routes do not interact with a database directly but instead proxy requests to the **Hindsight Dataplane API** via the `@vectorize-io/hindsight-client`.
- **Authentication**: Managed via session tokens, proxied through the control plane to the dataplane.

### Key Components
- **Bank Context (`src/lib/bank-context.tsx`)**: A critical React Context that manages the currently selected `bank_id`. It derives the active bank from the URL path (e.g., `/banks/[bankId]/...`). Most features are bank-scoped.
- **Client-Dataplane Interaction**:
  - `src/lib/hindsight-client.ts`: Provides the `hindsightClient` and `lowLevelClient` for communicating with the dataplane.
  - **Crucial Gotcha**: Bank IDs can contain special characters (e.g., `agent::channel::user`). The `dataplaneBankUrl` utility in `hindsight-client.ts` uses `encodeURIComponent` to ensure these are handled correctly in URL paths.
- **Internationalization (i18n)**:
  - Uses `next-intl` with a locale-based routing structure (`app/[locale]/...`).
  - Error handling is centralized in `src/lib/i18n/api-errors.ts`, which localizes error payloads based on the user's locale (from cookies or `Accept-Language` headers).

## 🎨 Conventions & Patterns

### Code Organization
- `src/app/api/...`: Backend logic and proxy routes.
- `src/components/...`: UI components.
  - `src/components/ui/...`: Low-level, reusable primitive components (Radix/Shadcn style).
  - `src/components/[view-name].tsx`: Higher-level, feature-specific views.
    - **Diagnostic/Specialized Views**: This codebase contains specialized views for deep inspection, such as `search-debug-view.tsx`, `llm-requests-view.tsx`, `observation-history-view.tsx`, and `think-view.tsx`.
- `src/lib/...`: Shared utilities, context providers, and API clients.
- `src/messages/...`: JSON files containing translation strings for supported locales.

### API Error Handling Pattern
When creating or modifying API routes, follow this pattern for errors:
1. Catch the error.
2. Use `localizeApiErrorPayload(request, { error, errorKey })`.
3. Return the localized payload with the appropriate HTTP status code.

### Naming & Style
- **TypeScript**: Strict typing is expected.
- **Components**: PascalCase for component files and functions.
- **Files**: kebab-case for most non-component files.
- **Contexts**: Use custom hooks (e.g., `useBank()`) to consume context rather than accessing the context object directly.

## ⚠️ Important Gotchas

- **Asynchronous Operations**: Many write operations (like creating a mental model) return an `operation_id` and a `202 Accepted` status. The actual work happens in the background on the dataplane. Agents should look for "polling" or "status check" patterns when handling these.
- **Bank Scoping**: Always check if a requested resource or action is scoped to a `bankId`. The `BankProvider` is the source of truth for the active bank.
- **Whitelabeling**: The application supports whitelabeling via `src/lib/whitelabel-config.ts` and a `BrandStyleInjector`. UI changes should respect these configurations.
- **URL Encoding**: Never manually construct dataplane URLs with `bankId` without using the `dataplaneBankUrl` utility, due to potential special characters in IDs.
- **Next.js Metadata**: The project contains several warnings regarding viewport and themeColor being in the `metadata` export. Future changes should move these to the `viewport` export to comply with Next.js 15+ standards.
