import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright config for e2e testing the multi-tenant dashboard.
 *
 * Run against prod:
 *   1. Start the mTLS proxy:  node e2e/mtls-proxy.mjs
 *   2. Copy .env.e2e to .env.local and fill in tenant keys
 *   3. Start dev server:      npm run dev
 *   4. Run tests:             npx playwright test
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1, // sequential — tests share browser state for tenant/bank selection
  reporter: [["html", { open: "never" }], ["list"]],

  use: {
    baseURL: process.env.E2E_BASE_URL || "http://localhost:9999",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    // Increase timeouts for prod latency
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },

  timeout: 60_000,

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  // Don't auto-start the dev server — we expect it (and the mTLS proxy) to already be running
  // webServer: {
  //   command: "npm run dev",
  //   url: "http://localhost:9999",
  //   reuseExistingServer: true,
  // },
});
