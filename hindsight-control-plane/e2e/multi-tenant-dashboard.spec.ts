import { test, expect, type Page } from "@playwright/test";

/**
 * E2E tests for the multi-tenant dashboard feature.
 *
 * Prerequisites:
 *   1. mTLS proxy running:   node e2e/mtls-proxy.mjs
 *   2. .env.local configured with HINDSIGHT_CP_TENANT_KEY_MAP pointing at prod
 *   3. Dev server running:   npx next dev --turbopack -p 9999
 *
 * These tests are READ-ONLY — they never create, modify, or delete data on prod.
 */

// Helper: wait for the app to finish loading (tenant + bank contexts)
async function waitForAppReady(page: Page) {
  await page.waitForSelector("img[alt='Hindsight']", { timeout: 15_000 });
  await page.waitForTimeout(1000);
}

// Helper: get the tenant selector trigger (Radix Select)
function tenantSelector(page: Page) {
  return page.locator("button[role='combobox']").filter({ hasText: /jones|paula|Select tenant/i });
}

// Helper: get the bank selector trigger (cmdk Popover)
function bankSelector(page: Page) {
  return page.locator("button[role='combobox']").filter({ hasText: /Select a memory bank|openclaw/i });
}

// Helper: select jones tenant and wait for banks to load
async function selectJonesTenant(page: Page) {
  const trigger = tenantSelector(page);
  await trigger.click();
  await page.getByRole("option", { name: "household_jones" }).click();
  await page.waitForTimeout(3000); // wait for banks to load from prod
}

// Helper: open bank selector and pick the first bank
async function selectFirstBank(page: Page) {
  const bankTrigger = bankSelector(page);
  await bankTrigger.click();
  await page.locator("[cmdk-item]").first().waitFor({ timeout: 10_000 });
  await page.locator("[cmdk-item]").first().click();
  await expect(page).toHaveURL(/\/banks\/.+/);
  await page.waitForTimeout(1000); // let the bank page render
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Tenant Discovery
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Tenant Discovery", () => {
  test("tenants API returns jones and paula in multi-tenant mode", async ({ request }) => {
    const res = await request.get("/api/tenants");
    expect(res.ok()).toBeTruthy();

    const data = await res.json();
    expect(data.multi_tenant).toBe(true);
    expect(data.tenants).toContain("household_jones");
    expect(data.tenants).toContain("household_paula");
    expect(data.tenants.length).toBe(2);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Dashboard Landing
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Dashboard Landing", () => {
  test("root redirects to /dashboard", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/dashboard/);
  });

  test("dashboard shows welcome message", async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);
    await expect(page.getByText("Welcome to Hindsight")).toBeVisible();
  });

  test("tenant selector is visible in multi-tenant mode", async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);

    const selector = tenantSelector(page);
    await expect(selector).toBeVisible();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Tenant Switching
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Tenant Switching", () => {
  test("can switch between tenants", async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);

    const trigger = tenantSelector(page);
    await expect(trigger).toBeVisible();

    const initialText = await trigger.textContent();

    // Click to open the dropdown
    await trigger.click();

    // Both tenant options should be visible
    await expect(page.getByRole("option", { name: "household_jones" })).toBeVisible();
    await expect(page.getByRole("option", { name: "household_paula" })).toBeVisible();

    // Switch to the other tenant
    const targetTenant = initialText?.includes("household_jones") ? "household_paula" : "household_jones";
    await page.getByRole("option", { name: targetTenant }).click();

    await expect(trigger).toHaveText(new RegExp(targetTenant, "i"));
  });

  test("tenant selection persists across navigation", async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);

    const trigger = tenantSelector(page);

    // Switch to paula
    await trigger.click();
    await page.getByRole("option", { name: "household_paula" }).click();
    await expect(trigger).toHaveText(/paula/i);

    // Navigate away and back
    await page.goto("/dashboard");
    await waitForAppReady(page);

    // Tenant should still be paula (restored from localStorage)
    const triggerAfter = tenantSelector(page);
    await expect(triggerAfter).toHaveText(/paula/i);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Bank Loading per Tenant
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Bank Loading per Tenant", () => {
  test("banks API returns banks for jones tenant", async ({ request }) => {
    const res = await request.get("/api/banks?tenant=household_jones");
    expect(res.ok()).toBeTruthy();

    const data = await res.json();
    expect(data.banks).toBeDefined();
    expect(Array.isArray(data.banks)).toBe(true);
    expect(data.banks.length).toBeGreaterThan(0);
  });

  test("both tenant APIs return valid responses", async ({ request }) => {
    const jonesRes = await request.get("/api/banks?tenant=household_jones");
    const paulaRes = await request.get("/api/banks?tenant=household_paula");

    expect(jonesRes.ok()).toBeTruthy();
    expect(paulaRes.ok()).toBeTruthy();

    const jonesData = await jonesRes.json();
    const paulaData = await paulaRes.json();

    // Jones should have banks (prod has data), paula may not
    expect(jonesData.banks.length).toBeGreaterThan(0);
    expect(Array.isArray(paulaData.banks)).toBe(true);
  });

  test("switching tenant refreshes bank list in UI", async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);

    // Select jones tenant (which has banks)
    await selectJonesTenant(page);

    // Open bank selector and wait for items to appear
    const bankTrigger = bankSelector(page);
    await bankTrigger.click();
    await page.locator("[cmdk-item]").first().waitFor({ timeout: 10_000 });

    const jonesItems = await page.locator("[cmdk-item]").allTextContents();
    expect(jonesItems.length).toBeGreaterThan(0);

    // Close popover
    await page.keyboard.press("Escape");
    await page.waitForTimeout(500);

    // Switch to paula — verify the bank list changes (may be empty)
    const trigger = tenantSelector(page);
    await trigger.click();
    await page.getByRole("option", { name: "household_paula" }).click();
    await page.waitForTimeout(3000);

    // Re-open bank selector — paula may have no banks, which is fine
    // The key assertion is that switching didn't error out
    await bankSelector(page).click();
    await page.waitForTimeout(1000);

    // The empty state shows "No memory banks yet." or bank items
    const hasBanks = await page.locator("[cmdk-item]").count();
    const hasEmpty = await page.getByText("No memory banks yet").count();
    expect(hasBanks + hasEmpty).toBeGreaterThan(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. Bank Navigation
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Bank Navigation", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);
    await selectJonesTenant(page);
  });

  test("selecting a bank navigates to bank detail page", async ({ page }) => {
    await selectFirstBank(page);
    // Already asserted in helper — URL matches /banks/...
  });

  test("bank detail page has sidebar with navigation links", async ({ page }) => {
    await selectFirstBank(page);

    // Sidebar is collapsed by default — links have title attributes
    await expect(page.locator("a[title='Recall']")).toBeVisible();
    await expect(page.locator("a[title='Reflect']")).toBeVisible();
    await expect(page.locator("a[title='Memories']")).toBeVisible();
    await expect(page.locator("a[title='Documents']")).toBeVisible();
    await expect(page.locator("a[title='Entities']")).toBeVisible();
    await expect(page.locator("a[title='Bank Configuration']")).toBeVisible();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. Bank Detail Views (read-only)
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Bank Detail Views", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);
    await selectJonesTenant(page);
    await selectFirstBank(page);
  });

  test("can navigate to Memories view with sub-tabs", async ({ page }) => {
    await page.locator("a[title='Memories']").click();
    await expect(page).toHaveURL(/view=data/);

    // Sub-tabs are plain buttons (not Radix Tabs)
    await expect(page.getByRole("button", { name: /World Facts/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Experience/i })).toBeVisible();
  });

  test("can navigate to Recall view", async ({ page }) => {
    await page.locator("a[title='Recall']").click();
    await expect(page).toHaveURL(/view=recall/);
  });

  test("can navigate to Reflect view", async ({ page }) => {
    await page.locator("a[title='Reflect']").click();
    await expect(page).toHaveURL(/view=reflect/);
  });

  test("can navigate to Documents view", async ({ page }) => {
    await page.locator("a[title='Documents']").click();
    await expect(page).toHaveURL(/view=documents/);
  });

  test("can navigate to Entities view", async ({ page }) => {
    await page.locator("a[title='Entities']").click();
    await expect(page).toHaveURL(/view=entities/);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Theme Toggle
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Theme Toggle", () => {
  test("can toggle between light and dark theme", async ({ page }) => {
    await page.goto("/dashboard");
    await waitForAppReady(page);

    const themeButton = page.locator("button").filter({ has: page.locator("svg.lucide-sun, svg.lucide-moon") });
    await expect(themeButton).toBeVisible();

    const htmlEl = page.locator("html");
    const initialClass = await htmlEl.getAttribute("class");

    await themeButton.click();
    await page.waitForTimeout(300);

    const newClass = await htmlEl.getAttribute("class");
    expect(newClass).not.toBe(initialClass);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. Cross-tenant Isolation
// ─────────────────────────────────────────────────────────────────────────────

test.describe("Cross-tenant Isolation", () => {
  test("tenant selector correctly scopes API calls", async ({ page }) => {
    // Set up request listener BEFORE navigating
    const apiCalls: string[] = [];
    page.on("request", (req) => {
      const url = req.url();
      if (url.includes("/api/") && !url.includes("_next")) {
        apiCalls.push(url);
      }
    });

    await page.goto("/dashboard");
    await waitForAppReady(page);

    // Switch to jones — this should trigger bank fetch with tenant param
    const trigger = tenantSelector(page);
    await trigger.click();
    await page.getByRole("option", { name: "household_jones" }).click();
    await page.waitForTimeout(3000);

    // Verify at least one API call included the tenant parameter
    const jonesCalls = apiCalls.filter((url) => url.includes("tenant=household_jones"));
    expect(jonesCalls.length).toBeGreaterThan(0);

    // Clear and switch to paula
    apiCalls.length = 0;
    await trigger.click();
    await page.getByRole("option", { name: "household_paula" }).click();
    await page.waitForTimeout(3000);

    const paulaCalls = apiCalls.filter((url) => url.includes("tenant=household_paula"));
    expect(paulaCalls.length).toBeGreaterThan(0);
  });
});
