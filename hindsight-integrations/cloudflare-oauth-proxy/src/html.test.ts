import { describe, expect, it } from "vitest";
import { escapeHtml, loginPage } from "./html";

describe("escapeHtml", () => {
  it("escapes ampersands first so entities don't get double-encoded", () => {
    expect(escapeHtml("a & b")).toBe("a &amp; b");
  });

  it("escapes all reserved characters", () => {
    expect(escapeHtml(`<script>alert("x&'y")</script>`)).toBe(
      "&lt;script&gt;alert(&quot;x&amp;&#39;y&quot;)&lt;/script&gt;",
    );
  });

  it("leaves safe text untouched", () => {
    expect(escapeHtml("hello world")).toBe("hello world");
  });
});

describe("loginPage", () => {
  it("embeds the stateKey in the hidden form input", () => {
    const html = loginPage("abc-123");
    expect(html).toContain('name="stateKey" value="abc-123"');
  });

  it("escapes a malicious stateKey", () => {
    const html = loginPage('" onclick="alert(1)"');
    expect(html).not.toContain('onclick="alert(1)"');
    expect(html).toContain("&quot; onclick=&quot;alert(1)&quot;");
  });

  it("does not render the error block when no error is passed", () => {
    const html = loginPage("x");
    expect(html).not.toContain('class="error"');
  });

  it("renders an escaped error message when provided", () => {
    const html = loginPage("x", "<bad>");
    expect(html).toContain('<p class="error">&lt;bad&gt;</p>');
  });

  it("always posts to /authorize", () => {
    expect(loginPage("x")).toContain('action="/authorize"');
  });
});
