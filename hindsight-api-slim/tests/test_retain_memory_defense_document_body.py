"""End-to-end retain → verify Memory Defense scrubs secrets/PII from BOTH
memory_units AND the document body (documents.original_text).

This is the regression test for the "ghp_AAA... persists in raw documents" leak:
per-chunk screen() mutates `_content.content`, but the document body was being
built either (a) from `contents_dicts[i]['content']` BEFORE the mirror-into-dict
fix landed, or (b) from `document_body_override` (which carries the FULL
original unredacted body for oversized inputs).

Both paths are covered here.
"""

import pytest

# Mix of secret patterns covered by the redactor (keys, tokens, PII, DB URLs).
SECRETS = {
    "ssn": "123-45-6789",
    "github_pat": "ghp_" + "A" * 36,
    "github_app": "ghs_" + "B" * 36,
    "anthropic": "sk-ant-" + "C" * 40,
    "xai": "xai-" + "D" * 40,
    "groq": "gsk_" + "E" * 30,
    "huggingface": "hf_" + "F" * 35,
    "stripe_live": "sk_live_" + "G" * 30,
    "twilio_sid": "AC" + "0" * 32,
    "sendgrid": "SG." + "H" * 22 + "." + "I" * 43,
    "aws_access": "AKIA" + "J" * 16,
    "postgres_url": "postgres://user:p4ssw0rd@db.example.com:5432/app",
}


@pytest.mark.asyncio
async def test_regex_scrubs_secrets_from_document_body(api_client) -> None:
    bank = "md-doc-body-1"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await api_client.patch(
        f"/v1/default/banks/{bank}/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )

    doc_id = "leak-test-doc-1"
    body = "Audit log:\n" + "\n".join(f"- {label} = {value}" for label, value in SECRETS.items())

    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={
            "items": [
                {
                    "content": body,
                    "document_id": doc_id,
                }
            ],
        },
    )
    assert r.status_code == 200, r.text

    # 1) Memory units must not contain ANY secret value verbatim.
    r2 = await api_client.get(f"/v1/default/banks/{bank}/memories/list", params={"limit": 200})
    units = r2.json()["items"]
    for label, value in SECRETS.items():
        for unit in units:
            assert value not in unit["text"], f"memory_unit leaked {label}={value!r}: unit.text={unit['text']!r}"

    # 2) Document body must not contain ANY secret value verbatim.
    r3 = await api_client.get(f"/v1/default/banks/{bank}/documents/{doc_id}")
    assert r3.status_code == 200, r3.text
    original_text = r3.json()["original_text"]
    for label, value in SECRETS.items():
        assert value not in original_text, (
            f"document.original_text leaked {label}={value!r}\nfull body:\n{original_text}"
        )


@pytest.mark.asyncio
async def test_regex_scrubs_ssn_from_short_message(api_client) -> None:
    """The exact phrasing the user pasted that triggered the rage report:
    a single short message containing a US SSN.
    """
    bank = "md-doc-body-ssn"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await api_client.patch(
        f"/v1/default/banks/{bank}/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )

    doc_id = "ssn-rage-1"
    ssn = "123-45-6789"
    body = f"The user pasted their ssn us for debugging: {ssn} — please scrub and rotate."

    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={
            "items": [{"content": body, "document_id": doc_id}],
        },
    )
    assert r.status_code == 200, r.text

    r2 = await api_client.get(f"/v1/default/banks/{bank}/memories/list", params={"limit": 50})
    units = r2.json()["items"]
    for unit in units:
        assert ssn not in unit["text"], f"memory_unit leaked SSN: {unit['text']!r}"

    r3 = await api_client.get(f"/v1/default/banks/{bank}/documents/{doc_id}")
    assert r3.status_code == 200, r3.text
    original_text = r3.json()["original_text"]
    assert ssn not in original_text, f"document.original_text leaked SSN: {original_text!r}"
    assert "[REDACTED:ssn_us]" in original_text, (
        f"document.original_text should contain redaction marker, got: {original_text!r}"
    )


@pytest.mark.asyncio
async def test_regex_scrubs_secrets_in_multi_doc_batch(api_client) -> None:
    """Multiple items with distinct document_ids in a single POST trigger the
    multi-doc grouping recursion in retain_batch(). The recursion previously
    dropped memory_defense_extension, so screening was skipped for every item
    in the batch — even though single-item retains scrubbed correctly.
    """
    bank = "md-multi-doc-batch"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await api_client.patch(
        f"/v1/default/banks/{bank}/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )

    secrets = [
        ("anthropic", "sk-ant-" + "A" * 40),
        ("xai", "xai-" + "G" * 80),
        ("databricks", "dapi" + "L" * 32),
        ("ssn", "123-45-6789"),
    ]
    items = [
        {"content": f"User pasted {label}: {value} — scrub it.", "document_id": f"multi-doc-{label}"}
        for label, value in secrets
    ]
    r = await api_client.post(f"/v1/default/banks/{bank}/memories", json={"items": items})
    assert r.status_code == 200, r.text

    for label, value in secrets:
        r2 = await api_client.get(f"/v1/default/banks/{bank}/documents/multi-doc-{label}")
        assert r2.status_code == 200, r2.text
        body = r2.json()["original_text"]
        assert value not in body, f"{label} leaked in multi-doc batch: {body!r}"


@pytest.mark.asyncio
async def test_regex_scrubs_secrets_from_oversized_chunked_input(api_client) -> None:
    """When a single content item exceeds retain_batch_tokens (default 10k),
    `_split_contents_into_sub_batches` chunks it and carries the FULL original
    body in `document_body_override`. That override bypasses per-chunk
    screen(), so the orchestrator must scrub it before persisting.
    """
    bank = "md-doc-body-oversized"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await api_client.patch(
        f"/v1/default/banks/{bank}/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )

    # Build > 10k tokens of filler that includes a secret. ~3 chars/token, so
    # ~45KB of text is comfortably above the default 10k-token batch threshold.
    secret = "ghp_" + "Z" * 36
    ssn = "987-65-4321"
    padding = ("The quick brown fox jumps over the lazy dog. " * 50 + "\n") * 5  # ~12KB
    body = f"Audit:\n{padding}\nCredential: {secret}\nUser SSN: {ssn}\n{padding}{padding}{padding}"  # >45KB

    doc_id = "oversized-leak-1"
    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={
            "items": [{"content": body, "document_id": doc_id}],
        },
    )
    assert r.status_code == 200, r.text

    r3 = await api_client.get(f"/v1/default/banks/{bank}/documents/{doc_id}")
    assert r3.status_code == 200, r3.text
    original_text = r3.json()["original_text"]
    assert secret not in original_text, (
        f"oversized document.original_text leaked github token (length={len(original_text)})"
    )
    assert ssn not in original_text, f"oversized document.original_text leaked SSN (length={len(original_text)})"
