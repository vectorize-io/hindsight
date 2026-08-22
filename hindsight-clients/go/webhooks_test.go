package hindsight

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"testing"
)

type webhookSignatureVector struct {
	Secret    string `json:"secret"`
	RawBody   string `json:"raw_body"`
	Signature string `json:"signature"`
}

func loadWebhookSignatureVector(t *testing.T) webhookSignatureVector {
	t.Helper()
	contents, err := os.ReadFile("../testdata/webhook-signatures.json")
	if err != nil {
		t.Fatal(err)
	}
	var vector webhookSignatureVector
	if err := json.Unmarshal(contents, &vector); err != nil {
		t.Fatal(err)
	}
	return vector
}

func signWebhook(rawBody []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write(rawBody)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

func TestVerifyWebhookSignatureUsesExactRawBytes(t *testing.T) {
	vector := loadWebhookSignatureVector(t)
	rawBody := []byte(vector.RawBody)

	if !VerifyWebhookSignature(rawBody, vector.Signature, vector.Secret) {
		t.Fatal("expected shared signature vector to verify")
	}
	if VerifyWebhookSignature(append(rawBody, '\n'), vector.Signature, vector.Secret) {
		t.Fatal("signature must not verify after changing the raw body")
	}
	if VerifyWebhookSignature(rawBody, "SHA256="+vector.Signature[len("sha256="):], vector.Secret) {
		t.Fatal("malformed signature must not verify")
	}
}

func TestConstructWebhookEventReturnsGeneratedKnownType(t *testing.T) {
	vector := loadWebhookSignatureVector(t)
	headers := http.Header{"x-hindsight-signature": []string{vector.Signature}}

	event, err := ConstructWebhookEvent([]byte(vector.RawBody), headers, vector.Secret)
	if err != nil {
		t.Fatal(err)
	}
	retain, ok := event.(*RetainCompletedWebhookEvent)
	if !ok {
		t.Fatalf("expected *RetainCompletedWebhookEvent, got %T", event)
	}
	if retain.Data.GetDocumentId() != "doc-1" {
		t.Fatalf("unexpected document ID: %q", retain.Data.GetDocumentId())
	}
}

func TestConstructWebhookEventVerifiesBeforeParsing(t *testing.T) {
	rawBody := []byte("{")
	_, err := ConstructWebhookEvent(
		rawBody,
		http.Header{WebhookSignatureHeader: []string{"sha256=" + string(make([]byte, 64))}},
		"secret",
	)
	if !errors.Is(err, ErrInvalidWebhookSignature) {
		t.Fatalf("expected signature error, got %v", err)
	}

	_, err = ConstructWebhookEvent(
		rawBody,
		http.Header{WebhookSignatureHeader: []string{signWebhook(rawBody, "secret")}},
		"secret",
	)
	if !errors.Is(err, ErrInvalidWebhookPayload) {
		t.Fatalf("expected payload error, got %v", err)
	}
}

func TestParseWebhookEventFallsBackForUnknownEvent(t *testing.T) {
	rawBody := []byte(`{"event":"future.created","bank_id":"bank-1","operation_id":"op-2","status":"completed","timestamp":"2026-08-11T09:30:00Z","data":{"future":true},"future_envelope_field":42}`)

	event, err := ParseWebhookEvent(rawBody)
	if err != nil {
		t.Fatal(err)
	}
	envelope, ok := event.(*WebhookEventEnvelope)
	if !ok {
		t.Fatalf("expected *WebhookEventEnvelope, got %T", event)
	}
	if envelope.Data["future"] != true || envelope.AdditionalProperties["future_envelope_field"] != float64(42) {
		t.Fatalf("unknown fields were not preserved: %#v", envelope)
	}
}

func TestParseWebhookEventValidatesKnownData(t *testing.T) {
	rawBody := []byte(`{"event":"memory_defense.triggered","bank_id":"bank-1","operation_id":"op-3","status":"block","timestamp":"2026-08-11T09:30:00Z","data":{}}`)

	_, err := ParseWebhookEvent(rawBody)
	if !errors.Is(err, ErrInvalidWebhookPayload) {
		t.Fatalf("expected payload error, got %v", err)
	}
}
