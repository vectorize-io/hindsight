package hindsight

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const WebhookSignatureHeader = "X-Hindsight-Signature"

var (
	ErrInvalidWebhookSignature = errors.New("webhook signature verification failed")
	ErrInvalidWebhookPayload   = errors.New("invalid webhook payload")
)

// ConsumedWebhookEvent is implemented by every generated known event model
// and by WebhookEventEnvelope, which represents future event types.
type ConsumedWebhookEvent interface {
	GetEvent() string
}

// VerifyWebhookSignature verifies the current sha256=<hex> signature over the
// exact raw request bytes using a constant-time digest comparison.
func VerifyWebhookSignature(rawBody []byte, signature, secret string) bool {
	if len(signature) != len("sha256=")+sha256.Size*2 || signature[:len("sha256=")] != "sha256=" {
		return false
	}
	hexDigest := signature[len("sha256="):]
	for _, char := range hexDigest {
		if !('0' <= char && char <= '9') && !('a' <= char && char <= 'f') {
			return false
		}
	}

	provided, err := hex.DecodeString(hexDigest)
	if err != nil {
		return false
	}
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write(rawBody)
	return hmac.Equal(mac.Sum(nil), provided)
}

func requiredString(fields map[string]json.RawMessage, name string) (string, error) {
	raw, ok := fields[name]
	if !ok {
		return "", fmt.Errorf("%s is required", name)
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", fmt.Errorf("%s must be a string: %w", name, err)
	}
	return value, nil
}

func parseWebhookEnvelope(rawBody []byte) (*WebhookEventEnvelope, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(rawBody, &fields); err != nil {
		return nil, err
	}
	if fields == nil {
		return nil, errors.New("payload must be a JSON object")
	}

	for _, name := range []string{"event", "bank_id", "operation_id", "status"} {
		if _, err := requiredString(fields, name); err != nil {
			return nil, err
		}
	}
	timestamp, err := requiredString(fields, "timestamp")
	if err != nil {
		return nil, err
	}
	if _, err := time.Parse(time.RFC3339, timestamp); err != nil {
		return nil, fmt.Errorf("timestamp must be an RFC 3339 date-time: %w", err)
	}

	var data map[string]json.RawMessage
	rawData, ok := fields["data"]
	if !ok {
		return nil, errors.New("data is required")
	}
	if err := json.Unmarshal(rawData, &data); err != nil || data == nil {
		return nil, errors.New("data must be a JSON object")
	}

	var envelope WebhookEventEnvelope
	if err := json.Unmarshal(rawBody, &envelope); err != nil {
		return nil, err
	}
	return &envelope, nil
}

// ParseWebhookEvent validates and parses an event without authenticating it.
// Prefer ConstructWebhookEvent unless the webhook is explicitly unsigned.
func ParseWebhookEvent(rawBody []byte) (ConsumedWebhookEvent, error) {
	envelope, err := parseWebhookEnvelope(rawBody)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidWebhookPayload, err)
	}

	switch envelope.Event {
	case "consolidation.completed":
		var event ConsolidationCompletedWebhookEvent
		if err := json.Unmarshal(rawBody, &event); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidWebhookPayload, err)
		}
		return &event, nil
	case "retain.completed":
		var event RetainCompletedWebhookEvent
		if err := json.Unmarshal(rawBody, &event); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidWebhookPayload, err)
		}
		return &event, nil
	case "memory_defense.triggered":
		var event MemoryDefenseTriggeredWebhookEvent
		if err := json.Unmarshal(rawBody, &event); err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidWebhookPayload, err)
		}
		return &event, nil
	default:
		return envelope, nil
	}
}

// ConstructWebhookEvent verifies a signed raw request body before parsing it.
// X-Hindsight-Event is intentionally ignored because it is not signed.
func ConstructWebhookEvent(rawBody []byte, headers http.Header, secret string) (ConsumedWebhookEvent, error) {
	signature := headers.Get(WebhookSignatureHeader)
	if signature == "" {
		for name, values := range headers {
			if strings.EqualFold(name, WebhookSignatureHeader) && len(values) > 0 {
				signature = values[0]
				break
			}
		}
	}
	if !VerifyWebhookSignature(rawBody, signature, secret) {
		return nil, ErrInvalidWebhookSignature
	}
	return ParseWebhookEvent(rawBody)
}
