// Package hindsight calls the Hindsight bank template API.
//
// Manifests pass through as raw JSON instead of the generated Go client's
// typed models, so bank settings added in later Hindsight versions work
// without an operator change.
package hindsight

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// APIError is a non-2xx response from the Hindsight API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Body)
}

// Client calls one Hindsight API.
type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// NewClient returns a client for baseURL. apiKey may be empty.
func NewClient(baseURL, apiKey string, httpClient *http.Client) *Client {
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey, http: httpClient}
}

func bankPath(bankID string) string {
	return "/v1/default/banks/" + url.PathEscape(bankID)
}

// Import applies a template manifest, creating the bank when it is missing.
// Import skips unchanged mental models and directives, so repeating it is cheap.
func (c *Client) Import(ctx context.Context, bankID string, manifest json.RawMessage) error {
	return c.do(ctx, http.MethodPost, bankPath(bankID)+"/import", manifest)
}

func (c *Client) do(ctx context.Context, method, path string, body json.RawMessage) error {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 200 && resp.StatusCode <= 299 {
		return nil
	}
	payload, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	return &APIError{StatusCode: resp.StatusCode, Body: strings.TrimSpace(string(payload))}
}
