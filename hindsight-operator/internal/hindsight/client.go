// Package hindsight is a minimal client for the Hindsight bank template API.
//
// It is deliberately separate from hindsight-clients/go: that module is
// generated with typed models for one server version, while the operator
// passes template manifests through as raw JSON so that new bank settings
// work without an operator release.
package hindsight

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ErrNotFound is returned when the bank does not exist.
var ErrNotFound = errors.New("bank not found")

// APIError is a non-2xx response from the Hindsight API.
type APIError struct {
	Method     string
	Path       string
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s %s: HTTP %d: %s", e.Method, e.Path, e.StatusCode, e.Body)
}

// Permanent reports whether retrying the same request cannot succeed:
// the manifest was rejected or the credentials are wrong.
func (e *APIError) Permanent() bool {
	return e.StatusCode == http.StatusBadRequest ||
		e.StatusCode == http.StatusUnauthorized ||
		e.StatusCode == http.StatusForbidden ||
		e.StatusCode == http.StatusUnprocessableEntity
}

// ImportResult is the response of a template import.
type ImportResult struct {
	ConfigApplied       bool     `json:"config_applied"`
	MentalModelsCreated []string `json:"mental_models_created"`
	MentalModelsUpdated []string `json:"mental_models_updated"`
	DirectivesCreated   []string `json:"directives_created"`
	DirectivesUpdated   []string `json:"directives_updated"`
	OperationIDs        []string `json:"operation_ids"`
}

// Client calls one Hindsight API.
type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// NewClient returns a client for baseURL. apiKey may be empty.
func NewClient(baseURL, apiKey string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 60 * time.Second}
	}
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey, http: httpClient}
}

func bankPath(bankID string) string {
	return "/v1/default/banks/" + url.PathEscape(bankID)
}

// Export returns the bank's current template manifest, or ErrNotFound.
func (c *Client) Export(ctx context.Context, bankID string) (map[string]any, error) {
	var manifest map[string]any
	if err := c.do(ctx, http.MethodGet, bankPath(bankID)+"/export", nil, &manifest); err != nil {
		return nil, err
	}
	return manifest, nil
}

// Import applies a template manifest. The API creates the bank when it does
// not exist. With dryRun the API only validates the manifest.
func (c *Client) Import(ctx context.Context, bankID string, manifest map[string]any, dryRun bool) (*ImportResult, error) {
	path := bankPath(bankID) + "/import"
	if dryRun {
		path += "?dry_run=true"
	}
	var result ImportResult
	if err := c.do(ctx, http.MethodPost, path, manifest, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Defaults reads the server's default values for mental model, trigger, and
// directive fields from the template JSON Schema. The operator needs them to
// tell "field omitted" from "field changed", and reading them from the server
// keeps it correct across Hindsight versions.
func (c *Client) Defaults(ctx context.Context) (Defaults, error) {
	var schema struct {
		Defs map[string]struct {
			Properties map[string]struct {
				Default json.RawMessage `json:"default"`
			} `json:"properties"`
		} `json:"$defs"`
	}
	if err := c.do(ctx, http.MethodGet, "/v1/bank-template-schema", nil, &schema); err != nil {
		return Defaults{}, err
	}
	read := func(name string) map[string]any {
		out := map[string]any{}
		for field, prop := range schema.Defs[name].Properties {
			if len(prop.Default) == 0 {
				continue
			}
			var value any
			if err := json.Unmarshal(prop.Default, &value); err == nil && value != nil {
				out[field] = value
			}
		}
		return out
	}
	return Defaults{
		MentalModel: read("BankTemplateMentalModel"),
		Trigger:     read("MentalModelTrigger"),
		Directive:   read("BankTemplateDirective"),
	}, nil
}

// DeleteBank deletes the bank and all of its data. A missing bank is not an error.
func (c *Client) DeleteBank(ctx context.Context, bankID string) error {
	err := c.do(ctx, http.MethodDelete, bankPath(bankID), nil, nil)
	if errors.Is(err, ErrNotFound) {
		return nil
	}
	return err
}

func (c *Client) do(ctx context.Context, method, path string, body, out any) error {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
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
	payload, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("%s %s: %w", method, path, ErrNotFound)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return &APIError{Method: method, Path: path, StatusCode: resp.StatusCode, Body: truncate(string(payload), 512)}
	}
	if out == nil || len(payload) == 0 {
		return nil
	}
	return json.Unmarshal(payload, out)
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
