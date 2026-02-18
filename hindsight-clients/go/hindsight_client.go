package hindsight

import (
	"net/http"
	"time"
)

// NewAPIClientWithToken creates a new API client configured with a base URL and API token.
// The token is sent as a Bearer token in the Authorization header for all requests.
// Note: this uses http.DefaultClient which has no timeout. Use NewAPIClientWithTimeout
// to set a request timeout.
//
// Example:
//
//	client := hindsight.NewAPIClientWithToken("https://api.example.com", "your-api-token")
//	resp, _, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
func NewAPIClientWithToken(baseURL, token string) *APIClient {
	cfg := NewConfiguration()
	cfg.Servers = ServerConfigurations{
		{URL: baseURL},
	}
	cfg.AddDefaultHeader("Authorization", "Bearer "+token)
	return NewAPIClient(cfg)
}

// NewAPIClientWithTimeout creates a new API client configured with a base URL, API token,
// and a request timeout. Use 0 for no timeout.
//
// Example:
//
//	client := hindsight.NewAPIClientWithTimeout("https://api.example.com", "your-api-token", 30*time.Second)
//	resp, _, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
func NewAPIClientWithTimeout(baseURL, token string, timeout time.Duration) *APIClient {
	cfg := NewConfiguration()
	cfg.Servers = ServerConfigurations{
		{URL: baseURL},
	}
	cfg.AddDefaultHeader("Authorization", "Bearer "+token)
	cfg.HTTPClient = &http.Client{Timeout: timeout}
	return NewAPIClient(cfg)
}
