package hindsight

import (
	"net/http"
	"runtime/debug"
	"time"
)

// defaultUserAgent returns the User-Agent string sent on every request unless
// the caller overrides cfg.UserAgent. The version is read from build info so
// it stays in sync with the module version automatically; falls back to
// "devel" when running from an unpinned local checkout.
func defaultUserAgent() string {
	version := "devel"
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range info.Deps {
			if dep.Path == "github.com/vectorize-io/hindsight/hindsight-clients/go" {
				version = dep.Version
				break
			}
		}
	}
	return "hindsight-client-go/" + version
}

// DefaultUserAgent is the User-Agent string sent on every request unless the
// caller overrides cfg.UserAgent (e.g. for integrations identifying themselves).
var DefaultUserAgent = defaultUserAgent()

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
	cfg.UserAgent = DefaultUserAgent
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
	cfg.UserAgent = DefaultUserAgent
	cfg.Servers = ServerConfigurations{
		{URL: baseURL},
	}
	cfg.AddDefaultHeader("Authorization", "Bearer "+token)
	cfg.HTTPClient = &http.Client{Timeout: timeout}
	return NewAPIClient(cfg)
}
