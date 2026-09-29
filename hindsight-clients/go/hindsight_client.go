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

// TextContent wraps a plain string as a retain item's content.
//
// `content` accepts either a string or an ordered list of content blocks, so
// the generated type is a union and a bare string no longer satisfies it. That
// is a poor trade for the overwhelmingly common case — a caller retaining text
// should not have to take the address of a temporary — so this restores it:
//
//	Items: []MemoryItem{{Content: hindsight.TextContent("Alice joined Google")}}
//
// Use the ArrayOfContentAnyOfInner field directly to interleave attachments.
func TextContent(text string) Content {
	return Content{String: &text}
}
