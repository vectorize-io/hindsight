package hindsight

// NewAPIClientWithToken creates a new API client configured with a base URL and API token.
// The token is sent as a Bearer token in the Authorization header for all requests.
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
