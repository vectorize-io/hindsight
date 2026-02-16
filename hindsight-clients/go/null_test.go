package hindsight

import (
	"context"
	"os"
	"testing"
)

// Test that the client can handle null values in responses
func TestNullHandling(t *testing.T) {
	apiURL := os.Getenv("HINDSIGHT_API_URL")
	if apiURL == "" {
		apiURL = "http://localhost:8888"
	}

	cfg := NewConfiguration()
	cfg.Servers = ServerConfigurations{
		{URL: apiURL},
	}

	client := NewAPIClient(cfg)
	ctx := context.Background()

	// Test retain which returns operation_id as null
	req := RetainRequest{
		Items: []MemoryItem{
			{Content: "Test content for null handling"},
		},
	}

	resp, httpResp, err := client.MemoryAPI.RetainMemories(ctx, "test_null_bank").RetainRequest(req).Execute()
	if err != nil {
		t.Fatalf("Retain failed: %v", err)
	}
	defer httpResp.Body.Close()

	if !resp.GetSuccess() {
		t.Error("Expected success=true")
	}

	// Check that operation_id can be accessed even if null
	if resp.HasOperationId() {
		t.Logf("OperationId is set: %s", resp.GetOperationId())
	} else {
		t.Log("OperationId is not set (null or omitted) - this is OK!")
	}

	t.Logf("✅ Successfully handled response with nullable fields")
}
