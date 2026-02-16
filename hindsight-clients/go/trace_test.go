package hindsight

import (
	"context"
	"os"
	"testing"
)

// Test that the client can handle large responses with trace
func TestTraceResponse(t *testing.T) {
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

	// First retain some data
	retainReq := RetainRequest{
		Items: []MemoryItem{
			{Content: "The sky is blue"},
		},
	}
	_, _, err := client.MemoryAPI.RetainMemories(ctx, "test_trace_bank").RetainRequest(retainReq).Execute()
	if err != nil {
		t.Fatalf("Retain failed: %v", err)
	}

	// Now recall with trace enabled
	recallReq := RecallRequest{
		Query:     "What color is the sky?",
		MaxTokens: PtrInt32(2048),
		Trace:     PtrBool(true), // This was failing with ogen
		Types:     []string{"world"},
	}

	resp, httpResp, err := client.MemoryAPI.RecallMemories(ctx, "test_trace_bank").
		RecallRequest(recallReq).
		Execute()

	if err != nil {
		t.Fatalf("Recall with trace failed: %v (this was the bug!)", err)
	}
	defer httpResp.Body.Close()

	if resp.Trace != nil && len(resp.Trace) > 0 {
		t.Logf("✅ Successfully received trace data with %d keys!", len(resp.Trace))
	} else {
		t.Log("No trace data in response")
	}

	t.Logf("✅ Large trace response handled successfully (this was failing before!)")
}
