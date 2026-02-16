//go:build integration

package hindsight

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

func apiURL(t *testing.T) string {
	t.Helper()
	u := os.Getenv("HINDSIGHT_API_URL")
	if u == "" {
		u = "http://localhost:8888"
	}
	return u
}

func newClient(t *testing.T) *APIClient {
	t.Helper()
	cfg := NewConfiguration()
	cfg.Servers = ServerConfigurations{
		{URL: apiURL(t)},
	}
	return NewAPIClient(cfg)
}

func uniqueBank(t *testing.T) string {
	t.Helper()
	return fmt.Sprintf("go_test_%d", time.Now().UnixNano())
}

// --- Retain tests ---

func TestRetainSingle(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	req := RetainRequest{
		Items: []MemoryItem{
			{Content: "Alice loves artificial intelligence and machine learning"},
		},
	}

	resp, httpResp, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if !resp.GetSuccess() {
		t.Error("expected success=true")
	}
}

func TestRetainWithContext(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	timestamp := time.Date(2024, 1, 15, 10, 30, 0, 0, time.UTC)
	req := RetainRequest{
		Items: []MemoryItem{
			{
				Content:   "Bob went hiking in the mountains",
				Timestamp: *NewNullableTime(PtrTime(timestamp)),
				Context:   *NewNullableString(PtrString("outdoor activities")),
			},
		},
	}

	resp, httpResp, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if !resp.GetSuccess() {
		t.Error("expected success=true")
	}
}

func TestRetainBatch(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	req := RetainRequest{
		Items: []MemoryItem{
			{Content: "Charlie enjoys reading science fiction books"},
			{Content: "Diana is learning to play the guitar"},
			{Content: "Eve completed a marathon last month"},
		},
	}

	resp, httpResp, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if !resp.GetSuccess() {
		t.Error("expected success=true")
	}
	if resp.GetItemsCount() != 3 {
		t.Errorf("expected items_count=3, got %d", resp.GetItemsCount())
	}
}

func TestRetainWithTags(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	req := RetainRequest{
		Items: []MemoryItem{
			{
				Content: "New feature implementation for project Z",
				Tags:    []string{"project_z", "features"},
			},
		},
	}

	resp, httpResp, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if !resp.GetSuccess() {
		t.Error("expected success=true")
	}
}

func TestRetainBatchWithDocumentTags(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	req := RetainRequest{
		Items: []MemoryItem{
			{Content: "Document with tags test 1"},
			{Content: "Document with tags test 2"},
		},
		DocumentTags: []string{"test_doc", "batch"},
	}

	resp, httpResp, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if !resp.GetSuccess() {
		t.Error("expected success=true")
	}
}

// --- Recall tests ---

func setupRecallBank(t *testing.T, client *APIClient, bankID string) {
	t.Helper()
	ctx := context.Background()

	req := RetainRequest{
		Items: []MemoryItem{
			{Content: "Alice enjoys hiking in the mountains"},
			{Content: "Bob loves to read science fiction novels"},
			{Content: "Charlie is learning to play the piano"},
		},
	}

	_, _, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}

	// Give the system time to process
	time.Sleep(time.Second)
}

func TestRecallBasic(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)
	setupRecallBank(t, client, bankID)

	req := RecallRequest{
		Query: "outdoor activities",
	}

	resp, httpResp, err := client.MemoryAPI.RecallMemories(ctx, bankID).RecallRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.Results == nil {
		t.Error("expected results, got nil")
	}
}

func TestRecallWithMaxTokens(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)
	setupRecallBank(t, client, bankID)

	req := RecallRequest{
		Query:     "outdoor activities",
		MaxTokens: PtrInt32(1024),
	}

	resp, httpResp, err := client.MemoryAPI.RecallMemories(ctx, bankID).RecallRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.Results == nil {
		t.Error("expected results, got nil")
	}
}

func TestRecallFullFeatured(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)
	setupRecallBank(t, client, bankID)

	req := RecallRequest{
		Query:     "What are people's hobbies?",
		Types:     []string{"world"},
		MaxTokens: PtrInt32(2048),
		Trace:     PtrBool(true),
	}

	resp, httpResp, err := client.MemoryAPI.RecallMemories(ctx, bankID).RecallRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.Results == nil {
		t.Error("expected results, got nil")
	}

	// Verify trace data is present
	if resp.Trace != nil && len(resp.Trace) > 0 {
		t.Logf("✓ Trace data received with %d keys", len(resp.Trace))
	}
}

// --- Reflect tests ---

func setupReflectBank(t *testing.T, client *APIClient, bankID string) {
	t.Helper()
	ctx := context.Background()

	// Create bank with mission
	createReq := CreateBankRequest{
		Mission: *NewNullableString(PtrString("I am a helpful AI assistant interested in technology and science.")),
	}
	_, _, err := client.BanksAPI.CreateOrUpdateBank(ctx, bankID).CreateBankRequest(createReq).Execute()
	if err != nil {
		t.Fatal(err)
	}

	// Add memories
	retainReq := RetainRequest{
		Items: []MemoryItem{
			{Content: "Quantum computing uses quantum bits (qubits) for processing"},
			{Content: "Neural networks are inspired by biological neurons"},
		},
	}
	_, _, err = client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(retainReq).Execute()
	if err != nil {
		t.Fatal(err)
	}

	time.Sleep(time.Second)
}

func TestReflectBasic(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)
	setupReflectBank(t, client, bankID)

	req := ReflectRequest{
		Query: "What do you know about computing?",
	}

	resp, httpResp, err := client.MemoryAPI.Reflect(ctx, bankID).ReflectRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.GetText() == "" {
		t.Error("expected non-empty answer")
	}
}

func TestReflectWithMaxTokens(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)
	setupReflectBank(t, client, bankID)

	req := ReflectRequest{
		Query:     "Tell me about neural networks",
		MaxTokens: PtrInt32(500),
	}

	resp, httpResp, err := client.MemoryAPI.Reflect(ctx, bankID).ReflectRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.GetText() == "" {
		t.Error("expected non-empty answer")
	}
}

// --- Bank tests ---

func TestCreateBank(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	req := CreateBankRequest{
		Mission: *NewNullableString(PtrString("Test mission")),
	}

	resp, httpResp, err := client.BanksAPI.CreateOrUpdateBank(ctx, bankID).CreateBankRequest(req).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.GetBankId() != bankID {
		t.Errorf("expected bank_id=%s, got %s", bankID, resp.GetBankId())
	}
}

func TestSetMission(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	// Create bank with initial mission
	createReq := CreateBankRequest{
		Mission: *NewNullableString(PtrString("Initial mission")),
	}
	_, _, err := client.BanksAPI.CreateOrUpdateBank(ctx, bankID).CreateBankRequest(createReq).Execute()
	if err != nil {
		t.Fatal(err)
	}

	// Update mission by creating/updating bank again
	updateReq := CreateBankRequest{
		Mission: *NewNullableString(PtrString("Updated mission")),
	}
	resp, httpResp, err := client.BanksAPI.CreateOrUpdateBank(ctx, bankID).CreateBankRequest(updateReq).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.GetMission() != "Updated mission" {
		t.Errorf("expected mission='Updated mission', got %s", resp.GetMission())
	}
}

func TestListBanks(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()

	resp, httpResp, err := client.BanksAPI.ListBanks(ctx).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if resp.Banks == nil {
		t.Error("expected banks list, got nil")
	}
}

func TestDeleteBank(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	// Create bank
	createReq := CreateBankRequest{}
	_, _, err := client.BanksAPI.CreateOrUpdateBank(ctx, bankID).CreateBankRequest(createReq).Execute()
	if err != nil {
		t.Fatal(err)
	}

	// Delete bank
	resp, httpResp, err := client.BanksAPI.DeleteBank(ctx, bankID).Execute()
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()

	if !resp.GetSuccess() {
		t.Error("expected success=true")
	}
}

// --- End-to-end workflow test ---

func TestCompleteWorkflow(t *testing.T) {
	client := newClient(t)
	ctx := context.Background()
	bankID := uniqueBank(t)

	// 1. Create bank
	createReq := CreateBankRequest{
		Mission: *NewNullableString(PtrString("I am a helpful assistant")),
	}
	_, _, err := client.BanksAPI.CreateOrUpdateBank(ctx, bankID).CreateBankRequest(createReq).Execute()
	if err != nil {
		t.Fatal(err)
	}

	// 2. Retain memories
	retainReq := RetainRequest{
		Items: []MemoryItem{
			{Content: "Paris is the capital of France"},
			{Content: "The Eiffel Tower is in Paris"},
		},
	}
	retainResp, _, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(retainReq).Execute()
	if err != nil {
		t.Fatal(err)
	}
	if !retainResp.GetSuccess() {
		t.Error("retain failed")
	}

	time.Sleep(time.Second)

	// 3. Recall
	recallReq := RecallRequest{
		Query: "What is in Paris?",
	}
	recallResp, _, err := client.MemoryAPI.RecallMemories(ctx, bankID).RecallRequest(recallReq).Execute()
	if err != nil {
		t.Fatal(err)
	}
	if len(recallResp.Results) == 0 {
		t.Error("expected recall results")
	}

	// 4. Reflect
	reflectReq := ReflectRequest{
		Query: "Tell me about Paris",
	}
	reflectResp, _, err := client.MemoryAPI.Reflect(ctx, bankID).ReflectRequest(reflectReq).Execute()
	if err != nil {
		t.Fatal(err)
	}
	if reflectResp.GetText() == "" {
		t.Error("expected reflect answer")
	}

	t.Log("✓ Complete workflow passed")
}
