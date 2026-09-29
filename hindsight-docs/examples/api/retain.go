package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	hindsight "github.com/vectorize-io/hindsight/hindsight-clients/go"
)

func main() {
	apiURL := os.Getenv("HINDSIGHT_API_URL")
	if apiURL == "" {
		apiURL = "http://localhost:8888"
	}

	cfg := hindsight.NewConfiguration()
	cfg.Servers = hindsight.ServerConfigurations{{URL: apiURL}}
	client := hindsight.NewAPIClient(cfg)
	ctx := context.Background()

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:retain-basic]
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{Content: hindsight.TextContent("Alice works at Google as a software engineer")},
			},
		}).Execute()
	// [/docs:retain-basic]

	// [docs:retain-conversation]
	// Retain an entire conversation as a single document.
	conversation := "Alice (2024-03-15T09:00:00Z): Hi Bob! Did you end up going to the doctor last week?\n" +
		"Bob (2024-03-15T09:01:00Z): Yes, finally. Turns out I have a mild peanut allergy.\n" +
		"Alice (2024-03-15T09:02:00Z): Oh no! Are you okay?\n" +
		"Bob (2024-03-15T09:03:00Z): Yeah, nothing serious. Just need to carry an antihistamine.\n" +
		"Alice (2024-03-15T09:04:00Z): Good to know. We'll avoid peanuts at the team lunch."

	docID := "chat-2024-03-15-alice-bob"
	context_ := "team chat"
	ts := "2024-03-15T09:04:00Z"
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{
					Content:    hindsight.TextContent(conversation),
					Context:    *hindsight.NewNullableString(&context_),
					DocumentId: *hindsight.NewNullableString(&docID),
					Timestamp: *hindsight.NewNullableTimestamp(&hindsight.Timestamp{
						String: &ts,
					}),
				},
			},
		}).Execute()
	// [/docs:retain-conversation]

	// [docs:retain-with-context]
	ctxLabel := "career update"
	ts2 := "2024-03-15T10:00:00Z"
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{
					Content: hindsight.TextContent("Alice got promoted to senior engineer"),
					Context: *hindsight.NewNullableString(&ctxLabel),
					Timestamp: *hindsight.NewNullableTimestamp(&hindsight.Timestamp{
						String: &ts2,
					}),
				},
			},
		}).Execute()
	// [/docs:retain-with-context]

	// [docs:retain-batch]
	doc1 := "conversation_001_msg_1"
	doc2 := "conversation_001_msg_2"
	doc3 := "conversation_001_msg_3"
	ctx1 := "career"
	ctx2 := "relationship"
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{Content: hindsight.TextContent("Alice works at Google"), Context: *hindsight.NewNullableString(&ctx1), DocumentId: *hindsight.NewNullableString(&doc1)},
				{Content: hindsight.TextContent("Bob is a data scientist at Meta"), Context: *hindsight.NewNullableString(&ctx1), DocumentId: *hindsight.NewNullableString(&doc2)},
				{Content: hindsight.TextContent("Alice and Bob are friends"), Context: *hindsight.NewNullableString(&ctx2), DocumentId: *hindsight.NewNullableString(&doc3)},
			},
		}).Execute()
	// [/docs:retain-batch]

	// [docs:retain-async]
	// Start async ingestion (returns immediately)
	asyncTrue := true
	largeDoc1 := "large-doc-1"
	largeDoc2 := "large-doc-2"
	retainResp, _, _ := client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{Content: hindsight.TextContent("Large batch item 1"), DocumentId: *hindsight.NewNullableString(&largeDoc1)},
				{Content: hindsight.TextContent("Large batch item 2"), DocumentId: *hindsight.NewNullableString(&largeDoc2)},
			},
			Async: &asyncTrue,
		}).Execute()

	// Check if it was processed asynchronously
	fmt.Println("Async:", retainResp.GetAsync())
	// [/docs:retain-async]

	// [docs:retain-files]
	// Open a file and upload it — Hindsight converts it to text and extracts memories.
	// Supports: PDF, DOCX, PPTX, XLSX, images (OCR), audio (transcription), and text formats.
	f, err := os.Open("../../hindsight-docs/examples/api/sample.pdf")
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer f.Close()

	fileResp, _, _ := client.FilesAPI.FileRetain(ctx, "my-bank").
		Files([]*os.File{f}).
		Request(`{"files_metadata": [{"context": "quarterly report"}]}`).
		Execute()
	fmt.Println("Operation IDs:", fileResp.GetOperationIds()) // Track processing via the operations endpoint
	// [/docs:retain-files]

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/my-bank", apiURL), nil)
	http.DefaultClient.Do(req)

	fmt.Println("retain.go: All examples passed")
}
