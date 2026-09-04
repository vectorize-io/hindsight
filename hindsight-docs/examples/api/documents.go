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

	// [docs:document-retain]
	// Retain with document ID
	docID := "meeting-2024-03-15"
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{
					Content:    hindsight.TextContent("Alice presented the Q4 roadmap..."),
					DocumentId: *hindsight.NewNullableString(&docID),
				},
			},
		}).Execute()
	// [/docs:document-retain]

	// [docs:document-update]
	// Original
	planDoc := "project-plan"
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{
					Content:    hindsight.TextContent("Project deadline: March 31"),
					DocumentId: *hindsight.NewNullableString(&planDoc),
				},
			},
		}).Execute()

	// Update (deletes old facts, creates new ones)
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{
					Content:    hindsight.TextContent("Project deadline: April 15 (extended)"),
					DocumentId: *hindsight.NewNullableString(&planDoc),
				},
			},
		}).Execute()
	// [/docs:document-update]

	// [docs:document-get]
	doc, _, err := client.DocumentsAPI.GetDocument(ctx, "my-bank", "meeting-2024-03-15").Execute()
	if err != nil {
		log.Fatalf("Failed to get document: %v", err)
	}
	fmt.Printf("Document ID: %s\n", doc.GetId())
	fmt.Printf("Memory units: %d\n", doc.GetMemoryUnitCount())
	// [/docs:document-get]

	// [docs:document-delete]
	client.DocumentsAPI.DeleteDocument(ctx, "my-bank", "meeting-2024-03-15").Execute()
	// [/docs:document-delete]

	// [docs:document-list]
	// List all documents
	docs, _, err := client.DocumentsAPI.ListDocuments(ctx, "my-bank").Execute()
	if err != nil {
		log.Fatalf("Failed to list documents: %v", err)
	}
	for _, d := range docs.Items {
		id, _ := d["id"].(string)
		memCount, _ := d["memory_unit_count"].(float64)
		fmt.Printf("%s: %d memories\n", id, int(memCount))
	}
	// [/docs:document-list]

	// Cleanup (not shown in docs)
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/my-bank", apiURL), nil)
	http.DefaultClient.Do(req)

	fmt.Println("documents.go: All examples passed")
}
