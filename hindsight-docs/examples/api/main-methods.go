package main

import (
	"context"
	"fmt"
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

	// [docs:main-retain]
	// Store a fact or conversation into a memory bank
	client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{Content: hindsight.TextContent("Alice joined Google in March 2024 as a Senior ML Engineer")},
			},
		}).Execute()
	// [/docs:main-retain]

	// [docs:main-recall]
	// Search for memories using a natural language query
	resp, _, _ := client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query: "What does Alice do at Google?",
		}).Execute()

	for _, r := range resp.Results {
		fmt.Println(r.Text)
	}
	// [/docs:main-recall]

	// [docs:main-reflect]
	// Generate a reasoned response using memories and bank disposition
	answer, _, _ := client.MemoryAPI.Reflect(ctx, "my-bank").
		ReflectRequest(hindsight.ReflectRequest{
			Query: "Should we adopt TypeScript for our backend?",
		}).Execute()

	fmt.Println(answer.GetText())
	// [/docs:main-reflect]

	// Cleanup (not shown in docs)
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/my-bank", apiURL), nil)
	http.DefaultClient.Do(req)

	fmt.Println("main-methods.go: All examples passed")
}
