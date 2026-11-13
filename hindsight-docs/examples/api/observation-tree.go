package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	hindsight "github.com/vectorize-io/hindsight/hindsight-clients/go"
)

const mtBankID = "observation-tree-demo-bank-go"

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
	// Setup (not shown in docs)
	// =============================================================================
	client.BanksAPI.CreateOrUpdateBank(ctx, mtBankID).
		CreateBankRequest(hindsight.CreateBankRequest{
			Name: *hindsight.NewNullableString(hindsight.PtrString("Observation Tree Demo")),
		}).Execute()
	for _, content := range []string{
		"The API runs on Kubernetes behind an nginx ingress.",
		"Redis keys expire after an hour.",
		"The GPU nodes serve Gemma with vLLM.",
	} {
		client.MemoryAPI.RetainMemories(ctx, mtBankID).
			RetainRequest(hindsight.RetainRequest{
				Items: []hindsight.MemoryItem{{Content: hindsight.TextContent(content), Tags: []string{"team:platform"}}},
			}).Execute()
	}
	time.Sleep(5 * time.Second)
	outDir, _ := os.MkdirTemp("", "observation-tree-")

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:get-tree]
	// The whole bank: folders, topic files and each folder's index.md.
	tree, _, err := client.MemoryAPI.GetObservationTree(ctx, mtBankID).Execute()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%d observations\n", tree.GetObservationCount())

	var walk func(folder hindsight.ObservationTreeFolder)
	walk = func(folder hindsight.ObservationTreeFolder) {
		for _, f := range folder.GetFiles() {
			fmt.Printf("/%s (%d)\n", f.GetPath(), f.GetObservationCount())
		}
		for _, sub := range folder.GetFolders() {
			walk(sub)
		}
	}
	walk(*tree)

	// Only one scope: the same tag filters as recall.
	platform, _, err := client.MemoryAPI.GetObservationTree(ctx, mtBankID).
		Tags([]string{"team:platform"}).TagsMatch("any_strict").Execute()
	if err != nil {
		panic(err)
	}
	walk(*platform)
	// [/docs:get-tree]

	// [docs:write-bundle]
	// Write the bundle to disk: each file's markdown at its path.
	var write func(folder hindsight.ObservationTreeFolder)
	write = func(folder hindsight.ObservationTreeFolder) {
		for _, f := range folder.GetFiles() {
			path := filepath.Join(outDir, f.GetPath())
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				panic(err)
			}
			if err := os.WriteFile(path, []byte(f.GetContent()), 0o644); err != nil {
				panic(err)
			}
		}
		for _, sub := range folder.GetFolders() {
			write(sub)
		}
	}
	write(*tree)
	// [/docs:write-bundle]
	if _, err := os.Stat(filepath.Join(outDir, "index.md")); err != nil {
		panic(err)
	}

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/%s", apiURL, mtBankID), nil)
	http.DefaultClient.Do(req)

	fmt.Println("observation-tree.go: All examples passed")
}
