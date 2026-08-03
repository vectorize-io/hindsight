package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	hindsight "github.com/vectorize-io/hindsight/hindsight-clients/go"
)

const kpBankID = "knowledge-pages-demo-bank-go"

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
	client.BanksAPI.CreateOrUpdateBank(ctx, kpBankID).
		CreateBankRequest(hindsight.CreateBankRequest{
			Name: *hindsight.NewNullableString(hindsight.PtrString("Knowledge Pages Demo")),
		}).Execute()
	for _, content := range []string{
		"The API is deployed to Kubernetes with a rolling update",
		"Deploys run from the main branch after CI passes",
		"A failed deploy is rolled back by redeploying the previous tag",
	} {
		client.MemoryAPI.RetainMemories(ctx, kpBankID).
			RetainRequest(hindsight.RetainRequest{
				Items: []hindsight.MemoryItem{{Content: content}},
			}).Execute()
	}
	time.Sleep(2 * time.Second)

	// [docs:create-folder]
	// Create a folder (leave ParentId unset to create it at the root)
	folder, _, _ := client.KnowledgeBaseAPI.CreateKnowledgeFolder(ctx, kpBankID).
		CreateFolderRequest(hindsight.CreateFolderRequest{Name: "Operations"}).
		Execute()

	fmt.Printf("Folder ID: %s\n", folder.Id)
	// [/docs:create-folder]

	// [docs:create-page]
	// Create a page — content is generated in the background
	page, _, _ := client.KnowledgeBaseAPI.CreateKnowledgePage(ctx, kpBankID).
		CreatePageRequest(hindsight.CreatePageRequest{
			Name:        "Deploying the API",
			SourceQuery: "How is the API deployed?",
			ParentId:    *hindsight.NewNullableString(&folder.Id),
			Tags:        []string{"ops", "type:runbook"},
		}).Execute()

	// Poll the operation to know when the first build has finished
	fmt.Printf("Page ID: %s, operation: %s\n", page.PageId, page.GetOperationId())
	// [/docs:create-page]

	// Wait for the page's first build
	time.Sleep(20 * time.Second)

	// [docs:get-tree]
	// Fetch the whole knowledge base as a nested folder/page tree (no page bodies)
	tree, _, _ := client.KnowledgeBaseAPI.GetKnowledgeBaseTree(ctx, kpBankID).Execute()

	for _, root := range tree.Roots {
		fmt.Printf("%s: %s\n", root.Kind, root.Name)
		for _, child := range root.Children {
			fmt.Printf("  %s: %s (stale: %v)\n", child.Kind, child.Name, child.GetIsStale())
		}
	}
	// [/docs:get-tree]

	// [docs:get-page]
	// Read a page as a markdown document
	document, _, _ := client.KnowledgeBaseAPI.GetKnowledgePage(ctx, kpBankID, page.PageId).Execute()

	fmt.Println(document.Type)      // "runbook" — from the type:runbook tag
	fmt.Println(document.GetBody()) // the synthesized markdown body
	fmt.Println(document.Markdown)  // YAML frontmatter + body
	// [/docs:get-page]

	// [docs:search-pages]
	// Hybrid search (full-text + vector) over whole pages
	results, _, _ := client.KnowledgeBaseAPI.SearchKnowledgeBase(ctx, kpBankID).
		Q("how do we deploy").Limit(5).Execute()

	for _, hit := range results.Results {
		fmt.Printf("%.3f  %s: %s\n", hit.Score, hit.Name, hit.Snippet)
	}
	// [/docs:search-pages]

	// [docs:update-node]
	// Rename a node, move it, and/or update a page's options.
	// Changing SourceQuery rebuilds the page against the new question.
	client.KnowledgeBaseAPI.UpdateKnowledgeNode(ctx, kpBankID, page.PageId).
		UpdateNodeRequest(hindsight.UpdateNodeRequest{
			Name: *hindsight.NewNullableString(hindsight.PtrString("Deploying the API (v2)")),
			Tags: []string{"ops", "type:runbook", "reviewed"},
		}).Execute()
	// [/docs:update-node]

	// [docs:export]
	// Export the knowledge base as a portable markdown bundle
	bundle, _, _ := client.KnowledgeBaseAPI.ExportKnowledgeBase(ctx, kpBankID).Execute()

	for _, file := range bundle.Files {
		fmt.Println(file.Path) // index.md, <page-id>.md, <page-id>.log.md
	}
	// [/docs:export]

	// [docs:delete-node]
	// Delete a folder or page — deleting a folder removes its whole subtree
	client.KnowledgeBaseAPI.DeleteKnowledgeNode(ctx, kpBankID, folder.Id).Execute()
	// [/docs:delete-node]

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	cleanupKnowledgePages(apiURL)

	fmt.Println("knowledge-pages.go: All examples passed")
}

func cleanupKnowledgePages(apiURL string) {
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/%s", apiURL, kpBankID), nil)
	http.DefaultClient.Do(req)
}
