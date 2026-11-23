package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"time"

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

	// =========================================================================
	// Setup (not shown in docs): the files to upload, and a memory bank to link
	// =========================================================================
	_, thisFile, _, _ := runtime.Caller(0)
	samplePDF, err := os.ReadFile(filepath.Join(filepath.Dir(thisFile), "sample.pdf"))
	must(err)
	workDir, err := os.MkdirTemp("", "kb-example")
	must(err)
	defer os.RemoveAll(workDir)
	must(os.Chdir(workDir))
	must(os.WriteFile("x200-manual.pdf", samplePDF, 0o644))
	must(os.WriteFile("x200-quickstart.md", []byte("# X200 quick start\n\nHold the power button for two seconds.\n"), 0o644))
	_, _, err = client.BanksAPI.CreateOrUpdateBank(ctx, "support-agent").
		CreateBankRequest(hindsight.CreateBankRequest{}).Execute()
	must(err)

	// =========================================================================
	// Doc Examples
	// =========================================================================

	// [docs:create-knowledge-bank]
	_, _, err = client.KnowledgeBanksAPI.CreateKnowledgeBank(ctx).
		CreateKnowledgeBankRequest(*hindsight.NewCreateKnowledgeBankRequest("product-manuals")).
		Execute()
	// [/docs:create-knowledge-bank]
	must(err)

	// [docs:ingest-files]
	manual, _ := os.Open("x200-manual.pdf")
	quickstart, _ := os.Open("x200-quickstart.md")
	options, _ := json.Marshal(map[string]any{"files": []map[string]any{
		{"document_id": "x200-manual", "tags": []string{"x200"}},
		{"document_id": "x200-quickstart", "tags": []string{"x200", "quickstart"}},
	}})
	ingested, _, err := client.KnowledgeBanksAPI.IngestKnowledgeFiles(ctx, "product-manuals").
		Files([]*os.File{manual, quickstart}).
		Request(string(options)).
		Execute()
	// [/docs:ingest-files]
	must(err)
	fmt.Println(ingested.DocumentIds)

	// [docs:ingest-text]
	returnsID := "returns-policy"
	_, _, err = client.KnowledgeBanksAPI.IngestKnowledgeTexts(ctx, "product-manuals").
		KnowledgeIngestRequest(hindsight.KnowledgeIngestRequest{
			Items: []hindsight.KnowledgeIngestTextItem{{
				DocumentId: *hindsight.NewNullableString(&returnsID),
				Text:       "# Returns\n\nYou can return an X200 within 30 days...",
				Tags:       []string{"policy"},
			}},
		}).
		Execute()
	// [/docs:ingest-text]
	must(err)

	// [docs:search]
	query := "How long do I have to return an X200?"
	maxTokens := int32(2048)
	found, _, err := client.KnowledgeBanksAPI.SearchKnowledgeBank(ctx, "product-manuals").
		KnowledgeSearchRequest(hindsight.KnowledgeSearchRequest{
			Query:     *hindsight.NewNullableString(&query),
			MaxTokens: &maxTokens,
		}).
		Execute()
	// [/docs:search]
	must(err)
	for _, hit := range found.Results {
		fmt.Println(hit.DocumentId, hit.GetText())
	}

	// [docs:list-tags]
	tags, _, err := client.KnowledgeBanksAPI.ListKnowledgeBankTags(ctx, "product-manuals").
		Q("x2*").
		Execute()
	// [/docs:list-tags]
	must(err)
	for _, item := range tags.Items {
		fmt.Println(item.Tag, item.Count)
	}

	// [docs:link-memory-bank]
	_, _, err = client.BanksAPI.UpdateBankConfig(ctx, "support-agent").
		BankConfigUpdate(hindsight.BankConfigUpdate{
			Updates: map[string]interface{}{"knowledge_banks": []string{"product-manuals"}},
		}).
		Execute()
	// [/docs:link-memory-bank]
	must(err)

	// [docs:bank-config]
	_, _, err = client.KnowledgeBanksAPI.UpdateKnowledgeBankConfig(ctx, "product-manuals").
		KnowledgeBankConfigUpdate(hindsight.KnowledgeBankConfigUpdate{
			Updates: map[string]interface{}{
				"text_passage_tokens":  256,
				"disabled_media_types": []string{"video"},
			},
		}).
		Execute()
	// [/docs:bank-config]
	must(err)

	// =========================================================================
	// Checks (not shown in docs): the calls above must have worked, not just answered
	// =========================================================================
	foundReturns := false
	for _, hit := range found.Results {
		foundReturns = foundReturns || hit.DocumentId == "returns-policy"
	}
	check(foundReturns, "search finds the text")
	config, _, err := client.KnowledgeBanksAPI.GetKnowledgeBankConfig(ctx, "product-manuals").Execute()
	must(err)
	check(config.Overrides.GetTextPassageTokens() == 256, "override stored")
	// The file ingest runs in the background: wait for the quick start to be ready.
	var status string
	for i := 0; i < 120; i++ {
		doc, _, err := client.KnowledgeBanksAPI.GetKnowledgeDocument(ctx, "product-manuals", "x200-quickstart").Execute()
		must(err)
		status = doc.Status
		if status == "ready" || status == "failed" {
			break
		}
		time.Sleep(time.Second)
	}
	check(status == "ready", "quick start ready")

	// [docs:delete-knowledge-bank]
	_, _, err = client.KnowledgeBanksAPI.DeleteKnowledgeBank(ctx, "product-manuals").Execute()
	// [/docs:delete-knowledge-bank]
	must(err)

	// =========================================================================
	// Cleanup (not shown in docs)
	// =========================================================================
	client.BanksAPI.DeleteBank(ctx, "support-agent").Execute()

	fmt.Println("knowledge-banks.go: All examples passed")
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func check(ok bool, what string) {
	if !ok {
		log.Fatalf("check failed: %s", what)
	}
}
