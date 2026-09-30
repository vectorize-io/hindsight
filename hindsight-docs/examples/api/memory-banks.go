package main

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:create-bank]
	client.BanksAPI.CreateOrUpdateBank(ctx, "my-bank").
		CreateBankRequest(hindsight.CreateBankRequest{}).Execute()
	// [/docs:create-bank]

	// [docs:bank-with-disposition]
	client.BanksAPI.CreateOrUpdateBank(ctx, "architect-bank").
		CreateBankRequest(hindsight.CreateBankRequest{
			ReflectMission: *hindsight.NewNullableString(hindsight.PtrString(
				"You're a senior software architect - keep track of system designs, " +
					"technology decisions, and architectural patterns. Prefer simplicity over cutting-edge.",
			)),
			DispositionSkepticism: *hindsight.NewNullableInt32(hindsight.PtrInt32(4)),
			DispositionLiteralism: *hindsight.NewNullableInt32(hindsight.PtrInt32(4)),
			DispositionEmpathy:    *hindsight.NewNullableInt32(hindsight.PtrInt32(2)),
		}).Execute()
	// [/docs:bank-with-disposition]

	// [docs:bank-background]
	client.BanksAPI.CreateOrUpdateBank(ctx, "my-bank").
		CreateBankRequest(hindsight.CreateBankRequest{
			ReflectMission: *hindsight.NewNullableString(hindsight.PtrString(
				"I am a research assistant specializing in machine learning.",
			)),
		}).Execute()
	// [/docs:bank-background]

	// [docs:bank-mission]
	client.BanksAPI.CreateOrUpdateBank(ctx, "my-bank").
		CreateBankRequest(hindsight.CreateBankRequest{
			ReflectMission: *hindsight.NewNullableString(hindsight.PtrString(
				"You're a senior software architect - keep track of system designs, " +
					"technology decisions, and architectural patterns.",
			)),
		}).Execute()
	// [/docs:bank-mission]

	// [docs:bank-support-agent]
	client.BanksAPI.CreateOrUpdateBank(ctx, "support-bank").
		CreateBankRequest(hindsight.CreateBankRequest{}).Execute()
	client.BanksAPI.UpdateBankConfig(ctx, "support-bank").
		BankConfigUpdate(hindsight.BankConfigUpdate{
			Updates: map[string]interface{}{
				"observations_mission": "I am a customer support agent. Track customer preferences, " +
					"recurring issues, and resolution history to provide consistent, personalized support.",
			},
		}).Execute()
	// [/docs:bank-support-agent]

	// [docs:update-bank-config]
	client.BanksAPI.UpdateBankConfig(ctx, "my-bank").
		BankConfigUpdate(hindsight.BankConfigUpdate{
			Updates: map[string]interface{}{
				"retain_mission": "Always include technical decisions, API design choices, and architectural trade-offs. " +
					"Ignore meeting logistics and social exchanges.",
				"retain_extraction_mode": "verbose",
				"observations_mission": "Observations are stable facts about people and projects. " +
					"Always include preferences, skills, and recurring patterns. Ignore one-off events.",
				"disposition_skepticism": 4,
				"disposition_literalism": 4,
				"disposition_empathy":    2,
			},
		}).Execute()
	// [/docs:update-bank-config]

	// [docs:get-bank-config]
	// Returns resolved config (server defaults merged with bank overrides) and the raw overrides
	result, _, _ := client.BanksAPI.GetBankConfig(ctx, "my-bank").Execute()
	// result.Config     — full resolved configuration
	// result.Overrides  — only fields overridden at the bank level
	fmt.Println("Config keys:", len(result.GetConfig()))
	// [/docs:get-bank-config]

	// [docs:reset-bank-config]
	// Remove all bank-level overrides, reverting to server defaults
	client.BanksAPI.ResetBankConfig(ctx, "my-bank").Execute()
	// [/docs:reset-bank-config]

	// =============================================================================
	// Prompt preview and bank transfer
	// =============================================================================
	transferBanks := []string{"transfer-go", "transfer-go-copy", "transfer-go-other", "transfer-go-clone"}
	deleteBanks := func(ids []string) {
		for _, bankID := range ids {
			req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/%s", apiURL, bankID), nil)
			http.DefaultClient.Do(req)
		}
	}
	must := func(err error) {
		if err != nil {
			panic(err)
		}
	}
	deleteBanks(transferBanks)
	// Chunks mode stores text verbatim, so no LLM call is needed.
	_, _, err := client.BanksAPI.CreateOrUpdateBank(ctx, "transfer-go").
		CreateBankRequest(hindsight.CreateBankRequest{}).Execute()
	must(err)
	_, _, err = client.BanksAPI.UpdateBankConfig(ctx, "transfer-go").
		BankConfigUpdate(hindsight.BankConfigUpdate{
			Updates: map[string]interface{}{"retain_extraction_mode": "chunks"},
		}).Execute()
	must(err)
	resp, err := http.Post(apiURL+"/v1/default/banks/transfer-go/memories", "application/json", strings.NewReader(
		`{"items": [{"content": "Alice leads the payments team.", "document_id": "doc-1"},
		            {"content": "Bob moved to the Berlin office.", "document_id": "doc-2"}]}`))
	must(err)
	if resp.StatusCode != 200 {
		panic(fmt.Sprintf("retain failed: %d", resp.StatusCode))
	}
	_, _, err = client.BanksAPI.CreateOrUpdateBank(ctx, "transfer-go-other").
		CreateBankRequest(hindsight.CreateBankRequest{}).Execute()
	must(err)
	waitFor := func(bankID, operationID string) *hindsight.OperationStatusResponse {
		for i := 0; i < 120; i++ {
			st, _, err := client.OperationsAPI.GetOperationStatus(ctx, bankID, operationID).Execute()
			must(err)
			switch st.Status {
			case "completed":
				return st
			case "failed", "cancelled":
				panic(fmt.Sprintf("operation %s %s", operationID, st.Status))
			}
			time.Sleep(time.Second)
		}
		panic("operation " + operationID + " did not finish")
	}
	countDocs := func(bankID string) int32 {
		docs, _, err := client.DocumentsAPI.ListDocuments(ctx, bankID).Execute()
		must(err)
		return docs.Total
	}

	// [docs:prompts-preview]
	preview, _, err := client.BanksAPI.PreviewPrompt(ctx, "my-bank").
		PromptPreviewRequest(hindsight.PromptPreviewRequest{Operation: hindsight.PtrString("retain")}).
		Execute()
	for _, message := range preview.Messages {
		fmt.Println(message.Role, len(message.Blocks))
	}
	// [/docs:prompts-preview]
	must(err)

	// [docs:transfer-export]
	// Whole bank, memories + config, no history
	whole, _, err := client.BankTransferAPI.ExportBankTransfer(ctx, "transfer-go").Execute()

	// Just the memories
	memoriesOnly, _, err := client.BankTransferAPI.ExportBankTransfer(ctx, "transfer-go").
		IncludeBankConfig(false).Execute()

	// Specific documents (a document subset carries no bank-level sections)
	subset, _, err := client.BankTransferAPI.ExportBankTransfer(ctx, "transfer-go").
		DocumentId([]string{"doc-1", "doc-2"}).IncludeBankConfig(false).Execute()

	// Each returns an operation id: poll it, then download result_metadata["storage_key"]
	fmt.Println(whole.OperationId, memoriesOnly.OperationId, subset.OperationId)
	// [/docs:transfer-export]
	must(err)
	waitFor("transfer-go", memoriesOnly.OperationId)
	waitFor("transfer-go", subset.OperationId)
	exported := waitFor("transfer-go", whole.OperationId)
	archive, _, err := client.DocumentTransferAPI.
		DownloadFile(ctx, exported.ResultMetadata["storage_key"].(string)).Execute()
	must(err)
	archivePath := archive.Name()

	// [docs:transfer-import]
	// Restore a bank under a new id
	file, _ := os.Open(archivePath) // the ZIP downloaded from the export
	restore, _, err := client.BankTransferAPI.ImportBankTransfer(ctx, "transfer-go").
		File(file).TargetBankId("transfer-go-copy").Execute()
	// The restore is recorded against the bank in the URL — poll it there
	status, _, err := client.OperationsAPI.GetOperationStatus(ctx, "transfer-go", restore.OperationId).Execute()

	// Merge an archive's documents into an existing bank
	file, _ = os.Open(archivePath)
	merge, _, err := client.BankTransferAPI.ImportBankTransfer(ctx, "transfer-go-other").
		File(file).Mode("merge").DocumentConflict("replace").Execute()
	// [/docs:transfer-import]
	must(err)
	_ = status
	waitFor("transfer-go", restore.OperationId)
	waitFor("transfer-go-other", merge.OperationId)
	if n := countDocs("transfer-go-copy"); n != 2 {
		panic(fmt.Sprintf("restore copied %d documents", n))
	}

	// [docs:clone-bank]
	clone, _, err := client.BankTransferAPI.CloneBank(ctx, "transfer-go").
		TargetBankId("transfer-go-clone").Execute()
	// The operation is recorded against the source bank
	status, _, err = client.OperationsAPI.GetOperationStatus(ctx, "transfer-go", clone.OperationId).Execute()
	// [/docs:clone-bank]
	must(err)
	waitFor("transfer-go", clone.OperationId)
	if n := countDocs("transfer-go-clone"); n != 2 {
		panic(fmt.Sprintf("clone copied %d documents", n))
	}

	// [docs:document-export]
	// 1. Submit the export (whole bank; add .DocumentId([]string{...}) to scope it)
	export, _, err := client.DocumentTransferAPI.ExportDocuments(ctx, "transfer-go").Execute()

	// 2. Poll until completed
	var done *hindsight.OperationStatusResponse
	for {
		done, _, err = client.OperationsAPI.GetOperationStatus(ctx, "transfer-go", export.OperationId).Execute()
		if err != nil || done.Status == "completed" || done.Status == "failed" {
			break
		}
		time.Sleep(time.Second)
	}

	// 3. Download the archive (the client saves it to a temp file)
	zipFile, _, err := client.DocumentTransferAPI.
		DownloadFile(ctx, done.ResultMetadata["storage_key"].(string)).Execute()
	// [/docs:document-export]
	must(err)
	if done.Status != "completed" {
		panic("document export " + done.Status)
	}

	// [docs:document-import]
	file, _ = os.Open(zipFile.Name())
	imported, _, err := client.DocumentTransferAPI.ImportDocuments(ctx, "transfer-go-other").
		File(file).OnConflict("replace").Execute()

	status, _, err = client.OperationsAPI.GetOperationStatus(ctx, "transfer-go-other", imported.OperationId).Execute()
	// status.ResultMetadata -> {"documents_imported": 3, "facts_imported": 42, "observations_imported": 5, ...}
	// [/docs:document-import]
	must(err)
	waitFor("transfer-go-other", imported.OperationId)

	// [docs:transfer-import-external]
	doc := map[string]any{
		"id":            "session-2026-09-22",
		"original_text": "Full original session text...",
		"chunks":        []map[string]any{{"chunk_index": 0, "chunk_text": "Caller-defined source region..."}},
		"facts": []map[string]any{{
			"text":         "The user prefers lightweight local speech recognition models.",
			"fact_type":    "experience",
			"chunk_index":  0,
			"mentioned_at": "2026-09-22T18:34:00Z",
			"entities":     []string{"Parakeet"},
		}},
	}
	zipPath := filepath.Join(os.TempDir(), "import.zip")
	out, _ := os.Create(zipPath)
	zw := zip.NewWriter(out)
	w, _ := zw.Create("manifest.json")
	json.NewEncoder(w).Encode(map[string]any{"schema_version": 1, "source_bank_id": "external"})
	w, _ = zw.Create("documents/session-2026-09-22.json")
	json.NewEncoder(w).Encode(doc)
	zw.Close()
	out.Close()

	file, _ = os.Open(zipPath)
	external, _, err := client.BankTransferAPI.ImportBankTransfer(ctx, "transfer-go-other").
		File(file).Mode("merge").DocumentConflict("replace").Execute()
	// [/docs:transfer-import-external]
	must(err)
	waitFor("transfer-go-other", external.OperationId)
	importedDoc, _, err := client.DocumentsAPI.GetDocument(ctx, "transfer-go-other", "session-2026-09-22").Execute()
	must(err)
	if importedDoc.MemoryUnitCount != 1 {
		panic(fmt.Sprintf("external import stored %d facts", importedDoc.MemoryUnitCount))
	}
	os.Remove(zipPath)
	os.Remove(archivePath)
	os.Remove(zipFile.Name())
	deleteBanks(transferBanks)

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	for _, bankID := range []string{"my-bank", "architect-bank", "support-bank"} {
		req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/%s", apiURL, bankID), nil)
		http.DefaultClient.Do(req)
	}

	fmt.Println("memory-banks.go: All examples passed")
}
