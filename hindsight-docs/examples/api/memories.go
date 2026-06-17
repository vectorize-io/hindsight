package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	hindsight "github.com/vectorize-io/hindsight/hindsight-clients/go"
)

const memBankID = "memories-demo-bank"

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
	client.BanksAPI.CreateOrUpdateBank(ctx, memBankID).
		CreateBankRequest(hindsight.CreateBankRequest{
			Name: *hindsight.NewNullableString(hindsight.PtrString("Memories Demo")),
		}).Execute()
	for _, content := range []string{
		"The assistant visited Paris in 2023.",
		"The deploy server srv-04 runs PostgreSQL 14.",
	} {
		client.MemoryAPI.RetainMemories(ctx, memBankID).
			RetainRequest(hindsight.RetainRequest{
				Items: []hindsight.MemoryItem{{Content: content}},
			}).Execute()
	}
	time.Sleep(3 * time.Second)

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:list-memories]
	// List memory units in a bank. Invalidated rows are included by default.
	memories, _, _ := client.MemoryAPI.ListMemories(ctx, memBankID).Execute()
	for _, unit := range memories.GetItems() {
		fmt.Printf("- [%v] %v\n", unit["fact_type"], unit["text"])
	}

	// Filter to only the invalidated facts (e.g. to review duplicates).
	invalidated, _, _ := client.MemoryAPI.ListMemories(ctx, memBankID).State("invalidated").Execute()
	fmt.Printf("%d invalidated fact(s)\n", len(invalidated.GetItems()))
	// [/docs:list-memories]

	// Pick a raw fact (world/experience) to curate below.
	var memoryID string
	for _, unit := range memories.GetItems() {
		if ft, _ := unit["fact_type"].(string); ft == "world" || ft == "experience" {
			memoryID, _ = unit["id"].(string)
			break
		}
	}

	if memoryID != "" {
		// [docs:get-memory]
		// Fetch a single memory unit (entities, dates, state).
		memory, _, _ := client.MemoryAPI.GetMemory(ctx, memBankID, memoryID).Execute()
		fmt.Printf("Memory: %v\n", memory)
		// [/docs:get-memory]

		// [docs:edit-memory]
		// Correct the fact's text. Re-embeds, drops derived observations/links,
		// re-consolidates, and recomputes the graph automatically.
		client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
			UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
				Text:   *hindsight.NewNullableString(hindsight.PtrString("The user visited Paris in 2023.")),
				Reason: *hindsight.NewNullableString(hindsight.PtrString("wrong subject")),
			}).Execute()
		// [/docs:edit-memory]

		// [docs:edit-memory-fields]
		// Correct dates, fact type, and entities in one call. "" clears a field;
		// entities replaces the set ([] detaches all); omit to leave unchanged.
		client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
			UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
				OccurredStart: *hindsight.NewNullableString(hindsight.PtrString("2023-06-01")),
				FactType:      *hindsight.NewNullableString(hindsight.PtrString("experience")),
				Entities:      []string{"Alice", "Paris"},
			}).Execute()
		// [/docs:edit-memory-fields]

		// [docs:invalidate-memory]
		// Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
		// derived observations recomputed without it — but kept for audit.
		client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
			UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
				State:  *hindsight.NewNullableString(hindsight.PtrString("invalidated")),
				Reason: *hindsight.NewNullableString(hindsight.PtrString("server decommissioned 2026-06-01")),
			}).Execute()
		// [/docs:invalidate-memory]

		// [docs:restore-memory]
		// Restore a previously invalidated fact.
		client.MemoryAPI.UpdateMemory(ctx, memBankID, memoryID).
			UpdateMemoryRequest(hindsight.UpdateMemoryRequest{
				State: *hindsight.NewNullableString(hindsight.PtrString("valid")),
			}).Execute()
		// [/docs:restore-memory]
	}

	// An observation (derived) exposes how it evolved as sources arrived.
	var observationID string
	for _, unit := range memories.GetItems() {
		if ft, _ := unit["fact_type"].(string); ft == "observation" {
			observationID, _ = unit["id"].(string)
			break
		}
	}
	if observationID != "" {
		// [docs:observation-history]
		// Get the refresh history of a derived observation.
		history, _, _ := client.MemoryAPI.GetObservationHistory(ctx, memBankID, observationID).Execute()
		fmt.Printf("Observation history: %v\n", history)
		// [/docs:observation-history]
	}

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/%s", apiURL, memBankID), nil)
	http.DefaultClient.Do(req)

	fmt.Println("memories.go: All examples passed")
}
