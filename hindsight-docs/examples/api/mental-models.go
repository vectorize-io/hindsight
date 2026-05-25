package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"time"

	hindsight "github.com/vectorize-io/hindsight/hindsight-clients/go"
)

const mmBankID = "mental-models-demo-bank"

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
	client.BanksAPI.CreateOrUpdateBank(ctx, mmBankID).
		CreateBankRequest(hindsight.CreateBankRequest{
			Name: *hindsight.NewNullableString(hindsight.PtrString("Mental Models Demo")),
		}).Execute()
	for _, content := range []string{
		"The team prefers async communication via Slack",
		"For urgent issues, use the #incidents channel",
		"Weekly syncs happen every Monday at 10am",
	} {
		client.MemoryAPI.RetainMemories(ctx, mmBankID).
			RetainRequest(hindsight.RetainRequest{
				Items: []hindsight.MemoryItem{{Content: content}},
			}).Execute()
	}
	time.Sleep(2 * time.Second)

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:create-mental-model]
	// Create a mental model (runs reflect in background)
	result, _, _ := client.MentalModelsAPI.CreateMentalModel(ctx, mmBankID).
		CreateMentalModelRequest(hindsight.CreateMentalModelRequest{
			Name:        "Team Communication Preferences",
			SourceQuery: "How does the team prefer to communicate?",
			Tags:        []string{"team", "communication"},
		}).Execute()

	// Returns an operation_id — check operations endpoint for completion
	fmt.Printf("Operation ID: %s\n", result.GetOperationId())
	// [/docs:create-mental-model]

	// [docs:create-mental-model-with-id]
	// Create a mental model with a specific custom ID
	mmID := "communication-policy"
	resultWithID, _, _ := client.MentalModelsAPI.CreateMentalModel(ctx, mmBankID).
		CreateMentalModelRequest(hindsight.CreateMentalModelRequest{
			Id:          *hindsight.NewNullableString(&mmID),
			Name:        "Communication Policy",
			SourceQuery: "What are the team's communication guidelines?",
		}).Execute()

	fmt.Printf("Created with custom ID: %s\n", resultWithID.GetOperationId())
	// [/docs:create-mental-model-with-id]

	time.Sleep(5 * time.Second)

	// [docs:create-mental-model-with-trigger]
	// Create a mental model with automatic refresh enabled
	refreshTrue := true
	result2, _, _ := client.MentalModelsAPI.CreateMentalModel(ctx, mmBankID).
		CreateMentalModelRequest(hindsight.CreateMentalModelRequest{
			Name:        "Project Status",
			SourceQuery: "What is the current project status?",
			Trigger: &hindsight.MentalModelTriggerInput{
				RefreshAfterConsolidation: &refreshTrue,
			},
		}).Execute()

	// This mental model will automatically refresh when observations are updated
	fmt.Printf("Operation ID: %s\n", result2.GetOperationId())
	// [/docs:create-mental-model-with-trigger]

	time.Sleep(5 * time.Second)

	// [docs:list-mental-models]
	// List all mental models in a bank
	mentalModels, _, _ := client.MentalModelsAPI.ListMentalModels(ctx, mmBankID).Execute()

	for _, mm := range mentalModels.GetItems() {
		fmt.Printf("- %s: %s\n", mm.GetName(), mm.GetSourceQuery())
	}
	// [/docs:list-mental-models]

	if len(mentalModels.GetItems()) == 0 {
		fmt.Println("mental-models.go: All examples passed (no mental models created yet)")
		cleanupMentalModels(client, ctx, apiURL)
		return
	}

	mentalModelID := mentalModels.GetItems()[0].GetId()

	// [docs:get-mental-model]
	// Get a specific mental model
	mentalModel, _, _ := client.MentalModelsAPI.GetMentalModel(ctx, mmBankID, mentalModelID).Execute()

	fmt.Printf("Name: %s\n", mentalModel.GetName())
	fmt.Printf("Content: %s\n", mentalModel.GetContent())
	fmt.Printf("Last refreshed: %s\n", mentalModel.GetLastRefreshedAt())
	// [/docs:get-mental-model]

	// [docs:refresh-mental-model]
	// Refresh a mental model to update with current knowledge
	refreshResult, _, _ := client.MentalModelsAPI.RefreshMentalModel(ctx, mmBankID, mentalModelID).Execute()

	fmt.Printf("Refresh operation ID: %s\n", refreshResult.GetOperationId())
	// [/docs:refresh-mental-model]

	// [docs:clear-mental-model]
	// Clear a mental model's content, then refresh for a full re-synthesis
	client.MentalModelsAPI.ClearMentalModel(ctx, mmBankID, mentalModelID).Execute()

	// Trigger a fresh full rebuild
	fullRefreshResult, _, _ := client.MentalModelsAPI.RefreshMentalModel(ctx, mmBankID, mentalModelID).Execute()

	fmt.Printf("Full refresh operation ID: %s\n", fullRefreshResult.GetOperationId())
	// [/docs:clear-mental-model]

	// [docs:update-mental-model]
	// Update a mental model's metadata
	newName := "Updated Team Communication Preferences"
	refreshAfter := true
	updated, _, _ := client.MentalModelsAPI.UpdateMentalModel(ctx, mmBankID, mentalModelID).
		UpdateMentalModelRequest(hindsight.UpdateMentalModelRequest{
			Name: *hindsight.NewNullableString(&newName),
			Trigger: *hindsight.NewNullableMentalModelTriggerInput(&hindsight.MentalModelTriggerInput{
				RefreshAfterConsolidation: &refreshAfter,
			}),
		}).Execute()

	fmt.Printf("Updated name: %s\n", updated.GetName())
	// [/docs:update-mental-model]

	// [docs:get-mental-model-history]
	// Get the change history of a mental model
	history, _, _ := client.MentalModelsAPI.GetMentalModelHistory(ctx, mmBankID, mentalModelID).Execute()

	if entries, ok := history.([]interface{}); ok {
		for _, entry := range entries {
			if e, ok := entry.(map[string]interface{}); ok {
				fmt.Printf("Changed at: %v\n", e["changed_at"])
				fmt.Printf("Previous content: %v\n", e["previous_content"])
			}
		}
	}
	// [/docs:get-mental-model-history]

	// [docs:delete-mental-model]
	// Delete a mental model
	client.MentalModelsAPI.DeleteMentalModel(ctx, mmBankID, mentalModelID).Execute()
	// [/docs:delete-mental-model]

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	cleanupMentalModels(client, ctx, apiURL)

	fmt.Println("mental-models.go: All examples passed")
}

func cleanupMentalModels(client *hindsight.APIClient, ctx context.Context, apiURL string) {
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/%s", apiURL, mmBankID), nil)
	http.DefaultClient.Do(req)
}
