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

	// =============================================================================
	// Setup (not shown in docs)
	// =============================================================================
	for _, content := range []string{
		"Alice works at Google as a software engineer",
		"Alice loves hiking on weekends",
		"Bob is a data scientist who works with Alice",
	} {
		client.MemoryAPI.RetainMemories(ctx, "my-bank").
			RetainRequest(hindsight.RetainRequest{
				Items: []hindsight.MemoryItem{{Content: hindsight.TextContent(content)}},
			}).Execute()
	}

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:recall-basic]
	response, _, _ := client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query: "What does Alice do?",
		}).Execute()

	// response.Results is a slice of RecallResult, each with:
	// - Id:            fact ID
	// - Text:          the extracted fact
	// - Type:          "world", "experience", or "observation"
	// - Context:       context label set during retain
	// - Tags:          []string of tags
	// - Entities:      []string of entity names linked to this fact
	// - OccurredStart: ISO datetime of when the event started
	// - OccurredEnd:   ISO datetime of when the event ended
	// - MentionedAt:   ISO datetime of when the fact was retained
	// - DocumentId:    document this fact belongs to
	for _, r := range response.GetResults() {
		fmt.Println(r.GetText())
	}
	// [/docs:recall-basic]

	// [docs:recall-with-options]
	budgetHigh := hindsight.HIGH
	maxTokens := int32(8000)
	traceTrue := true
	detailedResponse, _, _ := client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:     "What does Alice do?",
			Types:     []string{"world", "experience"},
			Budget:    &budgetHigh,
			MaxTokens: &maxTokens,
			Trace:     &traceTrue,
		}).Execute()

	for _, r := range detailedResponse.GetResults() {
		fmt.Println("-", r.GetText())
	}
	// [/docs:recall-with-options]

	// [docs:recall-world-only]
	// Only world facts (objective information)
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query: "Where does Alice work?",
			Types: []string{"world"},
		}).Execute()
	// [/docs:recall-world-only]

	// [docs:recall-experience-only]
	// Only experience (conversations and events)
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query: "What have I recommended?",
			Types: []string{"experience"},
		}).Execute()
	// [/docs:recall-experience-only]

	// [docs:recall-observations-only]
	// Only observations (consolidated knowledge)
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query: "What patterns have I learned?",
			Types: []string{"observation"},
		}).Execute()
	// [/docs:recall-observations-only]

	// [docs:recall-source-facts]
	// Recall observations and include their source facts
	maxSFTokens := int32(4096)
	sfOpts := hindsight.SourceFactsIncludeOptions{MaxTokens: &maxSFTokens}
	obsResponse, _, _ := client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query: "What patterns have I learned about Alice?",
			Types: []string{"observation"},
			Include: &hindsight.IncludeOptions{
				SourceFacts: *hindsight.NewNullableSourceFactsIncludeOptions(&sfOpts),
			},
		}).Execute()

	for _, obs := range obsResponse.GetResults() {
		fmt.Printf("Observation: %s\n", obs.GetText())
		for _, factID := range obs.GetSourceFactIds() {
			if fact, ok := obsResponse.GetSourceFacts()[factID]; ok {
				fmt.Printf("  - [%s] %s\n", fact.GetType(), fact.GetText())
			}
		}
	}
	// [/docs:recall-source-facts]

	// [docs:recall-budget-levels]
	budgetLow := hindsight.LOW
	// Quick lookup
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:  "Alice's email",
			Budget: &budgetLow,
		}).Execute()

	// Deep exploration
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:  "How are Alice and Bob connected?",
			Budget: &budgetHigh,
		}).Execute()
	// [/docs:recall-budget-levels]

	// [docs:recall-token-budget]
	// Fill up to 4K tokens of context with relevant memories
	mt4k := int32(4096)
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:     "What do I know about Alice?",
			MaxTokens: &mt4k,
		}).Execute()

	// Smaller budget for quick lookups
	mt500 := int32(500)
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:     "Alice's email",
			MaxTokens: &mt500,
		}).Execute()
	// [/docs:recall-token-budget]

	// [docs:recall-with-tags]
	// Filter recall to only memories tagged for a specific user
	tagsMatch := "any"
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:     "What feedback did the user give?",
			Tags:      []string{"user:alice"},
			TagsMatch: &tagsMatch,
		}).Execute()
	// [/docs:recall-with-tags]

	// [docs:recall-tags-strict]
	// Strict mode: only return memories that have matching tags (exclude untagged)
	tagsMatchStrict := "any_strict"
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:     "What did the user say?",
			Tags:      []string{"user:alice"},
			TagsMatch: &tagsMatchStrict,
		}).Execute()
	// [/docs:recall-tags-strict]

	// [docs:recall-tags-all]
	// AND matching: require ALL specified tags to be present
	tagsMatchAll := "all_strict"
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:     "What bugs were reported?",
			Tags:      []string{"user:alice", "bug-report"},
			TagsMatch: &tagsMatchAll,
		}).Execute()
	// [/docs:recall-tags-all]

	// [docs:recall-tags-all-mode]
	// AND matching, includes untagged memories
	tagsMatchAllMode := "all"
	client.MemoryAPI.RecallMemories(ctx, "my-bank").
		RecallRequest(hindsight.RecallRequest{
			Query:     "communication tools",
			Tags:      []string{"user:alice", "team"},
			TagsMatch: &tagsMatchAllMode,
		}).Execute()
	// [/docs:recall-tags-all-mode]

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/my-bank", apiURL), nil)
	http.DefaultClient.Do(req)

	fmt.Println("recall.go: All examples passed")
}
