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
		"Alice has been working there for 5 years",
		"Alice recently got promoted to senior engineer",
	} {
		client.MemoryAPI.RetainMemories(ctx, "my-bank").
			RetainRequest(hindsight.RetainRequest{
				Items: []hindsight.MemoryItem{{Content: hindsight.TextContent(content)}},
			}).Execute()
	}

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:reflect-basic]
	client.MemoryAPI.Reflect(ctx, "my-bank").
		ReflectRequest(hindsight.ReflectRequest{
			Query: "What should I know about Alice?",
		}).Execute()
	// [/docs:reflect-basic]

	// [docs:reflect-with-params]
	budgetMid := hindsight.MID
	client.MemoryAPI.Reflect(ctx, "my-bank").
		ReflectRequest(hindsight.ReflectRequest{
			Query:  "We're considering a hybrid work policy. What do you think about remote work?",
			Budget: &budgetMid,
		}).Execute()
	// [/docs:reflect-with-params]

	// [docs:reflect-with-context]
	// Context is passed to the LLM to help it understand the situation
	ctxText := "We're in a budget review meeting discussing Q4 spending"
	client.MemoryAPI.Reflect(ctx, "my-bank").
		ReflectRequest(hindsight.ReflectRequest{
			Query:   "What do you think about the proposal?",
			Context: *hindsight.NewNullableString(&ctxText),
		}).Execute()
	// [/docs:reflect-with-context]

	// [docs:reflect-disposition]
	// Create a bank with specific disposition
	skepticism := int32(5)
	literalism := int32(4)
	empathy := int32(2)
	mission := "I am a risk-aware financial advisor"
	client.BanksAPI.CreateOrUpdateBank(ctx, "cautious-advisor").
		CreateBankRequest(hindsight.CreateBankRequest{
			Name:                  *hindsight.NewNullableString(hindsight.PtrString("Cautious Advisor")),
			ReflectMission:        *hindsight.NewNullableString(&mission),
			DispositionSkepticism: *hindsight.NewNullableInt32(&skepticism),
			DispositionLiteralism: *hindsight.NewNullableInt32(&literalism),
			DispositionEmpathy:    *hindsight.NewNullableInt32(&empathy),
		}).Execute()

	// Reflect responses will reflect this disposition
	client.MemoryAPI.Reflect(ctx, "cautious-advisor").
		ReflectRequest(hindsight.ReflectRequest{
			Query: "Should I invest in crypto?",
		}).Execute()
	// Response will likely emphasize risks and caution
	// [/docs:reflect-disposition]

	// [docs:reflect-sources]
	// include.facts enables the based_on field in the response
	sourcesResponse, _, _ := client.MemoryAPI.Reflect(ctx, "my-bank").
		ReflectRequest(hindsight.ReflectRequest{
			Query: "Tell me about Alice",
			Include: &hindsight.ReflectIncludeOptions{
				Facts: map[string]interface{}{}, // empty map enables fact inclusion
			},
		}).Execute()

	fmt.Println("Response:", sourcesResponse.GetText())
	fmt.Println("\nBased on:")
	if basedOn := sourcesResponse.GetBasedOn(); basedOn.Memories != nil {
		for _, fact := range basedOn.GetMemories() {
			fmt.Printf("  - [%s] %s\n", fact.GetType(), fact.GetText())
		}
	}
	// [/docs:reflect-sources]

	// [docs:reflect-with-tags]
	// Filter reflection to only consider memories for a specific user
	tagsMatch := "any_strict"
	client.MemoryAPI.Reflect(ctx, "my-bank").
		ReflectRequest(hindsight.ReflectRequest{
			Query:     "What does this user think about our product?",
			Tags:      []string{"user:alice"},
			TagsMatch: &tagsMatch,
		}).Execute()
	// [/docs:reflect-with-tags]

	// [docs:reflect-structured-output]
	// Define JSON schema for structured output
	responseSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"recommendation": map[string]interface{}{"type": "string"},
			"confidence":     map[string]interface{}{"type": "string", "enum": []string{"low", "medium", "high"}},
			"key_factors":    map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}},
			"risks":          map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}},
		},
		"required": []string{"recommendation", "confidence", "key_factors"},
	}

	structuredResponse, _, _ := client.MemoryAPI.Reflect(ctx, "my-bank").
		ReflectRequest(hindsight.ReflectRequest{
			Query:          "Should we hire Alice for the ML team lead position?",
			ResponseSchema: responseSchema,
		}).Execute()

	// Access structured output
	if out := structuredResponse.GetStructuredOutput(); out != nil {
		fmt.Println("Recommendation:", out["recommendation"])
		fmt.Println("Key factors:", out["key_factors"])
	}
	// [/docs:reflect-structured-output]

	// =============================================================================
	// Cleanup (not shown in docs)
	// =============================================================================
	for _, bankID := range []string{"my-bank", "cautious-advisor"} {
		req, _ := http.NewRequest("DELETE", fmt.Sprintf("%s/v1/default/banks/%s", apiURL, bankID), nil)
		http.DefaultClient.Do(req)
	}

	fmt.Println("reflect.go: All examples passed")
}
