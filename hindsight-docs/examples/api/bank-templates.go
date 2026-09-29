package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

func main() {
	apiURL := os.Getenv("HINDSIGHT_API_URL")
	if apiURL == "" {
		apiURL = "http://localhost:8888"
	}

	// =============================================================================
	// Doc Examples
	// =============================================================================

	// [docs:import-template]
	template := map[string]interface{}{
		"version": "1",
		"bank": map[string]interface{}{
			"retain_mission":      "Extract customer issues, resolutions, and sentiment.",
			"enable_observations": true,
			"observations_mission": "Track recurring customer pain points.",
		},
		"mental_models": []map[string]interface{}{
			{
				"id":           "sentiment-overview",
				"name":         "Customer Sentiment Overview",
				"source_query": "What is the overall sentiment trend?",
				"trigger":      map[string]interface{}{"refresh_after_consolidation": true},
			},
		},
		"directives": []map[string]interface{}{
			{
				"name":     "Acknowledge frustration",
				"content":  "Always acknowledge frustration before offering solutions.",
				"priority": 10,
			},
		},
	}

	body, _ := json.Marshal(template)
	resp, _ := http.Post(
		apiURL+"/v1/default/banks/my-bank/import",
		"application/json",
		bytes.NewReader(body),
	)
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	fmt.Println(string(respBody))
	// [/docs:import-template]

	// [docs:import-dry-run]
	resp, _ = http.Post(
		apiURL+"/v1/default/banks/my-bank/import?dry_run=true",
		"application/json",
		bytes.NewReader(body),
	)
	defer resp.Body.Close()
	dryRunBody, _ := io.ReadAll(resp.Body)
	fmt.Println(string(dryRunBody))
	// [/docs:import-dry-run]

	// [docs:export-template]
	resp, _ = http.Get(apiURL + "/v1/default/banks/my-bank/export")
	defer resp.Body.Close()
	exported, _ := io.ReadAll(resp.Body)
	fmt.Println(string(exported))
	// [/docs:export-template]

	// [docs:export-reimport]
	// Export from source bank
	resp, _ = http.Get(apiURL + "/v1/default/banks/source-bank/export")
	defer resp.Body.Close()
	srcExported, _ := io.ReadAll(resp.Body)

	// Import into a new bank
	resp, _ = http.Post(
		apiURL+"/v1/default/banks/new-bank/import",
		"application/json",
		bytes.NewReader(srcExported),
	)
	defer resp.Body.Close()
	// [/docs:export-reimport]

	// [docs:get-schema]
	resp, _ = http.Get(apiURL + "/v1/bank-template-schema")
	defer resp.Body.Close()
	schema, _ := io.ReadAll(resp.Body)
	fmt.Println(string(schema))
	// [/docs:get-schema]
}
