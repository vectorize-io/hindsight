package hindsight

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSemanticCandidatesResponseRequiresCompletenessProvenance(t *testing.T) {
	payload := []byte(`{
		"candidates": [],
		"limit": 10,
		"returned": 0,
		"limit_reached": false,
		"min_similarity": 0.5
	}`)

	var response SemanticCandidatesResponse
	err := json.Unmarshal(payload, &response)
	require.Error(t, err)
	require.Contains(t, err.Error(), "required property")
}

func TestSemanticCandidatesResponseDeserializesCompleteContract(t *testing.T) {
	payload := []byte(`{
		"candidates": [{"id": "memory-1", "type": "world", "score": 0.75}],
		"limit": 10,
		"returned": 1,
		"limit_reached": false,
		"exhaustive": false,
		"total_relation": "unknown",
		"min_similarity": 0.5,
		"score": {"name": "cosine_similarity", "approximate": true},
		"corpus_scope": "full_bank",
		"scope": "valid_memory_units"
	}`)

	var response SemanticCandidatesResponse
	require.NoError(t, json.Unmarshal(payload, &response))
	require.False(t, response.Exhaustive)
	require.Equal(t, "unknown", response.TotalRelation)
	require.Equal(t, "cosine_similarity", response.Score.Name)
	require.True(t, response.Score.Approximate)
	require.Equal(t, "full_bank", response.CorpusScope)
	require.Equal(t, "valid_memory_units", response.Scope)
}
