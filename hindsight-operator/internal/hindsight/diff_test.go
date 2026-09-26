package hindsight

import (
	"encoding/json"
	"reflect"
	"testing"
)

var testDefaults = Defaults{
	MentalModel: map[string]any{"tags": []any{}, "max_tokens": float64(2048), "trigger": map[string]any{}},
	Trigger:     map[string]any{"mode": "full", "refresh_after_consolidation": false, "exclude_mental_models": false},
	Directive:   map[string]any{"priority": float64(0), "is_active": true, "tags": []any{}},
}

func decode(t *testing.T, s string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

// The export fills in every server default; a template that omits them must
// still count as in sync, or the operator would regenerate summaries forever.
func TestPlanTreatsOmittedDefaultsAsInSync(t *testing.T) {
	desired := decode(t, `{"version":"1","mental_models":[{"id":"prefs","name":"Prefs","source_query":"q"}],
		"directives":[{"name":"d","content":"c"}]}`)
	current := decode(t, `{"version":"1","bank":{"retain_mission":"x"},
		"mental_models":[{"id":"prefs","name":"Prefs","source_query":"q","tags":[],"max_tokens":2048,
			"trigger":{"mode":"full","refresh_after_consolidation":false,"exclude_mental_models":false}}],
		"directives":[{"name":"d","content":"c","priority":0,"is_active":true,"tags":[]}]}`)

	plan, err := ComputePlan(desired, current, testDefaults)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Empty() {
		t.Fatalf("expected no changes, got %v", plan.Changes)
	}
}

// Only changed items may be sent: import regenerates every mental model it
// receives, including unchanged ones.
func TestPlanSendsOnlyChangedItems(t *testing.T) {
	desired := decode(t, `{"version":"1",
		"bank":{"retain_mission":"new","reflect_mission":"same"},
		"mental_models":[
			{"id":"same","name":"Same","source_query":"q"},
			{"id":"edited","name":"Edited","source_query":"new query","trigger":{"mode":"delta"}},
			{"id":"added","name":"Added","source_query":"q"}]}`)
	current := decode(t, `{"version":"1",
		"bank":{"retain_mission":"old","reflect_mission":"same","observations_mission":"untouched"},
		"mental_models":[
			{"id":"same","name":"Same","source_query":"q"},
			{"id":"edited","name":"Edited","source_query":"old query","trigger":{"mode":"delta"}},
			{"id":"unmanaged","name":"Other","source_query":"q"}]}`)

	plan, err := ComputePlan(desired, current, testDefaults)
	if err != nil {
		t.Fatal(err)
	}
	wantChanges := []string{"bank.retain_mission", "mental_models/edited", "mental_models/added"}
	if !reflect.DeepEqual(plan.Changes, wantChanges) {
		t.Fatalf("changes = %v, want %v", plan.Changes, wantChanges)
	}
	if got := plan.Manifest["bank"]; !reflect.DeepEqual(got, map[string]any{"retain_mission": "new"}) {
		t.Fatalf("bank = %v", got)
	}
	var ids []string
	for _, m := range plan.Manifest["mental_models"].([]any) {
		ids = append(ids, m.(map[string]any)["id"].(string))
	}
	if !reflect.DeepEqual(ids, []string{"edited", "added"}) {
		t.Fatalf("mental model ids = %v", ids)
	}
	if plan.Manifest["version"] != "1" {
		t.Fatalf("version not carried: %v", plan.Manifest["version"])
	}
}

func TestPlanDetectsChangedDefaultField(t *testing.T) {
	desired := decode(t, `{"version":"1","directives":[{"name":"d","content":"c","is_active":false}]}`)
	current := decode(t, `{"version":"1","directives":[{"name":"d","content":"c","priority":0,"is_active":true,"tags":[]}]}`)

	plan, err := ComputePlan(desired, current, testDefaults)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.Changes, []string{"directives/d"}) {
		t.Fatalf("changes = %v", plan.Changes)
	}
}

// Import ignores null bank settings, so they must never be reported as drift.
func TestPlanIgnoresNullBankSettings(t *testing.T) {
	desired := decode(t, `{"version":"1","bank":{"retain_mission":null}}`)
	current := decode(t, `{"version":"1","bank":{"retain_mission":"set elsewhere"}}`)

	plan, err := ComputePlan(desired, current, testDefaults)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Empty() {
		t.Fatalf("expected no changes, got %v", plan.Changes)
	}
}

func TestPlanAgainstMissingBankListsEverything(t *testing.T) {
	desired := decode(t, `{"version":"1","bank":{"retain_mission":"m"},"mental_models":[{"id":"a","name":"A","source_query":"q"}]}`)

	plan, err := ComputePlan(desired, nil, testDefaults)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.Changes, []string{"bank.retain_mission", "mental_models/a"}) {
		t.Fatalf("changes = %v", plan.Changes)
	}
}

func TestPlanRejectsEntryWithoutKey(t *testing.T) {
	desired := decode(t, `{"version":"1","mental_models":[{"name":"no id","source_query":"q"}]}`)
	if _, err := ComputePlan(desired, nil, testDefaults); err == nil {
		t.Fatal("expected an error for a mental model without id")
	}
}

// The export serializes nested models with every default filled in, for
// example entity label groups and trigger tag groups. A template that omits
// those nested defaults must still count as in sync.
func TestPlanIgnoresNestedServerDefaults(t *testing.T) {
	desired := decode(t, `{"version":"1",
		"bank":{"entity_labels":[{"key":"sentiment","type":"value","values":[{"value":"positive"}]}]},
		"mental_models":[{"id":"m","name":"M","source_query":"q","trigger":{"tag_groups":[{"tags":["x"]}]}}]}`)
	current := decode(t, `{"version":"1",
		"bank":{"entity_labels":[{"key":"sentiment","type":"value","description":"","optional":true,"tag":false,"fields":{},
			"values":[{"value":"positive","description":""}]}]},
		"mental_models":[{"id":"m","name":"M","source_query":"q","tags":[],"max_tokens":2048,
			"trigger":{"mode":"full","refresh_after_consolidation":false,"tag_groups":[{"tags":["x"],"match":"any_strict","resolve":"exact"}]}}]}`)

	plan, err := ComputePlan(desired, current, testDefaults)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Empty() {
		t.Fatalf("expected no changes, got %v", plan.Changes)
	}
}

// Containment must not hide real changes inside nested values.
func TestPlanDetectsNestedChanges(t *testing.T) {
	current := decode(t, `{"version":"1","bank":{"entity_labels":[{"key":"sentiment","optional":true,"values":[{"value":"positive"},{"value":"negative"}]}]}}`)
	for name, desired := range map[string]string{
		"changed value": `{"version":"1","bank":{"entity_labels":[{"key":"sentiment","optional":false,"values":[{"value":"positive"},{"value":"negative"}]}]}}`,
		"removed item":  `{"version":"1","bank":{"entity_labels":[{"key":"sentiment","values":[{"value":"positive"}]}]}}`,
		"reordered":     `{"version":"1","bank":{"entity_labels":[{"key":"sentiment","values":[{"value":"negative"},{"value":"positive"}]}]}}`,
	} {
		plan, err := ComputePlan(decode(t, desired), current, testDefaults)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(plan.Changes, []string{"bank.entity_labels"}) {
			t.Errorf("%s: changes = %v", name, plan.Changes)
		}
	}
}
