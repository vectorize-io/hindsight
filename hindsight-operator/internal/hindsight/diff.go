package hindsight

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
)

// Plan is the part of a desired template that differs from the bank.
type Plan struct {
	// Manifest holds only the changed bank settings, mental models, and
	// directives. Importing it leaves everything else untouched.
	Manifest map[string]any
	// Changes names each changed item, for example "bank.retain_mission",
	// "mental_models/preferences", or "directives/Be concise".
	Changes []string
}

// Empty reports whether the bank already matches the desired template.
func (p *Plan) Empty() bool { return len(p.Changes) == 0 }

// Defaults holds the server's default values for mental model, trigger, and
// directive fields, read from GET /v1/bank-template-schema.
type Defaults struct {
	MentalModel map[string]any
	Trigger     map[string]any
	Directive   map[string]any
}

// ComputePlan compares a desired template with the bank's exported template.
//
// Import always regenerates every mental model it receives, even an unchanged
// one, so the operator must send only what changed. The comparison covers only
// what the desired template declares, which matches import semantics: omitted
// bank settings, mental models, and directives are left as they are.
// current is nil when the bank does not exist.
func ComputePlan(desired, current map[string]any, defaults Defaults) (*Plan, error) {
	plan := &Plan{Manifest: map[string]any{}}
	if version, ok := desired["version"]; ok {
		plan.Manifest["version"] = version
	}

	if raw, ok := desired["bank"]; ok && raw != nil {
		want, err := asObject(raw, "bank")
		if err != nil {
			return nil, err
		}
		have, _ := asObject(current["bank"], "bank")
		changed := map[string]any{}
		for _, key := range sortedKeys(want) {
			// Import ignores null settings, so they never count as drift.
			if want[key] == nil {
				continue
			}
			if !containsJSON(want[key], have[key]) {
				changed[key] = want[key]
				plan.Changes = append(plan.Changes, "bank."+key)
			}
		}
		if len(changed) > 0 {
			plan.Manifest["bank"] = changed
		}
	}

	models, err := diffList(desired, current, "mental_models", "id", func(item map[string]any) map[string]any {
		out := withDefaults(item, defaults.MentalModel)
		if trigger, ok := out["trigger"].(map[string]any); ok || out["trigger"] == nil {
			out["trigger"] = withDefaults(trigger, defaults.Trigger)
		}
		return out
	}, plan)
	if err != nil {
		return nil, err
	}
	if models != nil {
		plan.Manifest["mental_models"] = models
	}

	directives, err := diffList(desired, current, "directives", "name", func(item map[string]any) map[string]any {
		return withDefaults(item, defaults.Directive)
	}, plan)
	if err != nil {
		return nil, err
	}
	if directives != nil {
		plan.Manifest["directives"] = directives
	}
	return plan, nil
}

// diffList returns the desired items of section that are missing from, or
// differ from, the current template. Items match on key.
func diffList(desired, current map[string]any, section, key string, normalize func(map[string]any) map[string]any, plan *Plan) ([]any, error) {
	raw, ok := desired[section]
	if !ok || raw == nil {
		return nil, nil
	}
	want, err := asObjects(raw, section)
	if err != nil {
		return nil, err
	}
	haveList, err := asObjects(current[section], section)
	if err != nil {
		return nil, err
	}
	have := map[string]map[string]any{}
	for _, item := range haveList {
		if id, ok := item[key].(string); ok {
			have[id] = normalize(item)
		}
	}
	var changed []any
	for _, item := range want {
		id, ok := item[key].(string)
		if !ok || id == "" {
			return nil, fmt.Errorf("every entry in %s needs a string %q", section, key)
		}
		existing, found := have[id]
		if found && containsJSON(normalize(item), existing) {
			continue
		}
		changed = append(changed, item)
		plan.Changes = append(plan.Changes, section+"/"+id)
	}
	return changed, nil
}

// withDefaults returns a copy of item with absent fields set to the server
// defaults, so that "omitted" and "explicitly default" compare equal.
func withDefaults(item, defaults map[string]any) map[string]any {
	out := make(map[string]any, len(item)+len(defaults))
	for k, v := range defaults {
		out[k] = v
	}
	for k, v := range item {
		if v == nil {
			if _, hasDefault := defaults[k]; hasDefault {
				continue
			}
		}
		out[k] = v
	}
	return out
}

// containsJSON reports whether have matches every value that want declares.
//
// The export serializes nested models with all of their defaults, for example
// entity label groups gain "optional" and "tag", and tag groups gain "match".
// Those nested shapes are not described by the template schema (entity_labels
// is a free-form object), so the operator cannot fill their defaults the way it
// does for top-level mental model, trigger, and directive fields. Extra fields
// in have are therefore ignored, and null in want means "not declared", which
// matches import semantics. Lists must have the same length and match in order.
func containsJSON(want, have any) bool {
	return contains(canonical(want), canonical(have))
}

func contains(want, have any) bool {
	switch w := want.(type) {
	case nil:
		return true
	case map[string]any:
		h, ok := have.(map[string]any)
		if !ok {
			return false
		}
		for k, v := range w {
			if !contains(v, h[k]) {
				return false
			}
		}
		return true
	case []any:
		h, ok := have.([]any)
		if !ok || len(h) != len(w) {
			return false
		}
		for i := range w {
			if !contains(w[i], h[i]) {
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(want, have)
	}
}

// canonical round-trips a value through JSON so that numbers compare as
// float64 regardless of how they were decoded.
func canonical(v any) any {
	encoded, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(encoded, &out); err != nil {
		return v
	}
	return out
}

func asObject(v any, field string) (map[string]any, error) {
	if v == nil {
		return map[string]any{}, nil
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object", field)
	}
	return obj, nil
}

func asObjects(v any, field string) ([]map[string]any, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, fmt.Errorf("%s must be a list", field)
	}
	out := make([]map[string]any, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("every entry in %s must be an object", field)
		}
		out = append(out, obj)
	}
	return out, nil
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
