package hindsight

import (
	"encoding/json"
	"testing"
)

// An older client must parse values the server added after its release (#6862).

func TestUnknownEnumValueParses(t *testing.T) {
	var k DocumentEventKind
	if err := json.Unmarshal([]byte(`"a_kind_added_later"`), &k); err != nil {
		t.Fatal(err)
	}
	if k != "a_kind_added_later" || k.IsValid() {
		t.Fatalf("got %q valid=%v", k, k.IsValid())
	}
}

func TestUnknownUnionShapeIsKeptRaw(t *testing.T) {
	in := `{"tags_match":"any","exclude_mental_models":false,"tag_groups":[{"tags":["a"]},{"xor":[{"tags":["b"]}]}]}`
	var scope MentalModelRefreshScope
	if err := json.Unmarshal([]byte(in), &scope); err != nil {
		t.Fatal(err)
	}
	if scope.TagGroups[0].TagGroupLeaf == nil {
		t.Fatal("known shape not parsed")
	}
	if string(*scope.TagGroups[1].Raw) != `{"xor":[{"tags":["b"]}]}` {
		t.Fatalf("raw = %v", scope.TagGroups[1].Raw)
	}
	out, err := json.Marshal(scope.TagGroups[1])
	if err != nil || string(out) != `{"xor":[{"tags":["b"]}]}` {
		t.Fatalf("round trip = %s, %v", out, err)
	}
}

// Union structs stayed comparable with == when Raw was added; this stops compiling if not.
var _ = MentalModelRefreshScopeTagGroupsInner{} == MentalModelRefreshScopeTagGroupsInner{}
