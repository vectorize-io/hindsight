package hindsight

import (
	"encoding/json"
	"testing"
)

// A union embedded by value must still serialise as its member, not as the
// wrapper struct.
//
// `MemoryItem.Content` is an anyOf, and the generator gives such models a
// POINTER-receiver MarshalJSON. encoding/json only reaches a pointer-receiver
// marshaller through an addressable value, and the top level of
// json.Marshal(v) is not addressable — so the custom marshaller was skipped and
// the field went out as `{"ArrayOfContentAnyOfInner":null,"String":"hi"}`,
// which the API rejects with 422. It compiled, and it type-checked; only the
// wire bytes were wrong, which is why this asserts them directly.
//
// generate-clients.sh rewrites these receivers after generation. This test is
// what fails if that step is ever dropped.
func TestContentMarshalsAsItsMember(t *testing.T) {
	body, err := json.Marshal(RetainRequest{Items: []MemoryItem{{Content: TextContent("the sky is blue")}}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	const want = `{"items":[{"content":"the sky is blue"}]}`
	if string(body) != want {
		t.Errorf("content did not serialise as a bare string\n got: %s\nwant: %s", body, want)
	}
}

func TestContentBlocksMarshalAsAnArray(t *testing.T) {
	text := "click the button shown:"
	blocks := []ContentAnyOfInner{{TextContentBlock: &TextContentBlock{Type: "text", Text: text}}}
	body, err := json.Marshal(RetainRequest{Items: []MemoryItem{{
		Content: Content{ArrayOfContentAnyOfInner: &blocks},
	}}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	const want = `{"items":[{"content":[{"text":"click the button shown:","type":"text"}]}]}`
	if string(body) != want {
		t.Errorf("blocks did not serialise as an array\n got: %s\nwant: %s", body, want)
	}
}
