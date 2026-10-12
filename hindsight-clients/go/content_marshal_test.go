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

// A field the server adds to a response must not break an older client.
//
// The generator emits decoder.DisallowUnknownFields() in every model's
// UnmarshalJSON, so a newly added response field failed decoding with
// `json: unknown field "..."`. generate-clients.sh strips that call; this test
// fails if that step is ever dropped.
func TestResponseIgnoresUnknownFields(t *testing.T) {
	var got DirectiveListResponse
	data := []byte(`{"items":[],"total":0,"limit":20,"offset":0,"added_later":true}`)
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unknown response field rejected: %v", err)
	}
	if got.Limit != 20 {
		t.Errorf("limit = %d, want 20", got.Limit)
	}
}
