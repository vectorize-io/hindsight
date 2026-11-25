package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
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
	sessionID := fmt.Sprintf("chat-%08x", rand.Uint32())

	// [docs:sessions-append]
	// Send the messages since the last append, in the OpenAI chat format your agent likely
	// keeps already, each with your own id: sending one again (a retry after a lost response)
	// stores nothing twice. The first append creates the session.
	opts := hindsight.NewSessionOptions()
	opts.SetLive(true)
	opts.SetRetain("turns")

	text := func(s string) hindsight.Content1 { return hindsight.Content1{String: &s} }
	ask := hindsight.NewSessionMessage("1", "user")
	ask.SetContent(text("Half cents round the wrong way."))
	args := `{"path": "pay/rounding.py"}`
	call := hindsight.NewSessionMessage("2", "assistant")
	call.SetToolCalls([]hindsight.ToolCall{*hindsight.NewToolCall("t1",
		*hindsight.NewToolCallFunction("edit_file", hindsight.Arguments{String: &args}))})
	result := hindsight.NewSessionMessage("3", "tool")
	result.SetToolCallId("t1")
	result.SetContent(text("updated"))

	req := hindsight.NewSessionAppendRequest()
	req.SetSession(*opts)
	req.SetMessages([]hindsight.SessionMessage{*ask, *call, *result})
	appended, _, err := client.SessionsAPI.AppendSessionEvents(ctx, "my-bank", sessionID).
		SessionAppendRequest(*req).Execute()
	if err != nil {
		log.Fatalf("append: %v", err)
	}
	fmt.Println(appended.Notes, appended.Checking)
	// [/docs:sessions-append]

	// [docs:sessions-notes]
	// Notes come back with every append. `ack` is the seq of the last note you delivered:
	// the response carries the notes after it, so a note is handed over until you confirm it.
	delivered := int32(0)
	poll := hindsight.NewSessionAppendRequest()
	poll.SetAck(delivered)
	polled, _, err := client.SessionsAPI.AppendSessionEvents(ctx, "my-bank", sessionID).
		SessionAppendRequest(*poll).Execute()
	if err != nil {
		log.Fatalf("poll: %v", err)
	}
	for _, note := range polled.Notes {
		fmt.Printf("#%d %s\n", note.Seq, note.Text)
		delivered = note.Seq
	}
	// [/docs:sessions-notes]

	// [docs:sessions-turn-end]
	// The end of a turn saves the conversation into the session's document
	// (`session:<id>` unless you named one), where memories are extracted from it.
	reply := hindsight.NewSessionMessage("4", "assistant")
	reply.SetContent(text("Fixed: half cents round down."))
	turn := hindsight.NewSessionAppendRequest()
	turn.SetMessages([]hindsight.SessionMessage{*reply})
	turn.SetTurnEnd(true)
	turn.SetAck(delivered)
	if _, _, err := client.SessionsAPI.AppendSessionEvents(ctx, "my-bank", sessionID).
		SessionAppendRequest(*turn).Execute(); err != nil {
		log.Fatalf("turn end: %v", err)
	}
	// [/docs:sessions-turn-end]

	// [docs:sessions-follow]
	// Follow the session's log: your events and Hindsight's checks, notes and saves, in one order.
	entries, _, err := client.SessionsAPI.ListSessionEvents(ctx, "my-bank", sessionID).After(0).Execute()
	if err != nil {
		log.Fatalf("events: %v", err)
	}
	for _, entry := range entries.Items {
		fmt.Println(entry.Seq, entry.Source, entry.Kind)
	}
	// poll again with After(<the last entry's Seq>) to get only what is new
	// [/docs:sessions-follow]

	// [docs:sessions-list]
	sessions, _, err := client.SessionsAPI.ListSessions(ctx, "my-bank").Limit(20).Execute()
	if err != nil {
		log.Fatalf("list: %v", err)
	}
	for _, s := range sessions.Items {
		fmt.Println(s.SessionId, s.GetLastEventAt(), s.NoteCount)
	}
	// [/docs:sessions-list]


	// [docs:sessions-delete]
	// Deletes the session and its log. The document it saved stays: that is memory now.
	if _, _, err := client.SessionsAPI.DeleteSession(ctx, "my-bank", sessionID).Execute(); err != nil {
		log.Fatalf("delete: %v", err)
	}
	// [/docs:sessions-delete]

	client.BanksAPI.DeleteBank(ctx, "my-bank").Execute()
}
