package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

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

	operationID := "550e8400-e29b-41d4-a716-446655440000"

	// [docs:operations-list]
	// List recent operations for a bank (default: 20 most recent).
	recent, _, err := client.OperationsAPI.ListOperations(ctx, "my-bank").Execute()
	if err != nil {
		log.Fatalf("list operations: %v", err)
	}
	for _, op := range recent.Operations {
		fmt.Println(op.Id, op.TaskType, op.Status)
	}

	// Filter by status and type.
	_, _, _ = client.OperationsAPI.ListOperations(ctx, "my-bank").
		Status("pending").
		Type_("graph_maintenance").
		Execute()

	// Hide retain_batch parent rows (show only individual child retain jobs).
	_, _, _ = client.OperationsAPI.ListOperations(ctx, "my-bank").
		ExcludeParents(true).
		Execute()
	// [/docs:operations-list]

	// [docs:operations-get]
	status, _, err := client.OperationsAPI.GetOperationStatus(ctx, "my-bank", operationID).Execute()
	if err != nil {
		log.Fatalf("get status: %v", err)
	}
	fmt.Println(status.Status, status.ErrorMessage)

	// Include the submission payload (can be large for retain batches).
	_, _, _ = client.OperationsAPI.GetOperationStatus(ctx, "my-bank", operationID).
		IncludePayload(true).
		Execute()
	// [/docs:operations-get]

	// [docs:operations-cancel]
	// Cancel a pending operation before a worker claims it.
	// Returns 409 if the operation is already processing/completed/failed.
	_, _, _ = client.OperationsAPI.CancelOperation(ctx, "my-bank", operationID).Execute()
	// [/docs:operations-cancel]

	// [docs:operations-retry]
	// Re-queue a failed (or cancelled) operation.
	// Returns 409 if the operation isn't in failed/cancelled state.
	_, _, _ = client.OperationsAPI.RetryOperation(ctx, "my-bank", operationID).Execute()
	// [/docs:operations-retry]

	// [docs:operations-async-retain]
	// Submit a large batch asynchronously — the call returns immediately with
	// an operation_id you can poll.
	async := true
	resp, _, err := client.MemoryAPI.RetainMemories(ctx, "my-bank").
		RetainRequest(hindsight.RetainRequest{
			Items: []hindsight.MemoryItem{
				{Content: "Alice joined Google in 2023"},
				{Content: "Bob prefers Python over JavaScript"},
			},
			Async: &async,
		}).Execute()
	if err != nil {
		log.Fatalf("retain: %v", err)
	}
	opID := resp.OperationId.Get()

	for {
		s, _, err := client.OperationsAPI.GetOperationStatus(ctx, "my-bank", *opID).Execute()
		if err != nil {
			log.Fatalf("poll: %v", err)
		}
		if s.Status == "completed" || s.Status == "failed" || s.Status == "cancelled" {
			fmt.Println("finished:", s.Status)
			break
		}
		time.Sleep(2 * time.Second)
	}
	// [/docs:operations-async-retain]
}
