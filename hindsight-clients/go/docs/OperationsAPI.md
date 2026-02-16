# \OperationsAPI

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**CancelOperation**](OperationsAPI.md#CancelOperation) | **Delete** /v1/default/banks/{bank_id}/operations/{operation_id} | Cancel a pending async operation
[**GetOperationStatus**](OperationsAPI.md#GetOperationStatus) | **Get** /v1/default/banks/{bank_id}/operations/{operation_id} | Get operation status
[**ListOperations**](OperationsAPI.md#ListOperations) | **Get** /v1/default/banks/{bank_id}/operations | List async operations



## CancelOperation

> CancelOperationResponse CancelOperation(ctx, bankId, operationId).Authorization(authorization).Execute()

Cancel a pending async operation



### Example

```go
package main

import (
	"context"
	"fmt"
	"os"
	openapiclient "github.com/vectorize-io/hindsight-client-go"
)

func main() {
	bankId := "bankId_example" // string | 
	operationId := "operationId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.OperationsAPI.CancelOperation(context.Background(), bankId, operationId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `OperationsAPI.CancelOperation``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `CancelOperation`: CancelOperationResponse
	fmt.Fprintf(os.Stdout, "Response from `OperationsAPI.CancelOperation`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**operationId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiCancelOperationRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

[**CancelOperationResponse**](CancelOperationResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## GetOperationStatus

> OperationStatusResponse GetOperationStatus(ctx, bankId, operationId).Authorization(authorization).Execute()

Get operation status



### Example

```go
package main

import (
	"context"
	"fmt"
	"os"
	openapiclient "github.com/vectorize-io/hindsight-client-go"
)

func main() {
	bankId := "bankId_example" // string | 
	operationId := "operationId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.OperationsAPI.GetOperationStatus(context.Background(), bankId, operationId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `OperationsAPI.GetOperationStatus``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetOperationStatus`: OperationStatusResponse
	fmt.Fprintf(os.Stdout, "Response from `OperationsAPI.GetOperationStatus`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**operationId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetOperationStatusRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

[**OperationStatusResponse**](OperationStatusResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ListOperations

> OperationsListResponse ListOperations(ctx, bankId).Status(status).Limit(limit).Offset(offset).Authorization(authorization).Execute()

List async operations



### Example

```go
package main

import (
	"context"
	"fmt"
	"os"
	openapiclient "github.com/vectorize-io/hindsight-client-go"
)

func main() {
	bankId := "bankId_example" // string | 
	status := "status_example" // string | Filter by status: pending, completed, or failed (optional)
	limit := int32(56) // int32 | Maximum number of operations to return (optional) (default to 20)
	offset := int32(56) // int32 | Number of operations to skip (optional) (default to 0)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.OperationsAPI.ListOperations(context.Background(), bankId).Status(status).Limit(limit).Offset(offset).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `OperationsAPI.ListOperations``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ListOperations`: OperationsListResponse
	fmt.Fprintf(os.Stdout, "Response from `OperationsAPI.ListOperations`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiListOperationsRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **status** | **string** | Filter by status: pending, completed, or failed | 
 **limit** | **int32** | Maximum number of operations to return | [default to 20]
 **offset** | **int32** | Number of operations to skip | [default to 0]
 **authorization** | **string** |  | 

### Return type

[**OperationsListResponse**](OperationsListResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)

