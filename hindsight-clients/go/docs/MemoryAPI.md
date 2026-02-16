# \MemoryAPI

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**ClearBankMemories**](MemoryAPI.md#ClearBankMemories) | **Delete** /v1/default/banks/{bank_id}/memories | Clear memory bank memories
[**GetGraph**](MemoryAPI.md#GetGraph) | **Get** /v1/default/banks/{bank_id}/graph | Get memory graph data
[**GetMemory**](MemoryAPI.md#GetMemory) | **Get** /v1/default/banks/{bank_id}/memories/{memory_id} | Get memory unit
[**ListMemories**](MemoryAPI.md#ListMemories) | **Get** /v1/default/banks/{bank_id}/memories/list | List memory units
[**ListTags**](MemoryAPI.md#ListTags) | **Get** /v1/default/banks/{bank_id}/tags | List tags
[**RecallMemories**](MemoryAPI.md#RecallMemories) | **Post** /v1/default/banks/{bank_id}/memories/recall | Recall memory
[**Reflect**](MemoryAPI.md#Reflect) | **Post** /v1/default/banks/{bank_id}/reflect | Reflect and generate answer
[**RetainMemories**](MemoryAPI.md#RetainMemories) | **Post** /v1/default/banks/{bank_id}/memories | Retain memories



## ClearBankMemories

> DeleteResponse ClearBankMemories(ctx, bankId).Type_(type_).Authorization(authorization).Execute()

Clear memory bank memories



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
	type_ := "type__example" // string | Optional fact type filter (world, experience, opinion) (optional)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.ClearBankMemories(context.Background(), bankId).Type_(type_).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.ClearBankMemories``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ClearBankMemories`: DeleteResponse
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.ClearBankMemories`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiClearBankMemoriesRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **type_** | **string** | Optional fact type filter (world, experience, opinion) | 
 **authorization** | **string** |  | 

### Return type

[**DeleteResponse**](DeleteResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## GetGraph

> GraphDataResponse GetGraph(ctx, bankId).Type_(type_).Limit(limit).Authorization(authorization).Execute()

Get memory graph data



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
	type_ := "type__example" // string |  (optional)
	limit := int32(56) // int32 |  (optional) (default to 1000)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.GetGraph(context.Background(), bankId).Type_(type_).Limit(limit).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.GetGraph``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetGraph`: GraphDataResponse
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.GetGraph`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetGraphRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **type_** | **string** |  | 
 **limit** | **int32** |  | [default to 1000]
 **authorization** | **string** |  | 

### Return type

[**GraphDataResponse**](GraphDataResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## GetMemory

> interface{} GetMemory(ctx, bankId, memoryId).Authorization(authorization).Execute()

Get memory unit



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
	memoryId := "memoryId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.GetMemory(context.Background(), bankId, memoryId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.GetMemory``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetMemory`: interface{}
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.GetMemory`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**memoryId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetMemoryRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

**interface{}**

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ListMemories

> ListMemoryUnitsResponse ListMemories(ctx, bankId).Type_(type_).Q(q).Limit(limit).Offset(offset).Authorization(authorization).Execute()

List memory units



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
	type_ := "type__example" // string |  (optional)
	q := "q_example" // string |  (optional)
	limit := int32(56) // int32 |  (optional) (default to 100)
	offset := int32(56) // int32 |  (optional) (default to 0)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.ListMemories(context.Background(), bankId).Type_(type_).Q(q).Limit(limit).Offset(offset).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.ListMemories``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ListMemories`: ListMemoryUnitsResponse
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.ListMemories`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiListMemoriesRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **type_** | **string** |  | 
 **q** | **string** |  | 
 **limit** | **int32** |  | [default to 100]
 **offset** | **int32** |  | [default to 0]
 **authorization** | **string** |  | 

### Return type

[**ListMemoryUnitsResponse**](ListMemoryUnitsResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ListTags

> ListTagsResponse ListTags(ctx, bankId).Q(q).Limit(limit).Offset(offset).Authorization(authorization).Execute()

List tags



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
	q := "q_example" // string | Wildcard pattern to filter tags (e.g., 'user:*' for user:alice, '*-admin' for role-admin). Use '*' as wildcard. Case-insensitive. (optional)
	limit := int32(56) // int32 | Maximum number of tags to return (optional) (default to 100)
	offset := int32(56) // int32 | Offset for pagination (optional) (default to 0)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.ListTags(context.Background(), bankId).Q(q).Limit(limit).Offset(offset).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.ListTags``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ListTags`: ListTagsResponse
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.ListTags`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiListTagsRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **q** | **string** | Wildcard pattern to filter tags (e.g., &#39;user:*&#39; for user:alice, &#39;*-admin&#39; for role-admin). Use &#39;*&#39; as wildcard. Case-insensitive. | 
 **limit** | **int32** | Maximum number of tags to return | [default to 100]
 **offset** | **int32** | Offset for pagination | [default to 0]
 **authorization** | **string** |  | 

### Return type

[**ListTagsResponse**](ListTagsResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## RecallMemories

> RecallResponse RecallMemories(ctx, bankId).RecallRequest(recallRequest).Authorization(authorization).Execute()

Recall memory



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
	recallRequest := *openapiclient.NewRecallRequest("Query_example") // RecallRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.RecallMemories(context.Background(), bankId).RecallRequest(recallRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.RecallMemories``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `RecallMemories`: RecallResponse
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.RecallMemories`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiRecallMemoriesRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **recallRequest** | [**RecallRequest**](RecallRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**RecallResponse**](RecallResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## Reflect

> ReflectResponse Reflect(ctx, bankId).ReflectRequest(reflectRequest).Authorization(authorization).Execute()

Reflect and generate answer



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
	reflectRequest := *openapiclient.NewReflectRequest("Query_example") // ReflectRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.Reflect(context.Background(), bankId).ReflectRequest(reflectRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.Reflect``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `Reflect`: ReflectResponse
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.Reflect`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiReflectRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **reflectRequest** | [**ReflectRequest**](ReflectRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**ReflectResponse**](ReflectResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## RetainMemories

> RetainResponse RetainMemories(ctx, bankId).RetainRequest(retainRequest).Authorization(authorization).Execute()

Retain memories



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
	retainRequest := *openapiclient.NewRetainRequest([]openapiclient.MemoryItem{*openapiclient.NewMemoryItem("Content_example")}) // RetainRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MemoryAPI.RetainMemories(context.Background(), bankId).RetainRequest(retainRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MemoryAPI.RetainMemories``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `RetainMemories`: RetainResponse
	fmt.Fprintf(os.Stdout, "Response from `MemoryAPI.RetainMemories`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiRetainMemoriesRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **retainRequest** | [**RetainRequest**](RetainRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**RetainResponse**](RetainResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)

