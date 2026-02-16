# \EntitiesAPI

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**GetEntity**](EntitiesAPI.md#GetEntity) | **Get** /v1/default/banks/{bank_id}/entities/{entity_id} | Get entity details
[**ListEntities**](EntitiesAPI.md#ListEntities) | **Get** /v1/default/banks/{bank_id}/entities | List entities
[**RegenerateEntityObservations**](EntitiesAPI.md#RegenerateEntityObservations) | **Post** /v1/default/banks/{bank_id}/entities/{entity_id}/regenerate | Regenerate entity observations (deprecated)



## GetEntity

> EntityDetailResponse GetEntity(ctx, bankId, entityId).Authorization(authorization).Execute()

Get entity details



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
	entityId := "entityId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.EntitiesAPI.GetEntity(context.Background(), bankId, entityId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `EntitiesAPI.GetEntity``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetEntity`: EntityDetailResponse
	fmt.Fprintf(os.Stdout, "Response from `EntitiesAPI.GetEntity`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**entityId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetEntityRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

[**EntityDetailResponse**](EntityDetailResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ListEntities

> EntityListResponse ListEntities(ctx, bankId).Limit(limit).Offset(offset).Authorization(authorization).Execute()

List entities



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
	limit := int32(56) // int32 | Maximum number of entities to return (optional) (default to 100)
	offset := int32(56) // int32 | Offset for pagination (optional) (default to 0)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.EntitiesAPI.ListEntities(context.Background(), bankId).Limit(limit).Offset(offset).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `EntitiesAPI.ListEntities``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ListEntities`: EntityListResponse
	fmt.Fprintf(os.Stdout, "Response from `EntitiesAPI.ListEntities`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiListEntitiesRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **limit** | **int32** | Maximum number of entities to return | [default to 100]
 **offset** | **int32** | Offset for pagination | [default to 0]
 **authorization** | **string** |  | 

### Return type

[**EntityListResponse**](EntityListResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## RegenerateEntityObservations

> EntityDetailResponse RegenerateEntityObservations(ctx, bankId, entityId).Authorization(authorization).Execute()

Regenerate entity observations (deprecated)



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
	entityId := "entityId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.EntitiesAPI.RegenerateEntityObservations(context.Background(), bankId, entityId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `EntitiesAPI.RegenerateEntityObservations``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `RegenerateEntityObservations`: EntityDetailResponse
	fmt.Fprintf(os.Stdout, "Response from `EntitiesAPI.RegenerateEntityObservations`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**entityId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiRegenerateEntityObservationsRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

[**EntityDetailResponse**](EntityDetailResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)

