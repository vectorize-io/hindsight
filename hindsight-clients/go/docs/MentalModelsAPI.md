# \MentalModelsAPI

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**CreateMentalModel**](MentalModelsAPI.md#CreateMentalModel) | **Post** /v1/default/banks/{bank_id}/mental-models | Create mental model
[**DeleteMentalModel**](MentalModelsAPI.md#DeleteMentalModel) | **Delete** /v1/default/banks/{bank_id}/mental-models/{mental_model_id} | Delete mental model
[**GetMentalModel**](MentalModelsAPI.md#GetMentalModel) | **Get** /v1/default/banks/{bank_id}/mental-models/{mental_model_id} | Get mental model
[**ListMentalModels**](MentalModelsAPI.md#ListMentalModels) | **Get** /v1/default/banks/{bank_id}/mental-models | List mental models
[**RefreshMentalModel**](MentalModelsAPI.md#RefreshMentalModel) | **Post** /v1/default/banks/{bank_id}/mental-models/{mental_model_id}/refresh | Refresh mental model
[**UpdateMentalModel**](MentalModelsAPI.md#UpdateMentalModel) | **Patch** /v1/default/banks/{bank_id}/mental-models/{mental_model_id} | Update mental model



## CreateMentalModel

> CreateMentalModelResponse CreateMentalModel(ctx, bankId).CreateMentalModelRequest(createMentalModelRequest).Authorization(authorization).Execute()

Create mental model



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
	createMentalModelRequest := *openapiclient.NewCreateMentalModelRequest("Name_example", "SourceQuery_example") // CreateMentalModelRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MentalModelsAPI.CreateMentalModel(context.Background(), bankId).CreateMentalModelRequest(createMentalModelRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MentalModelsAPI.CreateMentalModel``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `CreateMentalModel`: CreateMentalModelResponse
	fmt.Fprintf(os.Stdout, "Response from `MentalModelsAPI.CreateMentalModel`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiCreateMentalModelRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **createMentalModelRequest** | [**CreateMentalModelRequest**](CreateMentalModelRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**CreateMentalModelResponse**](CreateMentalModelResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## DeleteMentalModel

> interface{} DeleteMentalModel(ctx, bankId, mentalModelId).Authorization(authorization).Execute()

Delete mental model



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
	mentalModelId := "mentalModelId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MentalModelsAPI.DeleteMentalModel(context.Background(), bankId, mentalModelId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MentalModelsAPI.DeleteMentalModel``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `DeleteMentalModel`: interface{}
	fmt.Fprintf(os.Stdout, "Response from `MentalModelsAPI.DeleteMentalModel`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**mentalModelId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiDeleteMentalModelRequest struct via the builder pattern


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


## GetMentalModel

> MentalModelResponse GetMentalModel(ctx, bankId, mentalModelId).Authorization(authorization).Execute()

Get mental model



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
	mentalModelId := "mentalModelId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MentalModelsAPI.GetMentalModel(context.Background(), bankId, mentalModelId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MentalModelsAPI.GetMentalModel``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetMentalModel`: MentalModelResponse
	fmt.Fprintf(os.Stdout, "Response from `MentalModelsAPI.GetMentalModel`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**mentalModelId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetMentalModelRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

[**MentalModelResponse**](MentalModelResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ListMentalModels

> MentalModelListResponse ListMentalModels(ctx, bankId).Tags(tags).TagsMatch(tagsMatch).Limit(limit).Offset(offset).Authorization(authorization).Execute()

List mental models



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
	tags := []string{"Inner_example"} // []string | Filter by tags (optional)
	tagsMatch := "tagsMatch_example" // string | How to match tags (optional) (default to "any")
	limit := int32(56) // int32 |  (optional) (default to 100)
	offset := int32(56) // int32 |  (optional) (default to 0)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MentalModelsAPI.ListMentalModels(context.Background(), bankId).Tags(tags).TagsMatch(tagsMatch).Limit(limit).Offset(offset).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MentalModelsAPI.ListMentalModels``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ListMentalModels`: MentalModelListResponse
	fmt.Fprintf(os.Stdout, "Response from `MentalModelsAPI.ListMentalModels`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiListMentalModelsRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **tags** | **[]string** | Filter by tags | 
 **tagsMatch** | **string** | How to match tags | [default to &quot;any&quot;]
 **limit** | **int32** |  | [default to 100]
 **offset** | **int32** |  | [default to 0]
 **authorization** | **string** |  | 

### Return type

[**MentalModelListResponse**](MentalModelListResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## RefreshMentalModel

> AsyncOperationSubmitResponse RefreshMentalModel(ctx, bankId, mentalModelId).Authorization(authorization).Execute()

Refresh mental model



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
	mentalModelId := "mentalModelId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MentalModelsAPI.RefreshMentalModel(context.Background(), bankId, mentalModelId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MentalModelsAPI.RefreshMentalModel``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `RefreshMentalModel`: AsyncOperationSubmitResponse
	fmt.Fprintf(os.Stdout, "Response from `MentalModelsAPI.RefreshMentalModel`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**mentalModelId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiRefreshMentalModelRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

[**AsyncOperationSubmitResponse**](AsyncOperationSubmitResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## UpdateMentalModel

> MentalModelResponse UpdateMentalModel(ctx, bankId, mentalModelId).UpdateMentalModelRequest(updateMentalModelRequest).Authorization(authorization).Execute()

Update mental model



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
	mentalModelId := "mentalModelId_example" // string | 
	updateMentalModelRequest := *openapiclient.NewUpdateMentalModelRequest() // UpdateMentalModelRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.MentalModelsAPI.UpdateMentalModel(context.Background(), bankId, mentalModelId).UpdateMentalModelRequest(updateMentalModelRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `MentalModelsAPI.UpdateMentalModel``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `UpdateMentalModel`: MentalModelResponse
	fmt.Fprintf(os.Stdout, "Response from `MentalModelsAPI.UpdateMentalModel`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**mentalModelId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiUpdateMentalModelRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **updateMentalModelRequest** | [**UpdateMentalModelRequest**](UpdateMentalModelRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**MentalModelResponse**](MentalModelResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)

