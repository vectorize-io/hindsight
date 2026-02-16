# \DirectivesAPI

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**CreateDirective**](DirectivesAPI.md#CreateDirective) | **Post** /v1/default/banks/{bank_id}/directives | Create directive
[**DeleteDirective**](DirectivesAPI.md#DeleteDirective) | **Delete** /v1/default/banks/{bank_id}/directives/{directive_id} | Delete directive
[**GetDirective**](DirectivesAPI.md#GetDirective) | **Get** /v1/default/banks/{bank_id}/directives/{directive_id} | Get directive
[**ListDirectives**](DirectivesAPI.md#ListDirectives) | **Get** /v1/default/banks/{bank_id}/directives | List directives
[**UpdateDirective**](DirectivesAPI.md#UpdateDirective) | **Patch** /v1/default/banks/{bank_id}/directives/{directive_id} | Update directive



## CreateDirective

> DirectiveResponse CreateDirective(ctx, bankId).CreateDirectiveRequest(createDirectiveRequest).Authorization(authorization).Execute()

Create directive



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
	createDirectiveRequest := *openapiclient.NewCreateDirectiveRequest("Name_example", "Content_example") // CreateDirectiveRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.DirectivesAPI.CreateDirective(context.Background(), bankId).CreateDirectiveRequest(createDirectiveRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `DirectivesAPI.CreateDirective``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `CreateDirective`: DirectiveResponse
	fmt.Fprintf(os.Stdout, "Response from `DirectivesAPI.CreateDirective`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiCreateDirectiveRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **createDirectiveRequest** | [**CreateDirectiveRequest**](CreateDirectiveRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**DirectiveResponse**](DirectiveResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## DeleteDirective

> interface{} DeleteDirective(ctx, bankId, directiveId).Authorization(authorization).Execute()

Delete directive



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
	directiveId := "directiveId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.DirectivesAPI.DeleteDirective(context.Background(), bankId, directiveId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `DirectivesAPI.DeleteDirective``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `DeleteDirective`: interface{}
	fmt.Fprintf(os.Stdout, "Response from `DirectivesAPI.DeleteDirective`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**directiveId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiDeleteDirectiveRequest struct via the builder pattern


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


## GetDirective

> DirectiveResponse GetDirective(ctx, bankId, directiveId).Authorization(authorization).Execute()

Get directive



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
	directiveId := "directiveId_example" // string | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.DirectivesAPI.GetDirective(context.Background(), bankId, directiveId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `DirectivesAPI.GetDirective``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetDirective`: DirectiveResponse
	fmt.Fprintf(os.Stdout, "Response from `DirectivesAPI.GetDirective`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**directiveId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetDirectiveRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **authorization** | **string** |  | 

### Return type

[**DirectiveResponse**](DirectiveResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ListDirectives

> DirectiveListResponse ListDirectives(ctx, bankId).Tags(tags).TagsMatch(tagsMatch).ActiveOnly(activeOnly).Limit(limit).Offset(offset).Authorization(authorization).Execute()

List directives



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
	activeOnly := true // bool | Only return active directives (optional) (default to true)
	limit := int32(56) // int32 |  (optional) (default to 100)
	offset := int32(56) // int32 |  (optional) (default to 0)
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.DirectivesAPI.ListDirectives(context.Background(), bankId).Tags(tags).TagsMatch(tagsMatch).ActiveOnly(activeOnly).Limit(limit).Offset(offset).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `DirectivesAPI.ListDirectives``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ListDirectives`: DirectiveListResponse
	fmt.Fprintf(os.Stdout, "Response from `DirectivesAPI.ListDirectives`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiListDirectivesRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **tags** | **[]string** | Filter by tags | 
 **tagsMatch** | **string** | How to match tags | [default to &quot;any&quot;]
 **activeOnly** | **bool** | Only return active directives | [default to true]
 **limit** | **int32** |  | [default to 100]
 **offset** | **int32** |  | [default to 0]
 **authorization** | **string** |  | 

### Return type

[**DirectiveListResponse**](DirectiveListResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## UpdateDirective

> DirectiveResponse UpdateDirective(ctx, bankId, directiveId).UpdateDirectiveRequest(updateDirectiveRequest).Authorization(authorization).Execute()

Update directive



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
	directiveId := "directiveId_example" // string | 
	updateDirectiveRequest := *openapiclient.NewUpdateDirectiveRequest() // UpdateDirectiveRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.DirectivesAPI.UpdateDirective(context.Background(), bankId, directiveId).UpdateDirectiveRequest(updateDirectiveRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `DirectivesAPI.UpdateDirective``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `UpdateDirective`: DirectiveResponse
	fmt.Fprintf(os.Stdout, "Response from `DirectivesAPI.UpdateDirective`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 
**directiveId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiUpdateDirectiveRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------


 **updateDirectiveRequest** | [**UpdateDirectiveRequest**](UpdateDirectiveRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**DirectiveResponse**](DirectiveResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)

