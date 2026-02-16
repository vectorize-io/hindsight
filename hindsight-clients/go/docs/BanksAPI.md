# \BanksAPI

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**AddBankBackground**](BanksAPI.md#AddBankBackground) | **Post** /v1/default/banks/{bank_id}/background | Add/merge memory bank background (deprecated)
[**ClearObservations**](BanksAPI.md#ClearObservations) | **Delete** /v1/default/banks/{bank_id}/observations | Clear all observations
[**CreateOrUpdateBank**](BanksAPI.md#CreateOrUpdateBank) | **Put** /v1/default/banks/{bank_id} | Create or update memory bank
[**DeleteBank**](BanksAPI.md#DeleteBank) | **Delete** /v1/default/banks/{bank_id} | Delete memory bank
[**GetAgentStats**](BanksAPI.md#GetAgentStats) | **Get** /v1/default/banks/{bank_id}/stats | Get statistics for memory bank
[**GetBankConfig**](BanksAPI.md#GetBankConfig) | **Get** /v1/default/banks/{bank_id}/config | Get bank configuration
[**GetBankProfile**](BanksAPI.md#GetBankProfile) | **Get** /v1/default/banks/{bank_id}/profile | Get memory bank profile
[**ListBanks**](BanksAPI.md#ListBanks) | **Get** /v1/default/banks | List all memory banks
[**ResetBankConfig**](BanksAPI.md#ResetBankConfig) | **Delete** /v1/default/banks/{bank_id}/config | Reset bank configuration
[**TriggerConsolidation**](BanksAPI.md#TriggerConsolidation) | **Post** /v1/default/banks/{bank_id}/consolidate | Trigger consolidation
[**UpdateBank**](BanksAPI.md#UpdateBank) | **Patch** /v1/default/banks/{bank_id} | Partial update memory bank
[**UpdateBankConfig**](BanksAPI.md#UpdateBankConfig) | **Patch** /v1/default/banks/{bank_id}/config | Update bank configuration
[**UpdateBankDisposition**](BanksAPI.md#UpdateBankDisposition) | **Put** /v1/default/banks/{bank_id}/profile | Update memory bank disposition



## AddBankBackground

> BackgroundResponse AddBankBackground(ctx, bankId).AddBackgroundRequest(addBackgroundRequest).Authorization(authorization).Execute()

Add/merge memory bank background (deprecated)



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
	addBackgroundRequest := *openapiclient.NewAddBackgroundRequest("Content_example") // AddBackgroundRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.AddBankBackground(context.Background(), bankId).AddBackgroundRequest(addBackgroundRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.AddBankBackground``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `AddBankBackground`: BackgroundResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.AddBankBackground`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiAddBankBackgroundRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **addBackgroundRequest** | [**AddBackgroundRequest**](AddBackgroundRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**BackgroundResponse**](BackgroundResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ClearObservations

> DeleteResponse ClearObservations(ctx, bankId).Authorization(authorization).Execute()

Clear all observations



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.ClearObservations(context.Background(), bankId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.ClearObservations``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ClearObservations`: DeleteResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.ClearObservations`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiClearObservationsRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

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


## CreateOrUpdateBank

> BankProfileResponse CreateOrUpdateBank(ctx, bankId).CreateBankRequest(createBankRequest).Authorization(authorization).Execute()

Create or update memory bank



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
	createBankRequest := *openapiclient.NewCreateBankRequest() // CreateBankRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.CreateOrUpdateBank(context.Background(), bankId).CreateBankRequest(createBankRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.CreateOrUpdateBank``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `CreateOrUpdateBank`: BankProfileResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.CreateOrUpdateBank`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiCreateOrUpdateBankRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **createBankRequest** | [**CreateBankRequest**](CreateBankRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**BankProfileResponse**](BankProfileResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## DeleteBank

> DeleteResponse DeleteBank(ctx, bankId).Authorization(authorization).Execute()

Delete memory bank



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.DeleteBank(context.Background(), bankId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.DeleteBank``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `DeleteBank`: DeleteResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.DeleteBank`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiDeleteBankRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

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


## GetAgentStats

> BankStatsResponse GetAgentStats(ctx, bankId).Authorization(authorization).Execute()

Get statistics for memory bank



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.GetAgentStats(context.Background(), bankId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.GetAgentStats``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetAgentStats`: BankStatsResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.GetAgentStats`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetAgentStatsRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **authorization** | **string** |  | 

### Return type

[**BankStatsResponse**](BankStatsResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## GetBankConfig

> BankConfigResponse GetBankConfig(ctx, bankId).Authorization(authorization).Execute()

Get bank configuration



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.GetBankConfig(context.Background(), bankId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.GetBankConfig``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetBankConfig`: BankConfigResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.GetBankConfig`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetBankConfigRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **authorization** | **string** |  | 

### Return type

[**BankConfigResponse**](BankConfigResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## GetBankProfile

> BankProfileResponse GetBankProfile(ctx, bankId).Authorization(authorization).Execute()

Get memory bank profile



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.GetBankProfile(context.Background(), bankId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.GetBankProfile``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `GetBankProfile`: BankProfileResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.GetBankProfile`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiGetBankProfileRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **authorization** | **string** |  | 

### Return type

[**BankProfileResponse**](BankProfileResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ListBanks

> BankListResponse ListBanks(ctx).Authorization(authorization).Execute()

List all memory banks



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.ListBanks(context.Background()).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.ListBanks``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ListBanks`: BankListResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.ListBanks`: %v\n", resp)
}
```

### Path Parameters



### Other Parameters

Other parameters are passed through a pointer to a apiListBanksRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **authorization** | **string** |  | 

### Return type

[**BankListResponse**](BankListResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## ResetBankConfig

> BankConfigResponse ResetBankConfig(ctx, bankId).Authorization(authorization).Execute()

Reset bank configuration



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.ResetBankConfig(context.Background(), bankId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.ResetBankConfig``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `ResetBankConfig`: BankConfigResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.ResetBankConfig`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiResetBankConfigRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **authorization** | **string** |  | 

### Return type

[**BankConfigResponse**](BankConfigResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## TriggerConsolidation

> ConsolidationResponse TriggerConsolidation(ctx, bankId).Authorization(authorization).Execute()

Trigger consolidation



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
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.TriggerConsolidation(context.Background(), bankId).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.TriggerConsolidation``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `TriggerConsolidation`: ConsolidationResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.TriggerConsolidation`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiTriggerConsolidationRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **authorization** | **string** |  | 

### Return type

[**ConsolidationResponse**](ConsolidationResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## UpdateBank

> BankProfileResponse UpdateBank(ctx, bankId).CreateBankRequest(createBankRequest).Authorization(authorization).Execute()

Partial update memory bank



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
	createBankRequest := *openapiclient.NewCreateBankRequest() // CreateBankRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.UpdateBank(context.Background(), bankId).CreateBankRequest(createBankRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.UpdateBank``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `UpdateBank`: BankProfileResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.UpdateBank`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiUpdateBankRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **createBankRequest** | [**CreateBankRequest**](CreateBankRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**BankProfileResponse**](BankProfileResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## UpdateBankConfig

> BankConfigResponse UpdateBankConfig(ctx, bankId).BankConfigUpdate(bankConfigUpdate).Authorization(authorization).Execute()

Update bank configuration



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
	bankConfigUpdate := *openapiclient.NewBankConfigUpdate(map[string]interface{}{"key": interface{}(123)}) // BankConfigUpdate | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.UpdateBankConfig(context.Background(), bankId).BankConfigUpdate(bankConfigUpdate).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.UpdateBankConfig``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `UpdateBankConfig`: BankConfigResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.UpdateBankConfig`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiUpdateBankConfigRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **bankConfigUpdate** | [**BankConfigUpdate**](BankConfigUpdate.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**BankConfigResponse**](BankConfigResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)


## UpdateBankDisposition

> BankProfileResponse UpdateBankDisposition(ctx, bankId).UpdateDispositionRequest(updateDispositionRequest).Authorization(authorization).Execute()

Update memory bank disposition



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
	updateDispositionRequest := *openapiclient.NewUpdateDispositionRequest(*openapiclient.NewDispositionTraits(int32(123), int32(123), int32(123))) // UpdateDispositionRequest | 
	authorization := "authorization_example" // string |  (optional)

	configuration := openapiclient.NewConfiguration()
	apiClient := openapiclient.NewAPIClient(configuration)
	resp, r, err := apiClient.BanksAPI.UpdateBankDisposition(context.Background(), bankId).UpdateDispositionRequest(updateDispositionRequest).Authorization(authorization).Execute()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error when calling `BanksAPI.UpdateBankDisposition``: %v\n", err)
		fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	}
	// response from `UpdateBankDisposition`: BankProfileResponse
	fmt.Fprintf(os.Stdout, "Response from `BanksAPI.UpdateBankDisposition`: %v\n", resp)
}
```

### Path Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
**ctx** | **context.Context** | context for authentication, logging, cancellation, deadlines, tracing, etc.
**bankId** | **string** |  | 

### Other Parameters

Other parameters are passed through a pointer to a apiUpdateBankDispositionRequest struct via the builder pattern


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------

 **updateDispositionRequest** | [**UpdateDispositionRequest**](UpdateDispositionRequest.md) |  | 
 **authorization** | **string** |  | 

### Return type

[**BankProfileResponse**](BankProfileResponse.md)

### Authorization

No authorization required

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints)
[[Back to Model list]](../README.md#documentation-for-models)
[[Back to README]](../README.md)

