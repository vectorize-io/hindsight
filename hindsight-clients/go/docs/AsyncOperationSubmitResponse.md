# AsyncOperationSubmitResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**OperationId** | **string** |  | 
**Status** | **string** |  | 

## Methods

### NewAsyncOperationSubmitResponse

`func NewAsyncOperationSubmitResponse(operationId string, status string, ) *AsyncOperationSubmitResponse`

NewAsyncOperationSubmitResponse instantiates a new AsyncOperationSubmitResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewAsyncOperationSubmitResponseWithDefaults

`func NewAsyncOperationSubmitResponseWithDefaults() *AsyncOperationSubmitResponse`

NewAsyncOperationSubmitResponseWithDefaults instantiates a new AsyncOperationSubmitResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetOperationId

`func (o *AsyncOperationSubmitResponse) GetOperationId() string`

GetOperationId returns the OperationId field if non-nil, zero value otherwise.

### GetOperationIdOk

`func (o *AsyncOperationSubmitResponse) GetOperationIdOk() (*string, bool)`

GetOperationIdOk returns a tuple with the OperationId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOperationId

`func (o *AsyncOperationSubmitResponse) SetOperationId(v string)`

SetOperationId sets OperationId field to given value.


### GetStatus

`func (o *AsyncOperationSubmitResponse) GetStatus() string`

GetStatus returns the Status field if non-nil, zero value otherwise.

### GetStatusOk

`func (o *AsyncOperationSubmitResponse) GetStatusOk() (*string, bool)`

GetStatusOk returns a tuple with the Status field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetStatus

`func (o *AsyncOperationSubmitResponse) SetStatus(v string)`

SetStatus sets Status field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


