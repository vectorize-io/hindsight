# ConsolidationResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**OperationId** | **string** | ID of the async consolidation operation | 
**Deduplicated** | Pointer to **bool** | True if an existing pending task was reused | [optional] [default to false]

## Methods

### NewConsolidationResponse

`func NewConsolidationResponse(operationId string, ) *ConsolidationResponse`

NewConsolidationResponse instantiates a new ConsolidationResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewConsolidationResponseWithDefaults

`func NewConsolidationResponseWithDefaults() *ConsolidationResponse`

NewConsolidationResponseWithDefaults instantiates a new ConsolidationResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetOperationId

`func (o *ConsolidationResponse) GetOperationId() string`

GetOperationId returns the OperationId field if non-nil, zero value otherwise.

### GetOperationIdOk

`func (o *ConsolidationResponse) GetOperationIdOk() (*string, bool)`

GetOperationIdOk returns a tuple with the OperationId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOperationId

`func (o *ConsolidationResponse) SetOperationId(v string)`

SetOperationId sets OperationId field to given value.


### GetDeduplicated

`func (o *ConsolidationResponse) GetDeduplicated() bool`

GetDeduplicated returns the Deduplicated field if non-nil, zero value otherwise.

### GetDeduplicatedOk

`func (o *ConsolidationResponse) GetDeduplicatedOk() (*bool, bool)`

GetDeduplicatedOk returns a tuple with the Deduplicated field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDeduplicated

`func (o *ConsolidationResponse) SetDeduplicated(v bool)`

SetDeduplicated sets Deduplicated field to given value.

### HasDeduplicated

`func (o *ConsolidationResponse) HasDeduplicated() bool`

HasDeduplicated returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


