# RetainResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Success** | **bool** |  | 
**BankId** | **string** |  | 
**ItemsCount** | **int32** |  | 
**Async** | **bool** | Whether the operation was processed asynchronously | 
**OperationId** | Pointer to **NullableString** |  | [optional] 
**Usage** | Pointer to [**NullableTokenUsage**](TokenUsage.md) |  | [optional] 

## Methods

### NewRetainResponse

`func NewRetainResponse(success bool, bankId string, itemsCount int32, async bool, ) *RetainResponse`

NewRetainResponse instantiates a new RetainResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewRetainResponseWithDefaults

`func NewRetainResponseWithDefaults() *RetainResponse`

NewRetainResponseWithDefaults instantiates a new RetainResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetSuccess

`func (o *RetainResponse) GetSuccess() bool`

GetSuccess returns the Success field if non-nil, zero value otherwise.

### GetSuccessOk

`func (o *RetainResponse) GetSuccessOk() (*bool, bool)`

GetSuccessOk returns a tuple with the Success field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetSuccess

`func (o *RetainResponse) SetSuccess(v bool)`

SetSuccess sets Success field to given value.


### GetBankId

`func (o *RetainResponse) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *RetainResponse) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *RetainResponse) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetItemsCount

`func (o *RetainResponse) GetItemsCount() int32`

GetItemsCount returns the ItemsCount field if non-nil, zero value otherwise.

### GetItemsCountOk

`func (o *RetainResponse) GetItemsCountOk() (*int32, bool)`

GetItemsCountOk returns a tuple with the ItemsCount field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetItemsCount

`func (o *RetainResponse) SetItemsCount(v int32)`

SetItemsCount sets ItemsCount field to given value.


### GetAsync

`func (o *RetainResponse) GetAsync() bool`

GetAsync returns the Async field if non-nil, zero value otherwise.

### GetAsyncOk

`func (o *RetainResponse) GetAsyncOk() (*bool, bool)`

GetAsyncOk returns a tuple with the Async field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetAsync

`func (o *RetainResponse) SetAsync(v bool)`

SetAsync sets Async field to given value.


### GetOperationId

`func (o *RetainResponse) GetOperationId() string`

GetOperationId returns the OperationId field if non-nil, zero value otherwise.

### GetOperationIdOk

`func (o *RetainResponse) GetOperationIdOk() (*string, bool)`

GetOperationIdOk returns a tuple with the OperationId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOperationId

`func (o *RetainResponse) SetOperationId(v string)`

SetOperationId sets OperationId field to given value.

### HasOperationId

`func (o *RetainResponse) HasOperationId() bool`

HasOperationId returns a boolean if a field has been set.

### SetOperationIdNil

`func (o *RetainResponse) SetOperationIdNil(b bool)`

 SetOperationIdNil sets the value for OperationId to be an explicit nil

### UnsetOperationId
`func (o *RetainResponse) UnsetOperationId()`

UnsetOperationId ensures that no value is present for OperationId, not even an explicit nil
### GetUsage

`func (o *RetainResponse) GetUsage() TokenUsage`

GetUsage returns the Usage field if non-nil, zero value otherwise.

### GetUsageOk

`func (o *RetainResponse) GetUsageOk() (*TokenUsage, bool)`

GetUsageOk returns a tuple with the Usage field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetUsage

`func (o *RetainResponse) SetUsage(v TokenUsage)`

SetUsage sets Usage field to given value.

### HasUsage

`func (o *RetainResponse) HasUsage() bool`

HasUsage returns a boolean if a field has been set.

### SetUsageNil

`func (o *RetainResponse) SetUsageNil(b bool)`

 SetUsageNil sets the value for Usage to be an explicit nil

### UnsetUsage
`func (o *RetainResponse) UnsetUsage()`

UnsetUsage ensures that no value is present for Usage, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


