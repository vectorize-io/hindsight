# OperationsListResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**BankId** | **string** |  | 
**Total** | **int32** |  | 
**Limit** | **int32** |  | 
**Offset** | **int32** |  | 
**Operations** | [**[]OperationResponse**](OperationResponse.md) |  | 

## Methods

### NewOperationsListResponse

`func NewOperationsListResponse(bankId string, total int32, limit int32, offset int32, operations []OperationResponse, ) *OperationsListResponse`

NewOperationsListResponse instantiates a new OperationsListResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewOperationsListResponseWithDefaults

`func NewOperationsListResponseWithDefaults() *OperationsListResponse`

NewOperationsListResponseWithDefaults instantiates a new OperationsListResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetBankId

`func (o *OperationsListResponse) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *OperationsListResponse) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *OperationsListResponse) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetTotal

`func (o *OperationsListResponse) GetTotal() int32`

GetTotal returns the Total field if non-nil, zero value otherwise.

### GetTotalOk

`func (o *OperationsListResponse) GetTotalOk() (*int32, bool)`

GetTotalOk returns a tuple with the Total field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotal

`func (o *OperationsListResponse) SetTotal(v int32)`

SetTotal sets Total field to given value.


### GetLimit

`func (o *OperationsListResponse) GetLimit() int32`

GetLimit returns the Limit field if non-nil, zero value otherwise.

### GetLimitOk

`func (o *OperationsListResponse) GetLimitOk() (*int32, bool)`

GetLimitOk returns a tuple with the Limit field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLimit

`func (o *OperationsListResponse) SetLimit(v int32)`

SetLimit sets Limit field to given value.


### GetOffset

`func (o *OperationsListResponse) GetOffset() int32`

GetOffset returns the Offset field if non-nil, zero value otherwise.

### GetOffsetOk

`func (o *OperationsListResponse) GetOffsetOk() (*int32, bool)`

GetOffsetOk returns a tuple with the Offset field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOffset

`func (o *OperationsListResponse) SetOffset(v int32)`

SetOffset sets Offset field to given value.


### GetOperations

`func (o *OperationsListResponse) GetOperations() []OperationResponse`

GetOperations returns the Operations field if non-nil, zero value otherwise.

### GetOperationsOk

`func (o *OperationsListResponse) GetOperationsOk() (*[]OperationResponse, bool)`

GetOperationsOk returns a tuple with the Operations field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOperations

`func (o *OperationsListResponse) SetOperations(v []OperationResponse)`

SetOperations sets Operations field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


