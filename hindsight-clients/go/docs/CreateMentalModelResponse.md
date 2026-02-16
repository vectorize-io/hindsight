# CreateMentalModelResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**MentalModelId** | Pointer to **NullableString** |  | [optional] 
**OperationId** | **string** | Operation ID to track refresh progress | 

## Methods

### NewCreateMentalModelResponse

`func NewCreateMentalModelResponse(operationId string, ) *CreateMentalModelResponse`

NewCreateMentalModelResponse instantiates a new CreateMentalModelResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewCreateMentalModelResponseWithDefaults

`func NewCreateMentalModelResponseWithDefaults() *CreateMentalModelResponse`

NewCreateMentalModelResponseWithDefaults instantiates a new CreateMentalModelResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetMentalModelId

`func (o *CreateMentalModelResponse) GetMentalModelId() string`

GetMentalModelId returns the MentalModelId field if non-nil, zero value otherwise.

### GetMentalModelIdOk

`func (o *CreateMentalModelResponse) GetMentalModelIdOk() (*string, bool)`

GetMentalModelIdOk returns a tuple with the MentalModelId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMentalModelId

`func (o *CreateMentalModelResponse) SetMentalModelId(v string)`

SetMentalModelId sets MentalModelId field to given value.

### HasMentalModelId

`func (o *CreateMentalModelResponse) HasMentalModelId() bool`

HasMentalModelId returns a boolean if a field has been set.

### SetMentalModelIdNil

`func (o *CreateMentalModelResponse) SetMentalModelIdNil(b bool)`

 SetMentalModelIdNil sets the value for MentalModelId to be an explicit nil

### UnsetMentalModelId
`func (o *CreateMentalModelResponse) UnsetMentalModelId()`

UnsetMentalModelId ensures that no value is present for MentalModelId, not even an explicit nil
### GetOperationId

`func (o *CreateMentalModelResponse) GetOperationId() string`

GetOperationId returns the OperationId field if non-nil, zero value otherwise.

### GetOperationIdOk

`func (o *CreateMentalModelResponse) GetOperationIdOk() (*string, bool)`

GetOperationIdOk returns a tuple with the OperationId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOperationId

`func (o *CreateMentalModelResponse) SetOperationId(v string)`

SetOperationId sets OperationId field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


