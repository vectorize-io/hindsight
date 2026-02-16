# MentalModelResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | **string** |  | 
**BankId** | **string** |  | 
**Name** | **string** |  | 
**SourceQuery** | **string** |  | 
**Content** | **string** | The mental model content as well-formatted markdown (auto-generated from reflect endpoint) | 
**Tags** | Pointer to **[]string** |  | [optional] [default to []]
**MaxTokens** | Pointer to **int32** |  | [optional] [default to 2048]
**Trigger** | Pointer to [**MentalModelTrigger**](MentalModelTrigger.md) |  | [optional] 
**LastRefreshedAt** | Pointer to **NullableString** |  | [optional] 
**CreatedAt** | Pointer to **NullableString** |  | [optional] 
**ReflectResponse** | Pointer to **map[string]interface{}** |  | [optional] 

## Methods

### NewMentalModelResponse

`func NewMentalModelResponse(id string, bankId string, name string, sourceQuery string, content string, ) *MentalModelResponse`

NewMentalModelResponse instantiates a new MentalModelResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewMentalModelResponseWithDefaults

`func NewMentalModelResponseWithDefaults() *MentalModelResponse`

NewMentalModelResponseWithDefaults instantiates a new MentalModelResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *MentalModelResponse) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *MentalModelResponse) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *MentalModelResponse) SetId(v string)`

SetId sets Id field to given value.


### GetBankId

`func (o *MentalModelResponse) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *MentalModelResponse) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *MentalModelResponse) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetName

`func (o *MentalModelResponse) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *MentalModelResponse) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *MentalModelResponse) SetName(v string)`

SetName sets Name field to given value.


### GetSourceQuery

`func (o *MentalModelResponse) GetSourceQuery() string`

GetSourceQuery returns the SourceQuery field if non-nil, zero value otherwise.

### GetSourceQueryOk

`func (o *MentalModelResponse) GetSourceQueryOk() (*string, bool)`

GetSourceQueryOk returns a tuple with the SourceQuery field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetSourceQuery

`func (o *MentalModelResponse) SetSourceQuery(v string)`

SetSourceQuery sets SourceQuery field to given value.


### GetContent

`func (o *MentalModelResponse) GetContent() string`

GetContent returns the Content field if non-nil, zero value otherwise.

### GetContentOk

`func (o *MentalModelResponse) GetContentOk() (*string, bool)`

GetContentOk returns a tuple with the Content field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContent

`func (o *MentalModelResponse) SetContent(v string)`

SetContent sets Content field to given value.


### GetTags

`func (o *MentalModelResponse) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *MentalModelResponse) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *MentalModelResponse) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *MentalModelResponse) HasTags() bool`

HasTags returns a boolean if a field has been set.

### GetMaxTokens

`func (o *MentalModelResponse) GetMaxTokens() int32`

GetMaxTokens returns the MaxTokens field if non-nil, zero value otherwise.

### GetMaxTokensOk

`func (o *MentalModelResponse) GetMaxTokensOk() (*int32, bool)`

GetMaxTokensOk returns a tuple with the MaxTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMaxTokens

`func (o *MentalModelResponse) SetMaxTokens(v int32)`

SetMaxTokens sets MaxTokens field to given value.

### HasMaxTokens

`func (o *MentalModelResponse) HasMaxTokens() bool`

HasMaxTokens returns a boolean if a field has been set.

### GetTrigger

`func (o *MentalModelResponse) GetTrigger() MentalModelTrigger`

GetTrigger returns the Trigger field if non-nil, zero value otherwise.

### GetTriggerOk

`func (o *MentalModelResponse) GetTriggerOk() (*MentalModelTrigger, bool)`

GetTriggerOk returns a tuple with the Trigger field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTrigger

`func (o *MentalModelResponse) SetTrigger(v MentalModelTrigger)`

SetTrigger sets Trigger field to given value.

### HasTrigger

`func (o *MentalModelResponse) HasTrigger() bool`

HasTrigger returns a boolean if a field has been set.

### GetLastRefreshedAt

`func (o *MentalModelResponse) GetLastRefreshedAt() string`

GetLastRefreshedAt returns the LastRefreshedAt field if non-nil, zero value otherwise.

### GetLastRefreshedAtOk

`func (o *MentalModelResponse) GetLastRefreshedAtOk() (*string, bool)`

GetLastRefreshedAtOk returns a tuple with the LastRefreshedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLastRefreshedAt

`func (o *MentalModelResponse) SetLastRefreshedAt(v string)`

SetLastRefreshedAt sets LastRefreshedAt field to given value.

### HasLastRefreshedAt

`func (o *MentalModelResponse) HasLastRefreshedAt() bool`

HasLastRefreshedAt returns a boolean if a field has been set.

### SetLastRefreshedAtNil

`func (o *MentalModelResponse) SetLastRefreshedAtNil(b bool)`

 SetLastRefreshedAtNil sets the value for LastRefreshedAt to be an explicit nil

### UnsetLastRefreshedAt
`func (o *MentalModelResponse) UnsetLastRefreshedAt()`

UnsetLastRefreshedAt ensures that no value is present for LastRefreshedAt, not even an explicit nil
### GetCreatedAt

`func (o *MentalModelResponse) GetCreatedAt() string`

GetCreatedAt returns the CreatedAt field if non-nil, zero value otherwise.

### GetCreatedAtOk

`func (o *MentalModelResponse) GetCreatedAtOk() (*string, bool)`

GetCreatedAtOk returns a tuple with the CreatedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetCreatedAt

`func (o *MentalModelResponse) SetCreatedAt(v string)`

SetCreatedAt sets CreatedAt field to given value.

### HasCreatedAt

`func (o *MentalModelResponse) HasCreatedAt() bool`

HasCreatedAt returns a boolean if a field has been set.

### SetCreatedAtNil

`func (o *MentalModelResponse) SetCreatedAtNil(b bool)`

 SetCreatedAtNil sets the value for CreatedAt to be an explicit nil

### UnsetCreatedAt
`func (o *MentalModelResponse) UnsetCreatedAt()`

UnsetCreatedAt ensures that no value is present for CreatedAt, not even an explicit nil
### GetReflectResponse

`func (o *MentalModelResponse) GetReflectResponse() map[string]interface{}`

GetReflectResponse returns the ReflectResponse field if non-nil, zero value otherwise.

### GetReflectResponseOk

`func (o *MentalModelResponse) GetReflectResponseOk() (*map[string]interface{}, bool)`

GetReflectResponseOk returns a tuple with the ReflectResponse field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetReflectResponse

`func (o *MentalModelResponse) SetReflectResponse(v map[string]interface{})`

SetReflectResponse sets ReflectResponse field to given value.

### HasReflectResponse

`func (o *MentalModelResponse) HasReflectResponse() bool`

HasReflectResponse returns a boolean if a field has been set.

### SetReflectResponseNil

`func (o *MentalModelResponse) SetReflectResponseNil(b bool)`

 SetReflectResponseNil sets the value for ReflectResponse to be an explicit nil

### UnsetReflectResponse
`func (o *MentalModelResponse) UnsetReflectResponse()`

UnsetReflectResponse ensures that no value is present for ReflectResponse, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


