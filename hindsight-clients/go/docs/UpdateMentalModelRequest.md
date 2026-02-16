# UpdateMentalModelRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Name** | Pointer to **NullableString** |  | [optional] 
**SourceQuery** | Pointer to **NullableString** |  | [optional] 
**MaxTokens** | Pointer to **NullableInt32** |  | [optional] 
**Tags** | Pointer to **[]string** |  | [optional] 
**Trigger** | Pointer to [**NullableMentalModelTrigger**](MentalModelTrigger.md) |  | [optional] 

## Methods

### NewUpdateMentalModelRequest

`func NewUpdateMentalModelRequest() *UpdateMentalModelRequest`

NewUpdateMentalModelRequest instantiates a new UpdateMentalModelRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewUpdateMentalModelRequestWithDefaults

`func NewUpdateMentalModelRequestWithDefaults() *UpdateMentalModelRequest`

NewUpdateMentalModelRequestWithDefaults instantiates a new UpdateMentalModelRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetName

`func (o *UpdateMentalModelRequest) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *UpdateMentalModelRequest) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *UpdateMentalModelRequest) SetName(v string)`

SetName sets Name field to given value.

### HasName

`func (o *UpdateMentalModelRequest) HasName() bool`

HasName returns a boolean if a field has been set.

### SetNameNil

`func (o *UpdateMentalModelRequest) SetNameNil(b bool)`

 SetNameNil sets the value for Name to be an explicit nil

### UnsetName
`func (o *UpdateMentalModelRequest) UnsetName()`

UnsetName ensures that no value is present for Name, not even an explicit nil
### GetSourceQuery

`func (o *UpdateMentalModelRequest) GetSourceQuery() string`

GetSourceQuery returns the SourceQuery field if non-nil, zero value otherwise.

### GetSourceQueryOk

`func (o *UpdateMentalModelRequest) GetSourceQueryOk() (*string, bool)`

GetSourceQueryOk returns a tuple with the SourceQuery field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetSourceQuery

`func (o *UpdateMentalModelRequest) SetSourceQuery(v string)`

SetSourceQuery sets SourceQuery field to given value.

### HasSourceQuery

`func (o *UpdateMentalModelRequest) HasSourceQuery() bool`

HasSourceQuery returns a boolean if a field has been set.

### SetSourceQueryNil

`func (o *UpdateMentalModelRequest) SetSourceQueryNil(b bool)`

 SetSourceQueryNil sets the value for SourceQuery to be an explicit nil

### UnsetSourceQuery
`func (o *UpdateMentalModelRequest) UnsetSourceQuery()`

UnsetSourceQuery ensures that no value is present for SourceQuery, not even an explicit nil
### GetMaxTokens

`func (o *UpdateMentalModelRequest) GetMaxTokens() int32`

GetMaxTokens returns the MaxTokens field if non-nil, zero value otherwise.

### GetMaxTokensOk

`func (o *UpdateMentalModelRequest) GetMaxTokensOk() (*int32, bool)`

GetMaxTokensOk returns a tuple with the MaxTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMaxTokens

`func (o *UpdateMentalModelRequest) SetMaxTokens(v int32)`

SetMaxTokens sets MaxTokens field to given value.

### HasMaxTokens

`func (o *UpdateMentalModelRequest) HasMaxTokens() bool`

HasMaxTokens returns a boolean if a field has been set.

### SetMaxTokensNil

`func (o *UpdateMentalModelRequest) SetMaxTokensNil(b bool)`

 SetMaxTokensNil sets the value for MaxTokens to be an explicit nil

### UnsetMaxTokens
`func (o *UpdateMentalModelRequest) UnsetMaxTokens()`

UnsetMaxTokens ensures that no value is present for MaxTokens, not even an explicit nil
### GetTags

`func (o *UpdateMentalModelRequest) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *UpdateMentalModelRequest) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *UpdateMentalModelRequest) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *UpdateMentalModelRequest) HasTags() bool`

HasTags returns a boolean if a field has been set.

### SetTagsNil

`func (o *UpdateMentalModelRequest) SetTagsNil(b bool)`

 SetTagsNil sets the value for Tags to be an explicit nil

### UnsetTags
`func (o *UpdateMentalModelRequest) UnsetTags()`

UnsetTags ensures that no value is present for Tags, not even an explicit nil
### GetTrigger

`func (o *UpdateMentalModelRequest) GetTrigger() MentalModelTrigger`

GetTrigger returns the Trigger field if non-nil, zero value otherwise.

### GetTriggerOk

`func (o *UpdateMentalModelRequest) GetTriggerOk() (*MentalModelTrigger, bool)`

GetTriggerOk returns a tuple with the Trigger field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTrigger

`func (o *UpdateMentalModelRequest) SetTrigger(v MentalModelTrigger)`

SetTrigger sets Trigger field to given value.

### HasTrigger

`func (o *UpdateMentalModelRequest) HasTrigger() bool`

HasTrigger returns a boolean if a field has been set.

### SetTriggerNil

`func (o *UpdateMentalModelRequest) SetTriggerNil(b bool)`

 SetTriggerNil sets the value for Trigger to be an explicit nil

### UnsetTrigger
`func (o *UpdateMentalModelRequest) UnsetTrigger()`

UnsetTrigger ensures that no value is present for Trigger, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


