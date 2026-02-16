# UpdateDirectiveRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Name** | Pointer to **NullableString** |  | [optional] 
**Content** | Pointer to **NullableString** |  | [optional] 
**Priority** | Pointer to **NullableInt32** |  | [optional] 
**IsActive** | Pointer to **NullableBool** |  | [optional] 
**Tags** | Pointer to **[]string** |  | [optional] 

## Methods

### NewUpdateDirectiveRequest

`func NewUpdateDirectiveRequest() *UpdateDirectiveRequest`

NewUpdateDirectiveRequest instantiates a new UpdateDirectiveRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewUpdateDirectiveRequestWithDefaults

`func NewUpdateDirectiveRequestWithDefaults() *UpdateDirectiveRequest`

NewUpdateDirectiveRequestWithDefaults instantiates a new UpdateDirectiveRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetName

`func (o *UpdateDirectiveRequest) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *UpdateDirectiveRequest) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *UpdateDirectiveRequest) SetName(v string)`

SetName sets Name field to given value.

### HasName

`func (o *UpdateDirectiveRequest) HasName() bool`

HasName returns a boolean if a field has been set.

### SetNameNil

`func (o *UpdateDirectiveRequest) SetNameNil(b bool)`

 SetNameNil sets the value for Name to be an explicit nil

### UnsetName
`func (o *UpdateDirectiveRequest) UnsetName()`

UnsetName ensures that no value is present for Name, not even an explicit nil
### GetContent

`func (o *UpdateDirectiveRequest) GetContent() string`

GetContent returns the Content field if non-nil, zero value otherwise.

### GetContentOk

`func (o *UpdateDirectiveRequest) GetContentOk() (*string, bool)`

GetContentOk returns a tuple with the Content field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContent

`func (o *UpdateDirectiveRequest) SetContent(v string)`

SetContent sets Content field to given value.

### HasContent

`func (o *UpdateDirectiveRequest) HasContent() bool`

HasContent returns a boolean if a field has been set.

### SetContentNil

`func (o *UpdateDirectiveRequest) SetContentNil(b bool)`

 SetContentNil sets the value for Content to be an explicit nil

### UnsetContent
`func (o *UpdateDirectiveRequest) UnsetContent()`

UnsetContent ensures that no value is present for Content, not even an explicit nil
### GetPriority

`func (o *UpdateDirectiveRequest) GetPriority() int32`

GetPriority returns the Priority field if non-nil, zero value otherwise.

### GetPriorityOk

`func (o *UpdateDirectiveRequest) GetPriorityOk() (*int32, bool)`

GetPriorityOk returns a tuple with the Priority field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetPriority

`func (o *UpdateDirectiveRequest) SetPriority(v int32)`

SetPriority sets Priority field to given value.

### HasPriority

`func (o *UpdateDirectiveRequest) HasPriority() bool`

HasPriority returns a boolean if a field has been set.

### SetPriorityNil

`func (o *UpdateDirectiveRequest) SetPriorityNil(b bool)`

 SetPriorityNil sets the value for Priority to be an explicit nil

### UnsetPriority
`func (o *UpdateDirectiveRequest) UnsetPriority()`

UnsetPriority ensures that no value is present for Priority, not even an explicit nil
### GetIsActive

`func (o *UpdateDirectiveRequest) GetIsActive() bool`

GetIsActive returns the IsActive field if non-nil, zero value otherwise.

### GetIsActiveOk

`func (o *UpdateDirectiveRequest) GetIsActiveOk() (*bool, bool)`

GetIsActiveOk returns a tuple with the IsActive field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetIsActive

`func (o *UpdateDirectiveRequest) SetIsActive(v bool)`

SetIsActive sets IsActive field to given value.

### HasIsActive

`func (o *UpdateDirectiveRequest) HasIsActive() bool`

HasIsActive returns a boolean if a field has been set.

### SetIsActiveNil

`func (o *UpdateDirectiveRequest) SetIsActiveNil(b bool)`

 SetIsActiveNil sets the value for IsActive to be an explicit nil

### UnsetIsActive
`func (o *UpdateDirectiveRequest) UnsetIsActive()`

UnsetIsActive ensures that no value is present for IsActive, not even an explicit nil
### GetTags

`func (o *UpdateDirectiveRequest) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *UpdateDirectiveRequest) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *UpdateDirectiveRequest) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *UpdateDirectiveRequest) HasTags() bool`

HasTags returns a boolean if a field has been set.

### SetTagsNil

`func (o *UpdateDirectiveRequest) SetTagsNil(b bool)`

 SetTagsNil sets the value for Tags to be an explicit nil

### UnsetTags
`func (o *UpdateDirectiveRequest) UnsetTags()`

UnsetTags ensures that no value is present for Tags, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


