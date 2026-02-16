# CreateDirectiveRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Name** | **string** | Human-readable name for the directive | 
**Content** | **string** | The directive text to inject into prompts | 
**Priority** | Pointer to **int32** | Higher priority directives are injected first | [optional] [default to 0]
**IsActive** | Pointer to **bool** | Whether this directive is active | [optional] [default to true]
**Tags** | Pointer to **[]string** | Tags for filtering | [optional] [default to []]

## Methods

### NewCreateDirectiveRequest

`func NewCreateDirectiveRequest(name string, content string, ) *CreateDirectiveRequest`

NewCreateDirectiveRequest instantiates a new CreateDirectiveRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewCreateDirectiveRequestWithDefaults

`func NewCreateDirectiveRequestWithDefaults() *CreateDirectiveRequest`

NewCreateDirectiveRequestWithDefaults instantiates a new CreateDirectiveRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetName

`func (o *CreateDirectiveRequest) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *CreateDirectiveRequest) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *CreateDirectiveRequest) SetName(v string)`

SetName sets Name field to given value.


### GetContent

`func (o *CreateDirectiveRequest) GetContent() string`

GetContent returns the Content field if non-nil, zero value otherwise.

### GetContentOk

`func (o *CreateDirectiveRequest) GetContentOk() (*string, bool)`

GetContentOk returns a tuple with the Content field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContent

`func (o *CreateDirectiveRequest) SetContent(v string)`

SetContent sets Content field to given value.


### GetPriority

`func (o *CreateDirectiveRequest) GetPriority() int32`

GetPriority returns the Priority field if non-nil, zero value otherwise.

### GetPriorityOk

`func (o *CreateDirectiveRequest) GetPriorityOk() (*int32, bool)`

GetPriorityOk returns a tuple with the Priority field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetPriority

`func (o *CreateDirectiveRequest) SetPriority(v int32)`

SetPriority sets Priority field to given value.

### HasPriority

`func (o *CreateDirectiveRequest) HasPriority() bool`

HasPriority returns a boolean if a field has been set.

### GetIsActive

`func (o *CreateDirectiveRequest) GetIsActive() bool`

GetIsActive returns the IsActive field if non-nil, zero value otherwise.

### GetIsActiveOk

`func (o *CreateDirectiveRequest) GetIsActiveOk() (*bool, bool)`

GetIsActiveOk returns a tuple with the IsActive field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetIsActive

`func (o *CreateDirectiveRequest) SetIsActive(v bool)`

SetIsActive sets IsActive field to given value.

### HasIsActive

`func (o *CreateDirectiveRequest) HasIsActive() bool`

HasIsActive returns a boolean if a field has been set.

### GetTags

`func (o *CreateDirectiveRequest) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *CreateDirectiveRequest) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *CreateDirectiveRequest) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *CreateDirectiveRequest) HasTags() bool`

HasTags returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


