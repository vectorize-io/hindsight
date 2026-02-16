# AddBackgroundRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Content** | **string** | New background information to add or merge | 
**UpdateDisposition** | Pointer to **bool** | Deprecated - disposition is no longer auto-inferred from mission | [optional] [default to true]

## Methods

### NewAddBackgroundRequest

`func NewAddBackgroundRequest(content string, ) *AddBackgroundRequest`

NewAddBackgroundRequest instantiates a new AddBackgroundRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewAddBackgroundRequestWithDefaults

`func NewAddBackgroundRequestWithDefaults() *AddBackgroundRequest`

NewAddBackgroundRequestWithDefaults instantiates a new AddBackgroundRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetContent

`func (o *AddBackgroundRequest) GetContent() string`

GetContent returns the Content field if non-nil, zero value otherwise.

### GetContentOk

`func (o *AddBackgroundRequest) GetContentOk() (*string, bool)`

GetContentOk returns a tuple with the Content field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContent

`func (o *AddBackgroundRequest) SetContent(v string)`

SetContent sets Content field to given value.


### GetUpdateDisposition

`func (o *AddBackgroundRequest) GetUpdateDisposition() bool`

GetUpdateDisposition returns the UpdateDisposition field if non-nil, zero value otherwise.

### GetUpdateDispositionOk

`func (o *AddBackgroundRequest) GetUpdateDispositionOk() (*bool, bool)`

GetUpdateDispositionOk returns a tuple with the UpdateDisposition field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetUpdateDisposition

`func (o *AddBackgroundRequest) SetUpdateDisposition(v bool)`

SetUpdateDisposition sets UpdateDisposition field to given value.

### HasUpdateDisposition

`func (o *AddBackgroundRequest) HasUpdateDisposition() bool`

HasUpdateDisposition returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


