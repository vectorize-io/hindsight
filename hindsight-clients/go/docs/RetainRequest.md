# RetainRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Items** | [**[]MemoryItem**](MemoryItem.md) |  | 
**Async** | Pointer to **bool** | If true, process asynchronously in background. If false, wait for completion (default: false) | [optional] [default to false]
**DocumentTags** | Pointer to **[]string** |  | [optional] 

## Methods

### NewRetainRequest

`func NewRetainRequest(items []MemoryItem, ) *RetainRequest`

NewRetainRequest instantiates a new RetainRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewRetainRequestWithDefaults

`func NewRetainRequestWithDefaults() *RetainRequest`

NewRetainRequestWithDefaults instantiates a new RetainRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetItems

`func (o *RetainRequest) GetItems() []MemoryItem`

GetItems returns the Items field if non-nil, zero value otherwise.

### GetItemsOk

`func (o *RetainRequest) GetItemsOk() (*[]MemoryItem, bool)`

GetItemsOk returns a tuple with the Items field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetItems

`func (o *RetainRequest) SetItems(v []MemoryItem)`

SetItems sets Items field to given value.


### GetAsync

`func (o *RetainRequest) GetAsync() bool`

GetAsync returns the Async field if non-nil, zero value otherwise.

### GetAsyncOk

`func (o *RetainRequest) GetAsyncOk() (*bool, bool)`

GetAsyncOk returns a tuple with the Async field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetAsync

`func (o *RetainRequest) SetAsync(v bool)`

SetAsync sets Async field to given value.

### HasAsync

`func (o *RetainRequest) HasAsync() bool`

HasAsync returns a boolean if a field has been set.

### GetDocumentTags

`func (o *RetainRequest) GetDocumentTags() []string`

GetDocumentTags returns the DocumentTags field if non-nil, zero value otherwise.

### GetDocumentTagsOk

`func (o *RetainRequest) GetDocumentTagsOk() (*[]string, bool)`

GetDocumentTagsOk returns a tuple with the DocumentTags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDocumentTags

`func (o *RetainRequest) SetDocumentTags(v []string)`

SetDocumentTags sets DocumentTags field to given value.

### HasDocumentTags

`func (o *RetainRequest) HasDocumentTags() bool`

HasDocumentTags returns a boolean if a field has been set.

### SetDocumentTagsNil

`func (o *RetainRequest) SetDocumentTagsNil(b bool)`

 SetDocumentTagsNil sets the value for DocumentTags to be an explicit nil

### UnsetDocumentTags
`func (o *RetainRequest) UnsetDocumentTags()`

UnsetDocumentTags ensures that no value is present for DocumentTags, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


