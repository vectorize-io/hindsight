# MemoryItem

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Content** | **string** |  | 
**Timestamp** | Pointer to **NullableTime** |  | [optional] 
**Context** | Pointer to **NullableString** |  | [optional] 
**Metadata** | Pointer to **map[string]string** |  | [optional] 
**DocumentId** | Pointer to **NullableString** |  | [optional] 
**Entities** | Pointer to [**[]EntityInput**](EntityInput.md) |  | [optional] 
**Tags** | Pointer to **[]string** |  | [optional] 

## Methods

### NewMemoryItem

`func NewMemoryItem(content string, ) *MemoryItem`

NewMemoryItem instantiates a new MemoryItem object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewMemoryItemWithDefaults

`func NewMemoryItemWithDefaults() *MemoryItem`

NewMemoryItemWithDefaults instantiates a new MemoryItem object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetContent

`func (o *MemoryItem) GetContent() string`

GetContent returns the Content field if non-nil, zero value otherwise.

### GetContentOk

`func (o *MemoryItem) GetContentOk() (*string, bool)`

GetContentOk returns a tuple with the Content field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContent

`func (o *MemoryItem) SetContent(v string)`

SetContent sets Content field to given value.


### GetTimestamp

`func (o *MemoryItem) GetTimestamp() time.Time`

GetTimestamp returns the Timestamp field if non-nil, zero value otherwise.

### GetTimestampOk

`func (o *MemoryItem) GetTimestampOk() (*time.Time, bool)`

GetTimestampOk returns a tuple with the Timestamp field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTimestamp

`func (o *MemoryItem) SetTimestamp(v time.Time)`

SetTimestamp sets Timestamp field to given value.

### HasTimestamp

`func (o *MemoryItem) HasTimestamp() bool`

HasTimestamp returns a boolean if a field has been set.

### SetTimestampNil

`func (o *MemoryItem) SetTimestampNil(b bool)`

 SetTimestampNil sets the value for Timestamp to be an explicit nil

### UnsetTimestamp
`func (o *MemoryItem) UnsetTimestamp()`

UnsetTimestamp ensures that no value is present for Timestamp, not even an explicit nil
### GetContext

`func (o *MemoryItem) GetContext() string`

GetContext returns the Context field if non-nil, zero value otherwise.

### GetContextOk

`func (o *MemoryItem) GetContextOk() (*string, bool)`

GetContextOk returns a tuple with the Context field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContext

`func (o *MemoryItem) SetContext(v string)`

SetContext sets Context field to given value.

### HasContext

`func (o *MemoryItem) HasContext() bool`

HasContext returns a boolean if a field has been set.

### SetContextNil

`func (o *MemoryItem) SetContextNil(b bool)`

 SetContextNil sets the value for Context to be an explicit nil

### UnsetContext
`func (o *MemoryItem) UnsetContext()`

UnsetContext ensures that no value is present for Context, not even an explicit nil
### GetMetadata

`func (o *MemoryItem) GetMetadata() map[string]string`

GetMetadata returns the Metadata field if non-nil, zero value otherwise.

### GetMetadataOk

`func (o *MemoryItem) GetMetadataOk() (*map[string]string, bool)`

GetMetadataOk returns a tuple with the Metadata field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMetadata

`func (o *MemoryItem) SetMetadata(v map[string]string)`

SetMetadata sets Metadata field to given value.

### HasMetadata

`func (o *MemoryItem) HasMetadata() bool`

HasMetadata returns a boolean if a field has been set.

### SetMetadataNil

`func (o *MemoryItem) SetMetadataNil(b bool)`

 SetMetadataNil sets the value for Metadata to be an explicit nil

### UnsetMetadata
`func (o *MemoryItem) UnsetMetadata()`

UnsetMetadata ensures that no value is present for Metadata, not even an explicit nil
### GetDocumentId

`func (o *MemoryItem) GetDocumentId() string`

GetDocumentId returns the DocumentId field if non-nil, zero value otherwise.

### GetDocumentIdOk

`func (o *MemoryItem) GetDocumentIdOk() (*string, bool)`

GetDocumentIdOk returns a tuple with the DocumentId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDocumentId

`func (o *MemoryItem) SetDocumentId(v string)`

SetDocumentId sets DocumentId field to given value.

### HasDocumentId

`func (o *MemoryItem) HasDocumentId() bool`

HasDocumentId returns a boolean if a field has been set.

### SetDocumentIdNil

`func (o *MemoryItem) SetDocumentIdNil(b bool)`

 SetDocumentIdNil sets the value for DocumentId to be an explicit nil

### UnsetDocumentId
`func (o *MemoryItem) UnsetDocumentId()`

UnsetDocumentId ensures that no value is present for DocumentId, not even an explicit nil
### GetEntities

`func (o *MemoryItem) GetEntities() []EntityInput`

GetEntities returns the Entities field if non-nil, zero value otherwise.

### GetEntitiesOk

`func (o *MemoryItem) GetEntitiesOk() (*[]EntityInput, bool)`

GetEntitiesOk returns a tuple with the Entities field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetEntities

`func (o *MemoryItem) SetEntities(v []EntityInput)`

SetEntities sets Entities field to given value.

### HasEntities

`func (o *MemoryItem) HasEntities() bool`

HasEntities returns a boolean if a field has been set.

### SetEntitiesNil

`func (o *MemoryItem) SetEntitiesNil(b bool)`

 SetEntitiesNil sets the value for Entities to be an explicit nil

### UnsetEntities
`func (o *MemoryItem) UnsetEntities()`

UnsetEntities ensures that no value is present for Entities, not even an explicit nil
### GetTags

`func (o *MemoryItem) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *MemoryItem) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *MemoryItem) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *MemoryItem) HasTags() bool`

HasTags returns a boolean if a field has been set.

### SetTagsNil

`func (o *MemoryItem) SetTagsNil(b bool)`

 SetTagsNil sets the value for Tags to be an explicit nil

### UnsetTags
`func (o *MemoryItem) UnsetTags()`

UnsetTags ensures that no value is present for Tags, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


