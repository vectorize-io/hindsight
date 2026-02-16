# RecallResult

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | **string** |  | 
**Text** | **string** |  | 
**Type** | Pointer to **NullableString** |  | [optional] 
**Entities** | Pointer to **[]string** |  | [optional] 
**Context** | Pointer to **NullableString** |  | [optional] 
**OccurredStart** | Pointer to **NullableString** |  | [optional] 
**OccurredEnd** | Pointer to **NullableString** |  | [optional] 
**MentionedAt** | Pointer to **NullableString** |  | [optional] 
**DocumentId** | Pointer to **NullableString** |  | [optional] 
**Metadata** | Pointer to **map[string]string** |  | [optional] 
**ChunkId** | Pointer to **NullableString** |  | [optional] 
**Tags** | Pointer to **[]string** |  | [optional] 

## Methods

### NewRecallResult

`func NewRecallResult(id string, text string, ) *RecallResult`

NewRecallResult instantiates a new RecallResult object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewRecallResultWithDefaults

`func NewRecallResultWithDefaults() *RecallResult`

NewRecallResultWithDefaults instantiates a new RecallResult object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *RecallResult) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *RecallResult) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *RecallResult) SetId(v string)`

SetId sets Id field to given value.


### GetText

`func (o *RecallResult) GetText() string`

GetText returns the Text field if non-nil, zero value otherwise.

### GetTextOk

`func (o *RecallResult) GetTextOk() (*string, bool)`

GetTextOk returns a tuple with the Text field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetText

`func (o *RecallResult) SetText(v string)`

SetText sets Text field to given value.


### GetType

`func (o *RecallResult) GetType() string`

GetType returns the Type field if non-nil, zero value otherwise.

### GetTypeOk

`func (o *RecallResult) GetTypeOk() (*string, bool)`

GetTypeOk returns a tuple with the Type field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetType

`func (o *RecallResult) SetType(v string)`

SetType sets Type field to given value.

### HasType

`func (o *RecallResult) HasType() bool`

HasType returns a boolean if a field has been set.

### SetTypeNil

`func (o *RecallResult) SetTypeNil(b bool)`

 SetTypeNil sets the value for Type to be an explicit nil

### UnsetType
`func (o *RecallResult) UnsetType()`

UnsetType ensures that no value is present for Type, not even an explicit nil
### GetEntities

`func (o *RecallResult) GetEntities() []string`

GetEntities returns the Entities field if non-nil, zero value otherwise.

### GetEntitiesOk

`func (o *RecallResult) GetEntitiesOk() (*[]string, bool)`

GetEntitiesOk returns a tuple with the Entities field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetEntities

`func (o *RecallResult) SetEntities(v []string)`

SetEntities sets Entities field to given value.

### HasEntities

`func (o *RecallResult) HasEntities() bool`

HasEntities returns a boolean if a field has been set.

### SetEntitiesNil

`func (o *RecallResult) SetEntitiesNil(b bool)`

 SetEntitiesNil sets the value for Entities to be an explicit nil

### UnsetEntities
`func (o *RecallResult) UnsetEntities()`

UnsetEntities ensures that no value is present for Entities, not even an explicit nil
### GetContext

`func (o *RecallResult) GetContext() string`

GetContext returns the Context field if non-nil, zero value otherwise.

### GetContextOk

`func (o *RecallResult) GetContextOk() (*string, bool)`

GetContextOk returns a tuple with the Context field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContext

`func (o *RecallResult) SetContext(v string)`

SetContext sets Context field to given value.

### HasContext

`func (o *RecallResult) HasContext() bool`

HasContext returns a boolean if a field has been set.

### SetContextNil

`func (o *RecallResult) SetContextNil(b bool)`

 SetContextNil sets the value for Context to be an explicit nil

### UnsetContext
`func (o *RecallResult) UnsetContext()`

UnsetContext ensures that no value is present for Context, not even an explicit nil
### GetOccurredStart

`func (o *RecallResult) GetOccurredStart() string`

GetOccurredStart returns the OccurredStart field if non-nil, zero value otherwise.

### GetOccurredStartOk

`func (o *RecallResult) GetOccurredStartOk() (*string, bool)`

GetOccurredStartOk returns a tuple with the OccurredStart field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOccurredStart

`func (o *RecallResult) SetOccurredStart(v string)`

SetOccurredStart sets OccurredStart field to given value.

### HasOccurredStart

`func (o *RecallResult) HasOccurredStart() bool`

HasOccurredStart returns a boolean if a field has been set.

### SetOccurredStartNil

`func (o *RecallResult) SetOccurredStartNil(b bool)`

 SetOccurredStartNil sets the value for OccurredStart to be an explicit nil

### UnsetOccurredStart
`func (o *RecallResult) UnsetOccurredStart()`

UnsetOccurredStart ensures that no value is present for OccurredStart, not even an explicit nil
### GetOccurredEnd

`func (o *RecallResult) GetOccurredEnd() string`

GetOccurredEnd returns the OccurredEnd field if non-nil, zero value otherwise.

### GetOccurredEndOk

`func (o *RecallResult) GetOccurredEndOk() (*string, bool)`

GetOccurredEndOk returns a tuple with the OccurredEnd field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOccurredEnd

`func (o *RecallResult) SetOccurredEnd(v string)`

SetOccurredEnd sets OccurredEnd field to given value.

### HasOccurredEnd

`func (o *RecallResult) HasOccurredEnd() bool`

HasOccurredEnd returns a boolean if a field has been set.

### SetOccurredEndNil

`func (o *RecallResult) SetOccurredEndNil(b bool)`

 SetOccurredEndNil sets the value for OccurredEnd to be an explicit nil

### UnsetOccurredEnd
`func (o *RecallResult) UnsetOccurredEnd()`

UnsetOccurredEnd ensures that no value is present for OccurredEnd, not even an explicit nil
### GetMentionedAt

`func (o *RecallResult) GetMentionedAt() string`

GetMentionedAt returns the MentionedAt field if non-nil, zero value otherwise.

### GetMentionedAtOk

`func (o *RecallResult) GetMentionedAtOk() (*string, bool)`

GetMentionedAtOk returns a tuple with the MentionedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMentionedAt

`func (o *RecallResult) SetMentionedAt(v string)`

SetMentionedAt sets MentionedAt field to given value.

### HasMentionedAt

`func (o *RecallResult) HasMentionedAt() bool`

HasMentionedAt returns a boolean if a field has been set.

### SetMentionedAtNil

`func (o *RecallResult) SetMentionedAtNil(b bool)`

 SetMentionedAtNil sets the value for MentionedAt to be an explicit nil

### UnsetMentionedAt
`func (o *RecallResult) UnsetMentionedAt()`

UnsetMentionedAt ensures that no value is present for MentionedAt, not even an explicit nil
### GetDocumentId

`func (o *RecallResult) GetDocumentId() string`

GetDocumentId returns the DocumentId field if non-nil, zero value otherwise.

### GetDocumentIdOk

`func (o *RecallResult) GetDocumentIdOk() (*string, bool)`

GetDocumentIdOk returns a tuple with the DocumentId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDocumentId

`func (o *RecallResult) SetDocumentId(v string)`

SetDocumentId sets DocumentId field to given value.

### HasDocumentId

`func (o *RecallResult) HasDocumentId() bool`

HasDocumentId returns a boolean if a field has been set.

### SetDocumentIdNil

`func (o *RecallResult) SetDocumentIdNil(b bool)`

 SetDocumentIdNil sets the value for DocumentId to be an explicit nil

### UnsetDocumentId
`func (o *RecallResult) UnsetDocumentId()`

UnsetDocumentId ensures that no value is present for DocumentId, not even an explicit nil
### GetMetadata

`func (o *RecallResult) GetMetadata() map[string]string`

GetMetadata returns the Metadata field if non-nil, zero value otherwise.

### GetMetadataOk

`func (o *RecallResult) GetMetadataOk() (*map[string]string, bool)`

GetMetadataOk returns a tuple with the Metadata field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMetadata

`func (o *RecallResult) SetMetadata(v map[string]string)`

SetMetadata sets Metadata field to given value.

### HasMetadata

`func (o *RecallResult) HasMetadata() bool`

HasMetadata returns a boolean if a field has been set.

### SetMetadataNil

`func (o *RecallResult) SetMetadataNil(b bool)`

 SetMetadataNil sets the value for Metadata to be an explicit nil

### UnsetMetadata
`func (o *RecallResult) UnsetMetadata()`

UnsetMetadata ensures that no value is present for Metadata, not even an explicit nil
### GetChunkId

`func (o *RecallResult) GetChunkId() string`

GetChunkId returns the ChunkId field if non-nil, zero value otherwise.

### GetChunkIdOk

`func (o *RecallResult) GetChunkIdOk() (*string, bool)`

GetChunkIdOk returns a tuple with the ChunkId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetChunkId

`func (o *RecallResult) SetChunkId(v string)`

SetChunkId sets ChunkId field to given value.

### HasChunkId

`func (o *RecallResult) HasChunkId() bool`

HasChunkId returns a boolean if a field has been set.

### SetChunkIdNil

`func (o *RecallResult) SetChunkIdNil(b bool)`

 SetChunkIdNil sets the value for ChunkId to be an explicit nil

### UnsetChunkId
`func (o *RecallResult) UnsetChunkId()`

UnsetChunkId ensures that no value is present for ChunkId, not even an explicit nil
### GetTags

`func (o *RecallResult) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *RecallResult) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *RecallResult) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *RecallResult) HasTags() bool`

HasTags returns a boolean if a field has been set.

### SetTagsNil

`func (o *RecallResult) SetTagsNil(b bool)`

 SetTagsNil sets the value for Tags to be an explicit nil

### UnsetTags
`func (o *RecallResult) UnsetTags()`

UnsetTags ensures that no value is present for Tags, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


