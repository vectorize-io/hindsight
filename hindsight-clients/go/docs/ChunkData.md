# ChunkData

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | **string** |  | 
**Text** | **string** |  | 
**ChunkIndex** | **int32** |  | 
**Truncated** | Pointer to **bool** | Whether the chunk text was truncated due to token limits | [optional] [default to false]

## Methods

### NewChunkData

`func NewChunkData(id string, text string, chunkIndex int32, ) *ChunkData`

NewChunkData instantiates a new ChunkData object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewChunkDataWithDefaults

`func NewChunkDataWithDefaults() *ChunkData`

NewChunkDataWithDefaults instantiates a new ChunkData object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *ChunkData) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *ChunkData) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *ChunkData) SetId(v string)`

SetId sets Id field to given value.


### GetText

`func (o *ChunkData) GetText() string`

GetText returns the Text field if non-nil, zero value otherwise.

### GetTextOk

`func (o *ChunkData) GetTextOk() (*string, bool)`

GetTextOk returns a tuple with the Text field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetText

`func (o *ChunkData) SetText(v string)`

SetText sets Text field to given value.


### GetChunkIndex

`func (o *ChunkData) GetChunkIndex() int32`

GetChunkIndex returns the ChunkIndex field if non-nil, zero value otherwise.

### GetChunkIndexOk

`func (o *ChunkData) GetChunkIndexOk() (*int32, bool)`

GetChunkIndexOk returns a tuple with the ChunkIndex field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetChunkIndex

`func (o *ChunkData) SetChunkIndex(v int32)`

SetChunkIndex sets ChunkIndex field to given value.


### GetTruncated

`func (o *ChunkData) GetTruncated() bool`

GetTruncated returns the Truncated field if non-nil, zero value otherwise.

### GetTruncatedOk

`func (o *ChunkData) GetTruncatedOk() (*bool, bool)`

GetTruncatedOk returns a tuple with the Truncated field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTruncated

`func (o *ChunkData) SetTruncated(v bool)`

SetTruncated sets Truncated field to given value.

### HasTruncated

`func (o *ChunkData) HasTruncated() bool`

HasTruncated returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


