# IncludeOptions

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Entities** | Pointer to [**NullableEntityIncludeOptions**](EntityIncludeOptions.md) |  | [optional] 
**Chunks** | Pointer to [**NullableChunkIncludeOptions**](ChunkIncludeOptions.md) |  | [optional] 

## Methods

### NewIncludeOptions

`func NewIncludeOptions() *IncludeOptions`

NewIncludeOptions instantiates a new IncludeOptions object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewIncludeOptionsWithDefaults

`func NewIncludeOptionsWithDefaults() *IncludeOptions`

NewIncludeOptionsWithDefaults instantiates a new IncludeOptions object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetEntities

`func (o *IncludeOptions) GetEntities() EntityIncludeOptions`

GetEntities returns the Entities field if non-nil, zero value otherwise.

### GetEntitiesOk

`func (o *IncludeOptions) GetEntitiesOk() (*EntityIncludeOptions, bool)`

GetEntitiesOk returns a tuple with the Entities field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetEntities

`func (o *IncludeOptions) SetEntities(v EntityIncludeOptions)`

SetEntities sets Entities field to given value.

### HasEntities

`func (o *IncludeOptions) HasEntities() bool`

HasEntities returns a boolean if a field has been set.

### SetEntitiesNil

`func (o *IncludeOptions) SetEntitiesNil(b bool)`

 SetEntitiesNil sets the value for Entities to be an explicit nil

### UnsetEntities
`func (o *IncludeOptions) UnsetEntities()`

UnsetEntities ensures that no value is present for Entities, not even an explicit nil
### GetChunks

`func (o *IncludeOptions) GetChunks() ChunkIncludeOptions`

GetChunks returns the Chunks field if non-nil, zero value otherwise.

### GetChunksOk

`func (o *IncludeOptions) GetChunksOk() (*ChunkIncludeOptions, bool)`

GetChunksOk returns a tuple with the Chunks field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetChunks

`func (o *IncludeOptions) SetChunks(v ChunkIncludeOptions)`

SetChunks sets Chunks field to given value.

### HasChunks

`func (o *IncludeOptions) HasChunks() bool`

HasChunks returns a boolean if a field has been set.

### SetChunksNil

`func (o *IncludeOptions) SetChunksNil(b bool)`

 SetChunksNil sets the value for Chunks to be an explicit nil

### UnsetChunks
`func (o *IncludeOptions) UnsetChunks()`

UnsetChunks ensures that no value is present for Chunks, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


