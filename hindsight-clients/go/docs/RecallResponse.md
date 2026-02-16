# RecallResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Results** | [**[]RecallResult**](RecallResult.md) |  | 
**Trace** | Pointer to **map[string]interface{}** |  | [optional] 
**Entities** | Pointer to [**map[string]EntityStateResponse**](EntityStateResponse.md) |  | [optional] 
**Chunks** | Pointer to [**map[string]ChunkData**](ChunkData.md) |  | [optional] 

## Methods

### NewRecallResponse

`func NewRecallResponse(results []RecallResult, ) *RecallResponse`

NewRecallResponse instantiates a new RecallResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewRecallResponseWithDefaults

`func NewRecallResponseWithDefaults() *RecallResponse`

NewRecallResponseWithDefaults instantiates a new RecallResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetResults

`func (o *RecallResponse) GetResults() []RecallResult`

GetResults returns the Results field if non-nil, zero value otherwise.

### GetResultsOk

`func (o *RecallResponse) GetResultsOk() (*[]RecallResult, bool)`

GetResultsOk returns a tuple with the Results field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetResults

`func (o *RecallResponse) SetResults(v []RecallResult)`

SetResults sets Results field to given value.


### GetTrace

`func (o *RecallResponse) GetTrace() map[string]interface{}`

GetTrace returns the Trace field if non-nil, zero value otherwise.

### GetTraceOk

`func (o *RecallResponse) GetTraceOk() (*map[string]interface{}, bool)`

GetTraceOk returns a tuple with the Trace field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTrace

`func (o *RecallResponse) SetTrace(v map[string]interface{})`

SetTrace sets Trace field to given value.

### HasTrace

`func (o *RecallResponse) HasTrace() bool`

HasTrace returns a boolean if a field has been set.

### SetTraceNil

`func (o *RecallResponse) SetTraceNil(b bool)`

 SetTraceNil sets the value for Trace to be an explicit nil

### UnsetTrace
`func (o *RecallResponse) UnsetTrace()`

UnsetTrace ensures that no value is present for Trace, not even an explicit nil
### GetEntities

`func (o *RecallResponse) GetEntities() map[string]EntityStateResponse`

GetEntities returns the Entities field if non-nil, zero value otherwise.

### GetEntitiesOk

`func (o *RecallResponse) GetEntitiesOk() (*map[string]EntityStateResponse, bool)`

GetEntitiesOk returns a tuple with the Entities field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetEntities

`func (o *RecallResponse) SetEntities(v map[string]EntityStateResponse)`

SetEntities sets Entities field to given value.

### HasEntities

`func (o *RecallResponse) HasEntities() bool`

HasEntities returns a boolean if a field has been set.

### SetEntitiesNil

`func (o *RecallResponse) SetEntitiesNil(b bool)`

 SetEntitiesNil sets the value for Entities to be an explicit nil

### UnsetEntities
`func (o *RecallResponse) UnsetEntities()`

UnsetEntities ensures that no value is present for Entities, not even an explicit nil
### GetChunks

`func (o *RecallResponse) GetChunks() map[string]ChunkData`

GetChunks returns the Chunks field if non-nil, zero value otherwise.

### GetChunksOk

`func (o *RecallResponse) GetChunksOk() (*map[string]ChunkData, bool)`

GetChunksOk returns a tuple with the Chunks field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetChunks

`func (o *RecallResponse) SetChunks(v map[string]ChunkData)`

SetChunks sets Chunks field to given value.

### HasChunks

`func (o *RecallResponse) HasChunks() bool`

HasChunks returns a boolean if a field has been set.

### SetChunksNil

`func (o *RecallResponse) SetChunksNil(b bool)`

 SetChunksNil sets the value for Chunks to be an explicit nil

### UnsetChunks
`func (o *RecallResponse) UnsetChunks()`

UnsetChunks ensures that no value is present for Chunks, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


