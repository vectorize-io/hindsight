# ReflectFact

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | Pointer to **NullableString** |  | [optional] 
**Text** | **string** | Fact text. When type&#x3D;&#39;observation&#39;, this contains markdown-formatted consolidated knowledge | 
**Type** | Pointer to **NullableString** |  | [optional] 
**Context** | Pointer to **NullableString** |  | [optional] 
**OccurredStart** | Pointer to **NullableString** |  | [optional] 
**OccurredEnd** | Pointer to **NullableString** |  | [optional] 

## Methods

### NewReflectFact

`func NewReflectFact(text string, ) *ReflectFact`

NewReflectFact instantiates a new ReflectFact object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectFactWithDefaults

`func NewReflectFactWithDefaults() *ReflectFact`

NewReflectFactWithDefaults instantiates a new ReflectFact object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *ReflectFact) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *ReflectFact) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *ReflectFact) SetId(v string)`

SetId sets Id field to given value.

### HasId

`func (o *ReflectFact) HasId() bool`

HasId returns a boolean if a field has been set.

### SetIdNil

`func (o *ReflectFact) SetIdNil(b bool)`

 SetIdNil sets the value for Id to be an explicit nil

### UnsetId
`func (o *ReflectFact) UnsetId()`

UnsetId ensures that no value is present for Id, not even an explicit nil
### GetText

`func (o *ReflectFact) GetText() string`

GetText returns the Text field if non-nil, zero value otherwise.

### GetTextOk

`func (o *ReflectFact) GetTextOk() (*string, bool)`

GetTextOk returns a tuple with the Text field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetText

`func (o *ReflectFact) SetText(v string)`

SetText sets Text field to given value.


### GetType

`func (o *ReflectFact) GetType() string`

GetType returns the Type field if non-nil, zero value otherwise.

### GetTypeOk

`func (o *ReflectFact) GetTypeOk() (*string, bool)`

GetTypeOk returns a tuple with the Type field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetType

`func (o *ReflectFact) SetType(v string)`

SetType sets Type field to given value.

### HasType

`func (o *ReflectFact) HasType() bool`

HasType returns a boolean if a field has been set.

### SetTypeNil

`func (o *ReflectFact) SetTypeNil(b bool)`

 SetTypeNil sets the value for Type to be an explicit nil

### UnsetType
`func (o *ReflectFact) UnsetType()`

UnsetType ensures that no value is present for Type, not even an explicit nil
### GetContext

`func (o *ReflectFact) GetContext() string`

GetContext returns the Context field if non-nil, zero value otherwise.

### GetContextOk

`func (o *ReflectFact) GetContextOk() (*string, bool)`

GetContextOk returns a tuple with the Context field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContext

`func (o *ReflectFact) SetContext(v string)`

SetContext sets Context field to given value.

### HasContext

`func (o *ReflectFact) HasContext() bool`

HasContext returns a boolean if a field has been set.

### SetContextNil

`func (o *ReflectFact) SetContextNil(b bool)`

 SetContextNil sets the value for Context to be an explicit nil

### UnsetContext
`func (o *ReflectFact) UnsetContext()`

UnsetContext ensures that no value is present for Context, not even an explicit nil
### GetOccurredStart

`func (o *ReflectFact) GetOccurredStart() string`

GetOccurredStart returns the OccurredStart field if non-nil, zero value otherwise.

### GetOccurredStartOk

`func (o *ReflectFact) GetOccurredStartOk() (*string, bool)`

GetOccurredStartOk returns a tuple with the OccurredStart field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOccurredStart

`func (o *ReflectFact) SetOccurredStart(v string)`

SetOccurredStart sets OccurredStart field to given value.

### HasOccurredStart

`func (o *ReflectFact) HasOccurredStart() bool`

HasOccurredStart returns a boolean if a field has been set.

### SetOccurredStartNil

`func (o *ReflectFact) SetOccurredStartNil(b bool)`

 SetOccurredStartNil sets the value for OccurredStart to be an explicit nil

### UnsetOccurredStart
`func (o *ReflectFact) UnsetOccurredStart()`

UnsetOccurredStart ensures that no value is present for OccurredStart, not even an explicit nil
### GetOccurredEnd

`func (o *ReflectFact) GetOccurredEnd() string`

GetOccurredEnd returns the OccurredEnd field if non-nil, zero value otherwise.

### GetOccurredEndOk

`func (o *ReflectFact) GetOccurredEndOk() (*string, bool)`

GetOccurredEndOk returns a tuple with the OccurredEnd field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOccurredEnd

`func (o *ReflectFact) SetOccurredEnd(v string)`

SetOccurredEnd sets OccurredEnd field to given value.

### HasOccurredEnd

`func (o *ReflectFact) HasOccurredEnd() bool`

HasOccurredEnd returns a boolean if a field has been set.

### SetOccurredEndNil

`func (o *ReflectFact) SetOccurredEndNil(b bool)`

 SetOccurredEndNil sets the value for OccurredEnd to be an explicit nil

### UnsetOccurredEnd
`func (o *ReflectFact) UnsetOccurredEnd()`

UnsetOccurredEnd ensures that no value is present for OccurredEnd, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


