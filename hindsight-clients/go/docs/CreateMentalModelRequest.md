# CreateMentalModelRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | Pointer to **NullableString** |  | [optional] 
**Name** | **string** | Human-readable name for the mental model | 
**SourceQuery** | **string** | The query to run to generate content | 
**Tags** | Pointer to **[]string** | Tags for scoped visibility | [optional] [default to []]
**MaxTokens** | Pointer to **int32** | Maximum tokens for generated content | [optional] [default to 2048]
**Trigger** | Pointer to [**MentalModelTrigger**](MentalModelTrigger.md) | Trigger settings | [optional] 

## Methods

### NewCreateMentalModelRequest

`func NewCreateMentalModelRequest(name string, sourceQuery string, ) *CreateMentalModelRequest`

NewCreateMentalModelRequest instantiates a new CreateMentalModelRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewCreateMentalModelRequestWithDefaults

`func NewCreateMentalModelRequestWithDefaults() *CreateMentalModelRequest`

NewCreateMentalModelRequestWithDefaults instantiates a new CreateMentalModelRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *CreateMentalModelRequest) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *CreateMentalModelRequest) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *CreateMentalModelRequest) SetId(v string)`

SetId sets Id field to given value.

### HasId

`func (o *CreateMentalModelRequest) HasId() bool`

HasId returns a boolean if a field has been set.

### SetIdNil

`func (o *CreateMentalModelRequest) SetIdNil(b bool)`

 SetIdNil sets the value for Id to be an explicit nil

### UnsetId
`func (o *CreateMentalModelRequest) UnsetId()`

UnsetId ensures that no value is present for Id, not even an explicit nil
### GetName

`func (o *CreateMentalModelRequest) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *CreateMentalModelRequest) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *CreateMentalModelRequest) SetName(v string)`

SetName sets Name field to given value.


### GetSourceQuery

`func (o *CreateMentalModelRequest) GetSourceQuery() string`

GetSourceQuery returns the SourceQuery field if non-nil, zero value otherwise.

### GetSourceQueryOk

`func (o *CreateMentalModelRequest) GetSourceQueryOk() (*string, bool)`

GetSourceQueryOk returns a tuple with the SourceQuery field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetSourceQuery

`func (o *CreateMentalModelRequest) SetSourceQuery(v string)`

SetSourceQuery sets SourceQuery field to given value.


### GetTags

`func (o *CreateMentalModelRequest) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *CreateMentalModelRequest) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *CreateMentalModelRequest) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *CreateMentalModelRequest) HasTags() bool`

HasTags returns a boolean if a field has been set.

### GetMaxTokens

`func (o *CreateMentalModelRequest) GetMaxTokens() int32`

GetMaxTokens returns the MaxTokens field if non-nil, zero value otherwise.

### GetMaxTokensOk

`func (o *CreateMentalModelRequest) GetMaxTokensOk() (*int32, bool)`

GetMaxTokensOk returns a tuple with the MaxTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMaxTokens

`func (o *CreateMentalModelRequest) SetMaxTokens(v int32)`

SetMaxTokens sets MaxTokens field to given value.

### HasMaxTokens

`func (o *CreateMentalModelRequest) HasMaxTokens() bool`

HasMaxTokens returns a boolean if a field has been set.

### GetTrigger

`func (o *CreateMentalModelRequest) GetTrigger() MentalModelTrigger`

GetTrigger returns the Trigger field if non-nil, zero value otherwise.

### GetTriggerOk

`func (o *CreateMentalModelRequest) GetTriggerOk() (*MentalModelTrigger, bool)`

GetTriggerOk returns a tuple with the Trigger field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTrigger

`func (o *CreateMentalModelRequest) SetTrigger(v MentalModelTrigger)`

SetTrigger sets Trigger field to given value.

### HasTrigger

`func (o *CreateMentalModelRequest) HasTrigger() bool`

HasTrigger returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


