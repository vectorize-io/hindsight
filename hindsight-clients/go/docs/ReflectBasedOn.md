# ReflectBasedOn

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Memories** | Pointer to [**[]ReflectFact**](ReflectFact.md) | Memory facts used to generate the response | [optional] [default to []]
**MentalModels** | Pointer to [**[]ReflectMentalModel**](ReflectMentalModel.md) | Mental models used during reflection | [optional] [default to []]
**Directives** | Pointer to [**[]ReflectDirective**](ReflectDirective.md) | Directives applied during reflection | [optional] [default to []]

## Methods

### NewReflectBasedOn

`func NewReflectBasedOn() *ReflectBasedOn`

NewReflectBasedOn instantiates a new ReflectBasedOn object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectBasedOnWithDefaults

`func NewReflectBasedOnWithDefaults() *ReflectBasedOn`

NewReflectBasedOnWithDefaults instantiates a new ReflectBasedOn object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetMemories

`func (o *ReflectBasedOn) GetMemories() []ReflectFact`

GetMemories returns the Memories field if non-nil, zero value otherwise.

### GetMemoriesOk

`func (o *ReflectBasedOn) GetMemoriesOk() (*[]ReflectFact, bool)`

GetMemoriesOk returns a tuple with the Memories field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMemories

`func (o *ReflectBasedOn) SetMemories(v []ReflectFact)`

SetMemories sets Memories field to given value.

### HasMemories

`func (o *ReflectBasedOn) HasMemories() bool`

HasMemories returns a boolean if a field has been set.

### GetMentalModels

`func (o *ReflectBasedOn) GetMentalModels() []ReflectMentalModel`

GetMentalModels returns the MentalModels field if non-nil, zero value otherwise.

### GetMentalModelsOk

`func (o *ReflectBasedOn) GetMentalModelsOk() (*[]ReflectMentalModel, bool)`

GetMentalModelsOk returns a tuple with the MentalModels field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMentalModels

`func (o *ReflectBasedOn) SetMentalModels(v []ReflectMentalModel)`

SetMentalModels sets MentalModels field to given value.

### HasMentalModels

`func (o *ReflectBasedOn) HasMentalModels() bool`

HasMentalModels returns a boolean if a field has been set.

### GetDirectives

`func (o *ReflectBasedOn) GetDirectives() []ReflectDirective`

GetDirectives returns the Directives field if non-nil, zero value otherwise.

### GetDirectivesOk

`func (o *ReflectBasedOn) GetDirectivesOk() (*[]ReflectDirective, bool)`

GetDirectivesOk returns a tuple with the Directives field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDirectives

`func (o *ReflectBasedOn) SetDirectives(v []ReflectDirective)`

SetDirectives sets Directives field to given value.

### HasDirectives

`func (o *ReflectBasedOn) HasDirectives() bool`

HasDirectives returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


