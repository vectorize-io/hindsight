# ReflectMentalModel

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | **string** | Mental model ID | 
**Text** | **string** | Mental model content | 
**Context** | Pointer to **NullableString** |  | [optional] 

## Methods

### NewReflectMentalModel

`func NewReflectMentalModel(id string, text string, ) *ReflectMentalModel`

NewReflectMentalModel instantiates a new ReflectMentalModel object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectMentalModelWithDefaults

`func NewReflectMentalModelWithDefaults() *ReflectMentalModel`

NewReflectMentalModelWithDefaults instantiates a new ReflectMentalModel object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *ReflectMentalModel) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *ReflectMentalModel) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *ReflectMentalModel) SetId(v string)`

SetId sets Id field to given value.


### GetText

`func (o *ReflectMentalModel) GetText() string`

GetText returns the Text field if non-nil, zero value otherwise.

### GetTextOk

`func (o *ReflectMentalModel) GetTextOk() (*string, bool)`

GetTextOk returns a tuple with the Text field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetText

`func (o *ReflectMentalModel) SetText(v string)`

SetText sets Text field to given value.


### GetContext

`func (o *ReflectMentalModel) GetContext() string`

GetContext returns the Context field if non-nil, zero value otherwise.

### GetContextOk

`func (o *ReflectMentalModel) GetContextOk() (*string, bool)`

GetContextOk returns a tuple with the Context field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContext

`func (o *ReflectMentalModel) SetContext(v string)`

SetContext sets Context field to given value.

### HasContext

`func (o *ReflectMentalModel) HasContext() bool`

HasContext returns a boolean if a field has been set.

### SetContextNil

`func (o *ReflectMentalModel) SetContextNil(b bool)`

 SetContextNil sets the value for Context to be an explicit nil

### UnsetContext
`func (o *ReflectMentalModel) UnsetContext()`

UnsetContext ensures that no value is present for Context, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


