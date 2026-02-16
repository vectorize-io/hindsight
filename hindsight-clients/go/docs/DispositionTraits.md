# DispositionTraits

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Skepticism** | **int32** | How skeptical vs trusting (1&#x3D;trusting, 5&#x3D;skeptical) | 
**Literalism** | **int32** | How literally to interpret information (1&#x3D;flexible, 5&#x3D;literal) | 
**Empathy** | **int32** | How much to consider emotional context (1&#x3D;detached, 5&#x3D;empathetic) | 

## Methods

### NewDispositionTraits

`func NewDispositionTraits(skepticism int32, literalism int32, empathy int32, ) *DispositionTraits`

NewDispositionTraits instantiates a new DispositionTraits object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewDispositionTraitsWithDefaults

`func NewDispositionTraitsWithDefaults() *DispositionTraits`

NewDispositionTraitsWithDefaults instantiates a new DispositionTraits object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetSkepticism

`func (o *DispositionTraits) GetSkepticism() int32`

GetSkepticism returns the Skepticism field if non-nil, zero value otherwise.

### GetSkepticismOk

`func (o *DispositionTraits) GetSkepticismOk() (*int32, bool)`

GetSkepticismOk returns a tuple with the Skepticism field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetSkepticism

`func (o *DispositionTraits) SetSkepticism(v int32)`

SetSkepticism sets Skepticism field to given value.


### GetLiteralism

`func (o *DispositionTraits) GetLiteralism() int32`

GetLiteralism returns the Literalism field if non-nil, zero value otherwise.

### GetLiteralismOk

`func (o *DispositionTraits) GetLiteralismOk() (*int32, bool)`

GetLiteralismOk returns a tuple with the Literalism field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLiteralism

`func (o *DispositionTraits) SetLiteralism(v int32)`

SetLiteralism sets Literalism field to given value.


### GetEmpathy

`func (o *DispositionTraits) GetEmpathy() int32`

GetEmpathy returns the Empathy field if non-nil, zero value otherwise.

### GetEmpathyOk

`func (o *DispositionTraits) GetEmpathyOk() (*int32, bool)`

GetEmpathyOk returns a tuple with the Empathy field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetEmpathy

`func (o *DispositionTraits) SetEmpathy(v int32)`

SetEmpathy sets Empathy field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


