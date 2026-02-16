# EntityInput

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Text** | **string** | The entity name/text | 
**Type** | Pointer to **NullableString** |  | [optional] 

## Methods

### NewEntityInput

`func NewEntityInput(text string, ) *EntityInput`

NewEntityInput instantiates a new EntityInput object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewEntityInputWithDefaults

`func NewEntityInputWithDefaults() *EntityInput`

NewEntityInputWithDefaults instantiates a new EntityInput object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetText

`func (o *EntityInput) GetText() string`

GetText returns the Text field if non-nil, zero value otherwise.

### GetTextOk

`func (o *EntityInput) GetTextOk() (*string, bool)`

GetTextOk returns a tuple with the Text field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetText

`func (o *EntityInput) SetText(v string)`

SetText sets Text field to given value.


### GetType

`func (o *EntityInput) GetType() string`

GetType returns the Type field if non-nil, zero value otherwise.

### GetTypeOk

`func (o *EntityInput) GetTypeOk() (*string, bool)`

GetTypeOk returns a tuple with the Type field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetType

`func (o *EntityInput) SetType(v string)`

SetType sets Type field to given value.

### HasType

`func (o *EntityInput) HasType() bool`

HasType returns a boolean if a field has been set.

### SetTypeNil

`func (o *EntityInput) SetTypeNil(b bool)`

 SetTypeNil sets the value for Type to be an explicit nil

### UnsetType
`func (o *EntityInput) UnsetType()`

UnsetType ensures that no value is present for Type, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


