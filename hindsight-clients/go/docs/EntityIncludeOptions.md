# EntityIncludeOptions

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**MaxTokens** | Pointer to **int32** | Maximum tokens for entity observations | [optional] [default to 500]

## Methods

### NewEntityIncludeOptions

`func NewEntityIncludeOptions() *EntityIncludeOptions`

NewEntityIncludeOptions instantiates a new EntityIncludeOptions object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewEntityIncludeOptionsWithDefaults

`func NewEntityIncludeOptionsWithDefaults() *EntityIncludeOptions`

NewEntityIncludeOptionsWithDefaults instantiates a new EntityIncludeOptions object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetMaxTokens

`func (o *EntityIncludeOptions) GetMaxTokens() int32`

GetMaxTokens returns the MaxTokens field if non-nil, zero value otherwise.

### GetMaxTokensOk

`func (o *EntityIncludeOptions) GetMaxTokensOk() (*int32, bool)`

GetMaxTokensOk returns a tuple with the MaxTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMaxTokens

`func (o *EntityIncludeOptions) SetMaxTokens(v int32)`

SetMaxTokens sets MaxTokens field to given value.

### HasMaxTokens

`func (o *EntityIncludeOptions) HasMaxTokens() bool`

HasMaxTokens returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


