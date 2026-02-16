# TokenUsage

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**InputTokens** | Pointer to **int32** | Number of input/prompt tokens consumed | [optional] [default to 0]
**OutputTokens** | Pointer to **int32** | Number of output/completion tokens generated | [optional] [default to 0]
**TotalTokens** | Pointer to **int32** | Total tokens (input + output) | [optional] [default to 0]

## Methods

### NewTokenUsage

`func NewTokenUsage() *TokenUsage`

NewTokenUsage instantiates a new TokenUsage object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewTokenUsageWithDefaults

`func NewTokenUsageWithDefaults() *TokenUsage`

NewTokenUsageWithDefaults instantiates a new TokenUsage object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetInputTokens

`func (o *TokenUsage) GetInputTokens() int32`

GetInputTokens returns the InputTokens field if non-nil, zero value otherwise.

### GetInputTokensOk

`func (o *TokenUsage) GetInputTokensOk() (*int32, bool)`

GetInputTokensOk returns a tuple with the InputTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetInputTokens

`func (o *TokenUsage) SetInputTokens(v int32)`

SetInputTokens sets InputTokens field to given value.

### HasInputTokens

`func (o *TokenUsage) HasInputTokens() bool`

HasInputTokens returns a boolean if a field has been set.

### GetOutputTokens

`func (o *TokenUsage) GetOutputTokens() int32`

GetOutputTokens returns the OutputTokens field if non-nil, zero value otherwise.

### GetOutputTokensOk

`func (o *TokenUsage) GetOutputTokensOk() (*int32, bool)`

GetOutputTokensOk returns a tuple with the OutputTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOutputTokens

`func (o *TokenUsage) SetOutputTokens(v int32)`

SetOutputTokens sets OutputTokens field to given value.

### HasOutputTokens

`func (o *TokenUsage) HasOutputTokens() bool`

HasOutputTokens returns a boolean if a field has been set.

### GetTotalTokens

`func (o *TokenUsage) GetTotalTokens() int32`

GetTotalTokens returns the TotalTokens field if non-nil, zero value otherwise.

### GetTotalTokensOk

`func (o *TokenUsage) GetTotalTokensOk() (*int32, bool)`

GetTotalTokensOk returns a tuple with the TotalTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotalTokens

`func (o *TokenUsage) SetTotalTokens(v int32)`

SetTotalTokens sets TotalTokens field to given value.

### HasTotalTokens

`func (o *TokenUsage) HasTotalTokens() bool`

HasTotalTokens returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


