# ReflectIncludeOptions

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Facts** | Pointer to **map[string]interface{}** | Options for including facts (based_on) in reflect results. | [optional] 
**ToolCalls** | Pointer to [**NullableToolCallsIncludeOptions**](ToolCallsIncludeOptions.md) |  | [optional] 

## Methods

### NewReflectIncludeOptions

`func NewReflectIncludeOptions() *ReflectIncludeOptions`

NewReflectIncludeOptions instantiates a new ReflectIncludeOptions object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectIncludeOptionsWithDefaults

`func NewReflectIncludeOptionsWithDefaults() *ReflectIncludeOptions`

NewReflectIncludeOptionsWithDefaults instantiates a new ReflectIncludeOptions object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetFacts

`func (o *ReflectIncludeOptions) GetFacts() map[string]interface{}`

GetFacts returns the Facts field if non-nil, zero value otherwise.

### GetFactsOk

`func (o *ReflectIncludeOptions) GetFactsOk() (*map[string]interface{}, bool)`

GetFactsOk returns a tuple with the Facts field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetFacts

`func (o *ReflectIncludeOptions) SetFacts(v map[string]interface{})`

SetFacts sets Facts field to given value.

### HasFacts

`func (o *ReflectIncludeOptions) HasFacts() bool`

HasFacts returns a boolean if a field has been set.

### GetToolCalls

`func (o *ReflectIncludeOptions) GetToolCalls() ToolCallsIncludeOptions`

GetToolCalls returns the ToolCalls field if non-nil, zero value otherwise.

### GetToolCallsOk

`func (o *ReflectIncludeOptions) GetToolCallsOk() (*ToolCallsIncludeOptions, bool)`

GetToolCallsOk returns a tuple with the ToolCalls field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetToolCalls

`func (o *ReflectIncludeOptions) SetToolCalls(v ToolCallsIncludeOptions)`

SetToolCalls sets ToolCalls field to given value.

### HasToolCalls

`func (o *ReflectIncludeOptions) HasToolCalls() bool`

HasToolCalls returns a boolean if a field has been set.

### SetToolCallsNil

`func (o *ReflectIncludeOptions) SetToolCallsNil(b bool)`

 SetToolCallsNil sets the value for ToolCalls to be an explicit nil

### UnsetToolCalls
`func (o *ReflectIncludeOptions) UnsetToolCalls()`

UnsetToolCalls ensures that no value is present for ToolCalls, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


