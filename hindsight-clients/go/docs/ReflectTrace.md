# ReflectTrace

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ToolCalls** | Pointer to [**[]ReflectToolCall**](ReflectToolCall.md) | Tool calls made during reflection | [optional] [default to []]
**LlmCalls** | Pointer to [**[]ReflectLLMCall**](ReflectLLMCall.md) | LLM calls made during reflection | [optional] [default to []]

## Methods

### NewReflectTrace

`func NewReflectTrace() *ReflectTrace`

NewReflectTrace instantiates a new ReflectTrace object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectTraceWithDefaults

`func NewReflectTraceWithDefaults() *ReflectTrace`

NewReflectTraceWithDefaults instantiates a new ReflectTrace object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetToolCalls

`func (o *ReflectTrace) GetToolCalls() []ReflectToolCall`

GetToolCalls returns the ToolCalls field if non-nil, zero value otherwise.

### GetToolCallsOk

`func (o *ReflectTrace) GetToolCallsOk() (*[]ReflectToolCall, bool)`

GetToolCallsOk returns a tuple with the ToolCalls field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetToolCalls

`func (o *ReflectTrace) SetToolCalls(v []ReflectToolCall)`

SetToolCalls sets ToolCalls field to given value.

### HasToolCalls

`func (o *ReflectTrace) HasToolCalls() bool`

HasToolCalls returns a boolean if a field has been set.

### GetLlmCalls

`func (o *ReflectTrace) GetLlmCalls() []ReflectLLMCall`

GetLlmCalls returns the LlmCalls field if non-nil, zero value otherwise.

### GetLlmCallsOk

`func (o *ReflectTrace) GetLlmCallsOk() (*[]ReflectLLMCall, bool)`

GetLlmCallsOk returns a tuple with the LlmCalls field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLlmCalls

`func (o *ReflectTrace) SetLlmCalls(v []ReflectLLMCall)`

SetLlmCalls sets LlmCalls field to given value.

### HasLlmCalls

`func (o *ReflectTrace) HasLlmCalls() bool`

HasLlmCalls returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


