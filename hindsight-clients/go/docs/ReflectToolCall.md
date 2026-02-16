# ReflectToolCall

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Tool** | **string** | Tool name: lookup, recall, learn, expand | 
**Input** | **map[string]interface{}** | Tool input parameters | 
**Output** | Pointer to **map[string]interface{}** |  | [optional] 
**DurationMs** | **int32** | Execution time in milliseconds | 
**Iteration** | Pointer to **int32** | Iteration number (1-based) when this tool was called | [optional] [default to 0]

## Methods

### NewReflectToolCall

`func NewReflectToolCall(tool string, input map[string]interface{}, durationMs int32, ) *ReflectToolCall`

NewReflectToolCall instantiates a new ReflectToolCall object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectToolCallWithDefaults

`func NewReflectToolCallWithDefaults() *ReflectToolCall`

NewReflectToolCallWithDefaults instantiates a new ReflectToolCall object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetTool

`func (o *ReflectToolCall) GetTool() string`

GetTool returns the Tool field if non-nil, zero value otherwise.

### GetToolOk

`func (o *ReflectToolCall) GetToolOk() (*string, bool)`

GetToolOk returns a tuple with the Tool field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTool

`func (o *ReflectToolCall) SetTool(v string)`

SetTool sets Tool field to given value.


### GetInput

`func (o *ReflectToolCall) GetInput() map[string]interface{}`

GetInput returns the Input field if non-nil, zero value otherwise.

### GetInputOk

`func (o *ReflectToolCall) GetInputOk() (*map[string]interface{}, bool)`

GetInputOk returns a tuple with the Input field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetInput

`func (o *ReflectToolCall) SetInput(v map[string]interface{})`

SetInput sets Input field to given value.


### GetOutput

`func (o *ReflectToolCall) GetOutput() map[string]interface{}`

GetOutput returns the Output field if non-nil, zero value otherwise.

### GetOutputOk

`func (o *ReflectToolCall) GetOutputOk() (*map[string]interface{}, bool)`

GetOutputOk returns a tuple with the Output field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOutput

`func (o *ReflectToolCall) SetOutput(v map[string]interface{})`

SetOutput sets Output field to given value.

### HasOutput

`func (o *ReflectToolCall) HasOutput() bool`

HasOutput returns a boolean if a field has been set.

### SetOutputNil

`func (o *ReflectToolCall) SetOutputNil(b bool)`

 SetOutputNil sets the value for Output to be an explicit nil

### UnsetOutput
`func (o *ReflectToolCall) UnsetOutput()`

UnsetOutput ensures that no value is present for Output, not even an explicit nil
### GetDurationMs

`func (o *ReflectToolCall) GetDurationMs() int32`

GetDurationMs returns the DurationMs field if non-nil, zero value otherwise.

### GetDurationMsOk

`func (o *ReflectToolCall) GetDurationMsOk() (*int32, bool)`

GetDurationMsOk returns a tuple with the DurationMs field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDurationMs

`func (o *ReflectToolCall) SetDurationMs(v int32)`

SetDurationMs sets DurationMs field to given value.


### GetIteration

`func (o *ReflectToolCall) GetIteration() int32`

GetIteration returns the Iteration field if non-nil, zero value otherwise.

### GetIterationOk

`func (o *ReflectToolCall) GetIterationOk() (*int32, bool)`

GetIterationOk returns a tuple with the Iteration field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetIteration

`func (o *ReflectToolCall) SetIteration(v int32)`

SetIteration sets Iteration field to given value.

### HasIteration

`func (o *ReflectToolCall) HasIteration() bool`

HasIteration returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


