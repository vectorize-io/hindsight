# ToolCallsIncludeOptions

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Output** | Pointer to **bool** | Include tool outputs in the trace. Set to false to only include inputs (smaller payload). | [optional] [default to true]

## Methods

### NewToolCallsIncludeOptions

`func NewToolCallsIncludeOptions() *ToolCallsIncludeOptions`

NewToolCallsIncludeOptions instantiates a new ToolCallsIncludeOptions object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewToolCallsIncludeOptionsWithDefaults

`func NewToolCallsIncludeOptionsWithDefaults() *ToolCallsIncludeOptions`

NewToolCallsIncludeOptionsWithDefaults instantiates a new ToolCallsIncludeOptions object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetOutput

`func (o *ToolCallsIncludeOptions) GetOutput() bool`

GetOutput returns the Output field if non-nil, zero value otherwise.

### GetOutputOk

`func (o *ToolCallsIncludeOptions) GetOutputOk() (*bool, bool)`

GetOutputOk returns a tuple with the Output field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOutput

`func (o *ToolCallsIncludeOptions) SetOutput(v bool)`

SetOutput sets Output field to given value.

### HasOutput

`func (o *ToolCallsIncludeOptions) HasOutput() bool`

HasOutput returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


