# ReflectResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Text** | **string** | The reflect response as well-formatted markdown (headers, lists, bold/italic, code blocks, etc.) | 
**BasedOn** | Pointer to [**NullableReflectBasedOn**](ReflectBasedOn.md) |  | [optional] 
**StructuredOutput** | Pointer to **map[string]interface{}** |  | [optional] 
**Usage** | Pointer to [**NullableTokenUsage**](TokenUsage.md) |  | [optional] 
**Trace** | Pointer to [**NullableReflectTrace**](ReflectTrace.md) |  | [optional] 

## Methods

### NewReflectResponse

`func NewReflectResponse(text string, ) *ReflectResponse`

NewReflectResponse instantiates a new ReflectResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectResponseWithDefaults

`func NewReflectResponseWithDefaults() *ReflectResponse`

NewReflectResponseWithDefaults instantiates a new ReflectResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetText

`func (o *ReflectResponse) GetText() string`

GetText returns the Text field if non-nil, zero value otherwise.

### GetTextOk

`func (o *ReflectResponse) GetTextOk() (*string, bool)`

GetTextOk returns a tuple with the Text field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetText

`func (o *ReflectResponse) SetText(v string)`

SetText sets Text field to given value.


### GetBasedOn

`func (o *ReflectResponse) GetBasedOn() ReflectBasedOn`

GetBasedOn returns the BasedOn field if non-nil, zero value otherwise.

### GetBasedOnOk

`func (o *ReflectResponse) GetBasedOnOk() (*ReflectBasedOn, bool)`

GetBasedOnOk returns a tuple with the BasedOn field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBasedOn

`func (o *ReflectResponse) SetBasedOn(v ReflectBasedOn)`

SetBasedOn sets BasedOn field to given value.

### HasBasedOn

`func (o *ReflectResponse) HasBasedOn() bool`

HasBasedOn returns a boolean if a field has been set.

### SetBasedOnNil

`func (o *ReflectResponse) SetBasedOnNil(b bool)`

 SetBasedOnNil sets the value for BasedOn to be an explicit nil

### UnsetBasedOn
`func (o *ReflectResponse) UnsetBasedOn()`

UnsetBasedOn ensures that no value is present for BasedOn, not even an explicit nil
### GetStructuredOutput

`func (o *ReflectResponse) GetStructuredOutput() map[string]interface{}`

GetStructuredOutput returns the StructuredOutput field if non-nil, zero value otherwise.

### GetStructuredOutputOk

`func (o *ReflectResponse) GetStructuredOutputOk() (*map[string]interface{}, bool)`

GetStructuredOutputOk returns a tuple with the StructuredOutput field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetStructuredOutput

`func (o *ReflectResponse) SetStructuredOutput(v map[string]interface{})`

SetStructuredOutput sets StructuredOutput field to given value.

### HasStructuredOutput

`func (o *ReflectResponse) HasStructuredOutput() bool`

HasStructuredOutput returns a boolean if a field has been set.

### SetStructuredOutputNil

`func (o *ReflectResponse) SetStructuredOutputNil(b bool)`

 SetStructuredOutputNil sets the value for StructuredOutput to be an explicit nil

### UnsetStructuredOutput
`func (o *ReflectResponse) UnsetStructuredOutput()`

UnsetStructuredOutput ensures that no value is present for StructuredOutput, not even an explicit nil
### GetUsage

`func (o *ReflectResponse) GetUsage() TokenUsage`

GetUsage returns the Usage field if non-nil, zero value otherwise.

### GetUsageOk

`func (o *ReflectResponse) GetUsageOk() (*TokenUsage, bool)`

GetUsageOk returns a tuple with the Usage field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetUsage

`func (o *ReflectResponse) SetUsage(v TokenUsage)`

SetUsage sets Usage field to given value.

### HasUsage

`func (o *ReflectResponse) HasUsage() bool`

HasUsage returns a boolean if a field has been set.

### SetUsageNil

`func (o *ReflectResponse) SetUsageNil(b bool)`

 SetUsageNil sets the value for Usage to be an explicit nil

### UnsetUsage
`func (o *ReflectResponse) UnsetUsage()`

UnsetUsage ensures that no value is present for Usage, not even an explicit nil
### GetTrace

`func (o *ReflectResponse) GetTrace() ReflectTrace`

GetTrace returns the Trace field if non-nil, zero value otherwise.

### GetTraceOk

`func (o *ReflectResponse) GetTraceOk() (*ReflectTrace, bool)`

GetTraceOk returns a tuple with the Trace field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTrace

`func (o *ReflectResponse) SetTrace(v ReflectTrace)`

SetTrace sets Trace field to given value.

### HasTrace

`func (o *ReflectResponse) HasTrace() bool`

HasTrace returns a boolean if a field has been set.

### SetTraceNil

`func (o *ReflectResponse) SetTraceNil(b bool)`

 SetTraceNil sets the value for Trace to be an explicit nil

### UnsetTrace
`func (o *ReflectResponse) UnsetTrace()`

UnsetTrace ensures that no value is present for Trace, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


