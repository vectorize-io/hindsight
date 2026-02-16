# ReflectLLMCall

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Scope** | **string** | Call scope: agent_1, agent_2, final, etc. | 
**DurationMs** | **int32** | Execution time in milliseconds | 

## Methods

### NewReflectLLMCall

`func NewReflectLLMCall(scope string, durationMs int32, ) *ReflectLLMCall`

NewReflectLLMCall instantiates a new ReflectLLMCall object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectLLMCallWithDefaults

`func NewReflectLLMCallWithDefaults() *ReflectLLMCall`

NewReflectLLMCallWithDefaults instantiates a new ReflectLLMCall object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetScope

`func (o *ReflectLLMCall) GetScope() string`

GetScope returns the Scope field if non-nil, zero value otherwise.

### GetScopeOk

`func (o *ReflectLLMCall) GetScopeOk() (*string, bool)`

GetScopeOk returns a tuple with the Scope field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetScope

`func (o *ReflectLLMCall) SetScope(v string)`

SetScope sets Scope field to given value.


### GetDurationMs

`func (o *ReflectLLMCall) GetDurationMs() int32`

GetDurationMs returns the DurationMs field if non-nil, zero value otherwise.

### GetDurationMsOk

`func (o *ReflectLLMCall) GetDurationMsOk() (*int32, bool)`

GetDurationMsOk returns a tuple with the DurationMs field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDurationMs

`func (o *ReflectLLMCall) SetDurationMs(v int32)`

SetDurationMs sets DurationMs field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


