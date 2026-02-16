# RecallRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Query** | **string** |  | 
**Types** | Pointer to **[]string** |  | [optional] 
**Budget** | Pointer to [**Budget**](Budget.md) |  | [optional] 
**MaxTokens** | Pointer to **int32** |  | [optional] [default to 4096]
**Trace** | Pointer to **bool** |  | [optional] [default to false]
**QueryTimestamp** | Pointer to **NullableString** |  | [optional] 
**Include** | Pointer to [**IncludeOptions**](IncludeOptions.md) | Options for including additional data (entities are included by default) | [optional] 
**Tags** | Pointer to **[]string** |  | [optional] 
**TagsMatch** | Pointer to **string** | How to match tags: &#39;any&#39; (OR, includes untagged), &#39;all&#39; (AND, includes untagged), &#39;any_strict&#39; (OR, excludes untagged), &#39;all_strict&#39; (AND, excludes untagged). | [optional] [default to "any"]

## Methods

### NewRecallRequest

`func NewRecallRequest(query string, ) *RecallRequest`

NewRecallRequest instantiates a new RecallRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewRecallRequestWithDefaults

`func NewRecallRequestWithDefaults() *RecallRequest`

NewRecallRequestWithDefaults instantiates a new RecallRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetQuery

`func (o *RecallRequest) GetQuery() string`

GetQuery returns the Query field if non-nil, zero value otherwise.

### GetQueryOk

`func (o *RecallRequest) GetQueryOk() (*string, bool)`

GetQueryOk returns a tuple with the Query field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetQuery

`func (o *RecallRequest) SetQuery(v string)`

SetQuery sets Query field to given value.


### GetTypes

`func (o *RecallRequest) GetTypes() []string`

GetTypes returns the Types field if non-nil, zero value otherwise.

### GetTypesOk

`func (o *RecallRequest) GetTypesOk() (*[]string, bool)`

GetTypesOk returns a tuple with the Types field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTypes

`func (o *RecallRequest) SetTypes(v []string)`

SetTypes sets Types field to given value.

### HasTypes

`func (o *RecallRequest) HasTypes() bool`

HasTypes returns a boolean if a field has been set.

### SetTypesNil

`func (o *RecallRequest) SetTypesNil(b bool)`

 SetTypesNil sets the value for Types to be an explicit nil

### UnsetTypes
`func (o *RecallRequest) UnsetTypes()`

UnsetTypes ensures that no value is present for Types, not even an explicit nil
### GetBudget

`func (o *RecallRequest) GetBudget() Budget`

GetBudget returns the Budget field if non-nil, zero value otherwise.

### GetBudgetOk

`func (o *RecallRequest) GetBudgetOk() (*Budget, bool)`

GetBudgetOk returns a tuple with the Budget field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBudget

`func (o *RecallRequest) SetBudget(v Budget)`

SetBudget sets Budget field to given value.

### HasBudget

`func (o *RecallRequest) HasBudget() bool`

HasBudget returns a boolean if a field has been set.

### GetMaxTokens

`func (o *RecallRequest) GetMaxTokens() int32`

GetMaxTokens returns the MaxTokens field if non-nil, zero value otherwise.

### GetMaxTokensOk

`func (o *RecallRequest) GetMaxTokensOk() (*int32, bool)`

GetMaxTokensOk returns a tuple with the MaxTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMaxTokens

`func (o *RecallRequest) SetMaxTokens(v int32)`

SetMaxTokens sets MaxTokens field to given value.

### HasMaxTokens

`func (o *RecallRequest) HasMaxTokens() bool`

HasMaxTokens returns a boolean if a field has been set.

### GetTrace

`func (o *RecallRequest) GetTrace() bool`

GetTrace returns the Trace field if non-nil, zero value otherwise.

### GetTraceOk

`func (o *RecallRequest) GetTraceOk() (*bool, bool)`

GetTraceOk returns a tuple with the Trace field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTrace

`func (o *RecallRequest) SetTrace(v bool)`

SetTrace sets Trace field to given value.

### HasTrace

`func (o *RecallRequest) HasTrace() bool`

HasTrace returns a boolean if a field has been set.

### GetQueryTimestamp

`func (o *RecallRequest) GetQueryTimestamp() string`

GetQueryTimestamp returns the QueryTimestamp field if non-nil, zero value otherwise.

### GetQueryTimestampOk

`func (o *RecallRequest) GetQueryTimestampOk() (*string, bool)`

GetQueryTimestampOk returns a tuple with the QueryTimestamp field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetQueryTimestamp

`func (o *RecallRequest) SetQueryTimestamp(v string)`

SetQueryTimestamp sets QueryTimestamp field to given value.

### HasQueryTimestamp

`func (o *RecallRequest) HasQueryTimestamp() bool`

HasQueryTimestamp returns a boolean if a field has been set.

### SetQueryTimestampNil

`func (o *RecallRequest) SetQueryTimestampNil(b bool)`

 SetQueryTimestampNil sets the value for QueryTimestamp to be an explicit nil

### UnsetQueryTimestamp
`func (o *RecallRequest) UnsetQueryTimestamp()`

UnsetQueryTimestamp ensures that no value is present for QueryTimestamp, not even an explicit nil
### GetInclude

`func (o *RecallRequest) GetInclude() IncludeOptions`

GetInclude returns the Include field if non-nil, zero value otherwise.

### GetIncludeOk

`func (o *RecallRequest) GetIncludeOk() (*IncludeOptions, bool)`

GetIncludeOk returns a tuple with the Include field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetInclude

`func (o *RecallRequest) SetInclude(v IncludeOptions)`

SetInclude sets Include field to given value.

### HasInclude

`func (o *RecallRequest) HasInclude() bool`

HasInclude returns a boolean if a field has been set.

### GetTags

`func (o *RecallRequest) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *RecallRequest) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *RecallRequest) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *RecallRequest) HasTags() bool`

HasTags returns a boolean if a field has been set.

### SetTagsNil

`func (o *RecallRequest) SetTagsNil(b bool)`

 SetTagsNil sets the value for Tags to be an explicit nil

### UnsetTags
`func (o *RecallRequest) UnsetTags()`

UnsetTags ensures that no value is present for Tags, not even an explicit nil
### GetTagsMatch

`func (o *RecallRequest) GetTagsMatch() string`

GetTagsMatch returns the TagsMatch field if non-nil, zero value otherwise.

### GetTagsMatchOk

`func (o *RecallRequest) GetTagsMatchOk() (*string, bool)`

GetTagsMatchOk returns a tuple with the TagsMatch field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTagsMatch

`func (o *RecallRequest) SetTagsMatch(v string)`

SetTagsMatch sets TagsMatch field to given value.

### HasTagsMatch

`func (o *RecallRequest) HasTagsMatch() bool`

HasTagsMatch returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


