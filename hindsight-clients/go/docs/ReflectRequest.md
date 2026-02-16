# ReflectRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Query** | **string** |  | 
**Budget** | Pointer to [**Budget**](Budget.md) |  | [optional] 
**Context** | Pointer to **NullableString** |  | [optional] 
**MaxTokens** | Pointer to **int32** | Maximum tokens for the response | [optional] [default to 4096]
**Include** | Pointer to [**ReflectIncludeOptions**](ReflectIncludeOptions.md) | Options for including additional data (disabled by default) | [optional] 
**ResponseSchema** | Pointer to **map[string]interface{}** |  | [optional] 
**Tags** | Pointer to **[]string** |  | [optional] 
**TagsMatch** | Pointer to **string** | How to match tags: &#39;any&#39; (OR, includes untagged), &#39;all&#39; (AND, includes untagged), &#39;any_strict&#39; (OR, excludes untagged), &#39;all_strict&#39; (AND, excludes untagged). | [optional] [default to "any"]

## Methods

### NewReflectRequest

`func NewReflectRequest(query string, ) *ReflectRequest`

NewReflectRequest instantiates a new ReflectRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewReflectRequestWithDefaults

`func NewReflectRequestWithDefaults() *ReflectRequest`

NewReflectRequestWithDefaults instantiates a new ReflectRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetQuery

`func (o *ReflectRequest) GetQuery() string`

GetQuery returns the Query field if non-nil, zero value otherwise.

### GetQueryOk

`func (o *ReflectRequest) GetQueryOk() (*string, bool)`

GetQueryOk returns a tuple with the Query field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetQuery

`func (o *ReflectRequest) SetQuery(v string)`

SetQuery sets Query field to given value.


### GetBudget

`func (o *ReflectRequest) GetBudget() Budget`

GetBudget returns the Budget field if non-nil, zero value otherwise.

### GetBudgetOk

`func (o *ReflectRequest) GetBudgetOk() (*Budget, bool)`

GetBudgetOk returns a tuple with the Budget field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBudget

`func (o *ReflectRequest) SetBudget(v Budget)`

SetBudget sets Budget field to given value.

### HasBudget

`func (o *ReflectRequest) HasBudget() bool`

HasBudget returns a boolean if a field has been set.

### GetContext

`func (o *ReflectRequest) GetContext() string`

GetContext returns the Context field if non-nil, zero value otherwise.

### GetContextOk

`func (o *ReflectRequest) GetContextOk() (*string, bool)`

GetContextOk returns a tuple with the Context field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContext

`func (o *ReflectRequest) SetContext(v string)`

SetContext sets Context field to given value.

### HasContext

`func (o *ReflectRequest) HasContext() bool`

HasContext returns a boolean if a field has been set.

### SetContextNil

`func (o *ReflectRequest) SetContextNil(b bool)`

 SetContextNil sets the value for Context to be an explicit nil

### UnsetContext
`func (o *ReflectRequest) UnsetContext()`

UnsetContext ensures that no value is present for Context, not even an explicit nil
### GetMaxTokens

`func (o *ReflectRequest) GetMaxTokens() int32`

GetMaxTokens returns the MaxTokens field if non-nil, zero value otherwise.

### GetMaxTokensOk

`func (o *ReflectRequest) GetMaxTokensOk() (*int32, bool)`

GetMaxTokensOk returns a tuple with the MaxTokens field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMaxTokens

`func (o *ReflectRequest) SetMaxTokens(v int32)`

SetMaxTokens sets MaxTokens field to given value.

### HasMaxTokens

`func (o *ReflectRequest) HasMaxTokens() bool`

HasMaxTokens returns a boolean if a field has been set.

### GetInclude

`func (o *ReflectRequest) GetInclude() ReflectIncludeOptions`

GetInclude returns the Include field if non-nil, zero value otherwise.

### GetIncludeOk

`func (o *ReflectRequest) GetIncludeOk() (*ReflectIncludeOptions, bool)`

GetIncludeOk returns a tuple with the Include field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetInclude

`func (o *ReflectRequest) SetInclude(v ReflectIncludeOptions)`

SetInclude sets Include field to given value.

### HasInclude

`func (o *ReflectRequest) HasInclude() bool`

HasInclude returns a boolean if a field has been set.

### GetResponseSchema

`func (o *ReflectRequest) GetResponseSchema() map[string]interface{}`

GetResponseSchema returns the ResponseSchema field if non-nil, zero value otherwise.

### GetResponseSchemaOk

`func (o *ReflectRequest) GetResponseSchemaOk() (*map[string]interface{}, bool)`

GetResponseSchemaOk returns a tuple with the ResponseSchema field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetResponseSchema

`func (o *ReflectRequest) SetResponseSchema(v map[string]interface{})`

SetResponseSchema sets ResponseSchema field to given value.

### HasResponseSchema

`func (o *ReflectRequest) HasResponseSchema() bool`

HasResponseSchema returns a boolean if a field has been set.

### SetResponseSchemaNil

`func (o *ReflectRequest) SetResponseSchemaNil(b bool)`

 SetResponseSchemaNil sets the value for ResponseSchema to be an explicit nil

### UnsetResponseSchema
`func (o *ReflectRequest) UnsetResponseSchema()`

UnsetResponseSchema ensures that no value is present for ResponseSchema, not even an explicit nil
### GetTags

`func (o *ReflectRequest) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *ReflectRequest) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *ReflectRequest) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *ReflectRequest) HasTags() bool`

HasTags returns a boolean if a field has been set.

### SetTagsNil

`func (o *ReflectRequest) SetTagsNil(b bool)`

 SetTagsNil sets the value for Tags to be an explicit nil

### UnsetTags
`func (o *ReflectRequest) UnsetTags()`

UnsetTags ensures that no value is present for Tags, not even an explicit nil
### GetTagsMatch

`func (o *ReflectRequest) GetTagsMatch() string`

GetTagsMatch returns the TagsMatch field if non-nil, zero value otherwise.

### GetTagsMatchOk

`func (o *ReflectRequest) GetTagsMatchOk() (*string, bool)`

GetTagsMatchOk returns a tuple with the TagsMatch field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTagsMatch

`func (o *ReflectRequest) SetTagsMatch(v string)`

SetTagsMatch sets TagsMatch field to given value.

### HasTagsMatch

`func (o *ReflectRequest) HasTagsMatch() bool`

HasTagsMatch returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


