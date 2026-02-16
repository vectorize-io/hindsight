# BankConfigUpdate

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Updates** | **map[string]interface{}** | Configuration overrides. Keys can be in Python field format (llm_provider) or environment variable format (HINDSIGHT_API_LLM_PROVIDER). Only hierarchical fields can be overridden per-bank. | 

## Methods

### NewBankConfigUpdate

`func NewBankConfigUpdate(updates map[string]interface{}, ) *BankConfigUpdate`

NewBankConfigUpdate instantiates a new BankConfigUpdate object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewBankConfigUpdateWithDefaults

`func NewBankConfigUpdateWithDefaults() *BankConfigUpdate`

NewBankConfigUpdateWithDefaults instantiates a new BankConfigUpdate object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetUpdates

`func (o *BankConfigUpdate) GetUpdates() map[string]interface{}`

GetUpdates returns the Updates field if non-nil, zero value otherwise.

### GetUpdatesOk

`func (o *BankConfigUpdate) GetUpdatesOk() (*map[string]interface{}, bool)`

GetUpdatesOk returns a tuple with the Updates field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetUpdates

`func (o *BankConfigUpdate) SetUpdates(v map[string]interface{})`

SetUpdates sets Updates field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


