# BankConfigResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**BankId** | **string** | Bank identifier | 
**Config** | **map[string]interface{}** | Fully resolved configuration with all hierarchical overrides applied (Python field names) | 
**Overrides** | **map[string]interface{}** | Bank-specific configuration overrides only (Python field names) | 

## Methods

### NewBankConfigResponse

`func NewBankConfigResponse(bankId string, config map[string]interface{}, overrides map[string]interface{}, ) *BankConfigResponse`

NewBankConfigResponse instantiates a new BankConfigResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewBankConfigResponseWithDefaults

`func NewBankConfigResponseWithDefaults() *BankConfigResponse`

NewBankConfigResponseWithDefaults instantiates a new BankConfigResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetBankId

`func (o *BankConfigResponse) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *BankConfigResponse) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *BankConfigResponse) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetConfig

`func (o *BankConfigResponse) GetConfig() map[string]interface{}`

GetConfig returns the Config field if non-nil, zero value otherwise.

### GetConfigOk

`func (o *BankConfigResponse) GetConfigOk() (*map[string]interface{}, bool)`

GetConfigOk returns a tuple with the Config field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetConfig

`func (o *BankConfigResponse) SetConfig(v map[string]interface{})`

SetConfig sets Config field to given value.


### GetOverrides

`func (o *BankConfigResponse) GetOverrides() map[string]interface{}`

GetOverrides returns the Overrides field if non-nil, zero value otherwise.

### GetOverridesOk

`func (o *BankConfigResponse) GetOverridesOk() (*map[string]interface{}, bool)`

GetOverridesOk returns a tuple with the Overrides field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOverrides

`func (o *BankConfigResponse) SetOverrides(v map[string]interface{})`

SetOverrides sets Overrides field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


