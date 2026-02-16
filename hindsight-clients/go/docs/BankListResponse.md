# BankListResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Banks** | [**[]BankListItem**](BankListItem.md) |  | 

## Methods

### NewBankListResponse

`func NewBankListResponse(banks []BankListItem, ) *BankListResponse`

NewBankListResponse instantiates a new BankListResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewBankListResponseWithDefaults

`func NewBankListResponseWithDefaults() *BankListResponse`

NewBankListResponseWithDefaults instantiates a new BankListResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetBanks

`func (o *BankListResponse) GetBanks() []BankListItem`

GetBanks returns the Banks field if non-nil, zero value otherwise.

### GetBanksOk

`func (o *BankListResponse) GetBanksOk() (*[]BankListItem, bool)`

GetBanksOk returns a tuple with the Banks field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBanks

`func (o *BankListResponse) SetBanks(v []BankListItem)`

SetBanks sets Banks field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


