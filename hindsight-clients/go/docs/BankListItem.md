# BankListItem

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**BankId** | **string** |  | 
**Name** | Pointer to **NullableString** |  | [optional] 
**Disposition** | [**DispositionTraits**](DispositionTraits.md) |  | 
**Mission** | Pointer to **NullableString** |  | [optional] 
**CreatedAt** | Pointer to **NullableString** |  | [optional] 
**UpdatedAt** | Pointer to **NullableString** |  | [optional] 

## Methods

### NewBankListItem

`func NewBankListItem(bankId string, disposition DispositionTraits, ) *BankListItem`

NewBankListItem instantiates a new BankListItem object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewBankListItemWithDefaults

`func NewBankListItemWithDefaults() *BankListItem`

NewBankListItemWithDefaults instantiates a new BankListItem object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetBankId

`func (o *BankListItem) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *BankListItem) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *BankListItem) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetName

`func (o *BankListItem) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *BankListItem) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *BankListItem) SetName(v string)`

SetName sets Name field to given value.

### HasName

`func (o *BankListItem) HasName() bool`

HasName returns a boolean if a field has been set.

### SetNameNil

`func (o *BankListItem) SetNameNil(b bool)`

 SetNameNil sets the value for Name to be an explicit nil

### UnsetName
`func (o *BankListItem) UnsetName()`

UnsetName ensures that no value is present for Name, not even an explicit nil
### GetDisposition

`func (o *BankListItem) GetDisposition() DispositionTraits`

GetDisposition returns the Disposition field if non-nil, zero value otherwise.

### GetDispositionOk

`func (o *BankListItem) GetDispositionOk() (*DispositionTraits, bool)`

GetDispositionOk returns a tuple with the Disposition field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDisposition

`func (o *BankListItem) SetDisposition(v DispositionTraits)`

SetDisposition sets Disposition field to given value.


### GetMission

`func (o *BankListItem) GetMission() string`

GetMission returns the Mission field if non-nil, zero value otherwise.

### GetMissionOk

`func (o *BankListItem) GetMissionOk() (*string, bool)`

GetMissionOk returns a tuple with the Mission field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMission

`func (o *BankListItem) SetMission(v string)`

SetMission sets Mission field to given value.

### HasMission

`func (o *BankListItem) HasMission() bool`

HasMission returns a boolean if a field has been set.

### SetMissionNil

`func (o *BankListItem) SetMissionNil(b bool)`

 SetMissionNil sets the value for Mission to be an explicit nil

### UnsetMission
`func (o *BankListItem) UnsetMission()`

UnsetMission ensures that no value is present for Mission, not even an explicit nil
### GetCreatedAt

`func (o *BankListItem) GetCreatedAt() string`

GetCreatedAt returns the CreatedAt field if non-nil, zero value otherwise.

### GetCreatedAtOk

`func (o *BankListItem) GetCreatedAtOk() (*string, bool)`

GetCreatedAtOk returns a tuple with the CreatedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetCreatedAt

`func (o *BankListItem) SetCreatedAt(v string)`

SetCreatedAt sets CreatedAt field to given value.

### HasCreatedAt

`func (o *BankListItem) HasCreatedAt() bool`

HasCreatedAt returns a boolean if a field has been set.

### SetCreatedAtNil

`func (o *BankListItem) SetCreatedAtNil(b bool)`

 SetCreatedAtNil sets the value for CreatedAt to be an explicit nil

### UnsetCreatedAt
`func (o *BankListItem) UnsetCreatedAt()`

UnsetCreatedAt ensures that no value is present for CreatedAt, not even an explicit nil
### GetUpdatedAt

`func (o *BankListItem) GetUpdatedAt() string`

GetUpdatedAt returns the UpdatedAt field if non-nil, zero value otherwise.

### GetUpdatedAtOk

`func (o *BankListItem) GetUpdatedAtOk() (*string, bool)`

GetUpdatedAtOk returns a tuple with the UpdatedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetUpdatedAt

`func (o *BankListItem) SetUpdatedAt(v string)`

SetUpdatedAt sets UpdatedAt field to given value.

### HasUpdatedAt

`func (o *BankListItem) HasUpdatedAt() bool`

HasUpdatedAt returns a boolean if a field has been set.

### SetUpdatedAtNil

`func (o *BankListItem) SetUpdatedAtNil(b bool)`

 SetUpdatedAtNil sets the value for UpdatedAt to be an explicit nil

### UnsetUpdatedAt
`func (o *BankListItem) UnsetUpdatedAt()`

UnsetUpdatedAt ensures that no value is present for UpdatedAt, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


