# BankProfileResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**BankId** | **string** |  | 
**Name** | **string** |  | 
**Disposition** | [**DispositionTraits**](DispositionTraits.md) |  | 
**Mission** | **string** | The agent&#39;s mission - who they are and what they&#39;re trying to accomplish | 
**Background** | Pointer to **NullableString** |  | [optional] 

## Methods

### NewBankProfileResponse

`func NewBankProfileResponse(bankId string, name string, disposition DispositionTraits, mission string, ) *BankProfileResponse`

NewBankProfileResponse instantiates a new BankProfileResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewBankProfileResponseWithDefaults

`func NewBankProfileResponseWithDefaults() *BankProfileResponse`

NewBankProfileResponseWithDefaults instantiates a new BankProfileResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetBankId

`func (o *BankProfileResponse) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *BankProfileResponse) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *BankProfileResponse) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetName

`func (o *BankProfileResponse) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *BankProfileResponse) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *BankProfileResponse) SetName(v string)`

SetName sets Name field to given value.


### GetDisposition

`func (o *BankProfileResponse) GetDisposition() DispositionTraits`

GetDisposition returns the Disposition field if non-nil, zero value otherwise.

### GetDispositionOk

`func (o *BankProfileResponse) GetDispositionOk() (*DispositionTraits, bool)`

GetDispositionOk returns a tuple with the Disposition field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDisposition

`func (o *BankProfileResponse) SetDisposition(v DispositionTraits)`

SetDisposition sets Disposition field to given value.


### GetMission

`func (o *BankProfileResponse) GetMission() string`

GetMission returns the Mission field if non-nil, zero value otherwise.

### GetMissionOk

`func (o *BankProfileResponse) GetMissionOk() (*string, bool)`

GetMissionOk returns a tuple with the Mission field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMission

`func (o *BankProfileResponse) SetMission(v string)`

SetMission sets Mission field to given value.


### GetBackground

`func (o *BankProfileResponse) GetBackground() string`

GetBackground returns the Background field if non-nil, zero value otherwise.

### GetBackgroundOk

`func (o *BankProfileResponse) GetBackgroundOk() (*string, bool)`

GetBackgroundOk returns a tuple with the Background field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBackground

`func (o *BankProfileResponse) SetBackground(v string)`

SetBackground sets Background field to given value.

### HasBackground

`func (o *BankProfileResponse) HasBackground() bool`

HasBackground returns a boolean if a field has been set.

### SetBackgroundNil

`func (o *BankProfileResponse) SetBackgroundNil(b bool)`

 SetBackgroundNil sets the value for Background to be an explicit nil

### UnsetBackground
`func (o *BankProfileResponse) UnsetBackground()`

UnsetBackground ensures that no value is present for Background, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


