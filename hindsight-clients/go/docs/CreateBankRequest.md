# CreateBankRequest

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Name** | Pointer to **NullableString** |  | [optional] 
**Disposition** | Pointer to [**NullableDispositionTraits**](DispositionTraits.md) |  | [optional] 
**Mission** | Pointer to **NullableString** |  | [optional] 
**Background** | Pointer to **NullableString** |  | [optional] 

## Methods

### NewCreateBankRequest

`func NewCreateBankRequest() *CreateBankRequest`

NewCreateBankRequest instantiates a new CreateBankRequest object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewCreateBankRequestWithDefaults

`func NewCreateBankRequestWithDefaults() *CreateBankRequest`

NewCreateBankRequestWithDefaults instantiates a new CreateBankRequest object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetName

`func (o *CreateBankRequest) GetName() string`

GetName returns the Name field if non-nil, zero value otherwise.

### GetNameOk

`func (o *CreateBankRequest) GetNameOk() (*string, bool)`

GetNameOk returns a tuple with the Name field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetName

`func (o *CreateBankRequest) SetName(v string)`

SetName sets Name field to given value.

### HasName

`func (o *CreateBankRequest) HasName() bool`

HasName returns a boolean if a field has been set.

### SetNameNil

`func (o *CreateBankRequest) SetNameNil(b bool)`

 SetNameNil sets the value for Name to be an explicit nil

### UnsetName
`func (o *CreateBankRequest) UnsetName()`

UnsetName ensures that no value is present for Name, not even an explicit nil
### GetDisposition

`func (o *CreateBankRequest) GetDisposition() DispositionTraits`

GetDisposition returns the Disposition field if non-nil, zero value otherwise.

### GetDispositionOk

`func (o *CreateBankRequest) GetDispositionOk() (*DispositionTraits, bool)`

GetDispositionOk returns a tuple with the Disposition field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDisposition

`func (o *CreateBankRequest) SetDisposition(v DispositionTraits)`

SetDisposition sets Disposition field to given value.

### HasDisposition

`func (o *CreateBankRequest) HasDisposition() bool`

HasDisposition returns a boolean if a field has been set.

### SetDispositionNil

`func (o *CreateBankRequest) SetDispositionNil(b bool)`

 SetDispositionNil sets the value for Disposition to be an explicit nil

### UnsetDisposition
`func (o *CreateBankRequest) UnsetDisposition()`

UnsetDisposition ensures that no value is present for Disposition, not even an explicit nil
### GetMission

`func (o *CreateBankRequest) GetMission() string`

GetMission returns the Mission field if non-nil, zero value otherwise.

### GetMissionOk

`func (o *CreateBankRequest) GetMissionOk() (*string, bool)`

GetMissionOk returns a tuple with the Mission field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMission

`func (o *CreateBankRequest) SetMission(v string)`

SetMission sets Mission field to given value.

### HasMission

`func (o *CreateBankRequest) HasMission() bool`

HasMission returns a boolean if a field has been set.

### SetMissionNil

`func (o *CreateBankRequest) SetMissionNil(b bool)`

 SetMissionNil sets the value for Mission to be an explicit nil

### UnsetMission
`func (o *CreateBankRequest) UnsetMission()`

UnsetMission ensures that no value is present for Mission, not even an explicit nil
### GetBackground

`func (o *CreateBankRequest) GetBackground() string`

GetBackground returns the Background field if non-nil, zero value otherwise.

### GetBackgroundOk

`func (o *CreateBankRequest) GetBackgroundOk() (*string, bool)`

GetBackgroundOk returns a tuple with the Background field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBackground

`func (o *CreateBankRequest) SetBackground(v string)`

SetBackground sets Background field to given value.

### HasBackground

`func (o *CreateBankRequest) HasBackground() bool`

HasBackground returns a boolean if a field has been set.

### SetBackgroundNil

`func (o *CreateBankRequest) SetBackgroundNil(b bool)`

 SetBackgroundNil sets the value for Background to be an explicit nil

### UnsetBackground
`func (o *CreateBankRequest) UnsetBackground()`

UnsetBackground ensures that no value is present for Background, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


