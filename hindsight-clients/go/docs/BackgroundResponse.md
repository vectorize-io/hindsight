# BackgroundResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Mission** | **string** |  | 
**Background** | Pointer to **NullableString** |  | [optional] 
**Disposition** | Pointer to [**NullableDispositionTraits**](DispositionTraits.md) |  | [optional] 

## Methods

### NewBackgroundResponse

`func NewBackgroundResponse(mission string, ) *BackgroundResponse`

NewBackgroundResponse instantiates a new BackgroundResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewBackgroundResponseWithDefaults

`func NewBackgroundResponseWithDefaults() *BackgroundResponse`

NewBackgroundResponseWithDefaults instantiates a new BackgroundResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetMission

`func (o *BackgroundResponse) GetMission() string`

GetMission returns the Mission field if non-nil, zero value otherwise.

### GetMissionOk

`func (o *BackgroundResponse) GetMissionOk() (*string, bool)`

GetMissionOk returns a tuple with the Mission field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMission

`func (o *BackgroundResponse) SetMission(v string)`

SetMission sets Mission field to given value.


### GetBackground

`func (o *BackgroundResponse) GetBackground() string`

GetBackground returns the Background field if non-nil, zero value otherwise.

### GetBackgroundOk

`func (o *BackgroundResponse) GetBackgroundOk() (*string, bool)`

GetBackgroundOk returns a tuple with the Background field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBackground

`func (o *BackgroundResponse) SetBackground(v string)`

SetBackground sets Background field to given value.

### HasBackground

`func (o *BackgroundResponse) HasBackground() bool`

HasBackground returns a boolean if a field has been set.

### SetBackgroundNil

`func (o *BackgroundResponse) SetBackgroundNil(b bool)`

 SetBackgroundNil sets the value for Background to be an explicit nil

### UnsetBackground
`func (o *BackgroundResponse) UnsetBackground()`

UnsetBackground ensures that no value is present for Background, not even an explicit nil
### GetDisposition

`func (o *BackgroundResponse) GetDisposition() DispositionTraits`

GetDisposition returns the Disposition field if non-nil, zero value otherwise.

### GetDispositionOk

`func (o *BackgroundResponse) GetDispositionOk() (*DispositionTraits, bool)`

GetDispositionOk returns a tuple with the Disposition field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDisposition

`func (o *BackgroundResponse) SetDisposition(v DispositionTraits)`

SetDisposition sets Disposition field to given value.

### HasDisposition

`func (o *BackgroundResponse) HasDisposition() bool`

HasDisposition returns a boolean if a field has been set.

### SetDispositionNil

`func (o *BackgroundResponse) SetDispositionNil(b bool)`

 SetDispositionNil sets the value for Disposition to be an explicit nil

### UnsetDisposition
`func (o *BackgroundResponse) UnsetDisposition()`

UnsetDisposition ensures that no value is present for Disposition, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


