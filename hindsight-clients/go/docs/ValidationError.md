# ValidationError

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Loc** | [**[]ValidationErrorLocInner**](ValidationErrorLocInner.md) |  | 
**Msg** | **string** |  | 
**Type** | **string** |  | 

## Methods

### NewValidationError

`func NewValidationError(loc []ValidationErrorLocInner, msg string, type_ string, ) *ValidationError`

NewValidationError instantiates a new ValidationError object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewValidationErrorWithDefaults

`func NewValidationErrorWithDefaults() *ValidationError`

NewValidationErrorWithDefaults instantiates a new ValidationError object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetLoc

`func (o *ValidationError) GetLoc() []ValidationErrorLocInner`

GetLoc returns the Loc field if non-nil, zero value otherwise.

### GetLocOk

`func (o *ValidationError) GetLocOk() (*[]ValidationErrorLocInner, bool)`

GetLocOk returns a tuple with the Loc field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLoc

`func (o *ValidationError) SetLoc(v []ValidationErrorLocInner)`

SetLoc sets Loc field to given value.


### GetMsg

`func (o *ValidationError) GetMsg() string`

GetMsg returns the Msg field if non-nil, zero value otherwise.

### GetMsgOk

`func (o *ValidationError) GetMsgOk() (*string, bool)`

GetMsgOk returns a tuple with the Msg field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMsg

`func (o *ValidationError) SetMsg(v string)`

SetMsg sets Msg field to given value.


### GetType

`func (o *ValidationError) GetType() string`

GetType returns the Type field if non-nil, zero value otherwise.

### GetTypeOk

`func (o *ValidationError) GetTypeOk() (*string, bool)`

GetTypeOk returns a tuple with the Type field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetType

`func (o *ValidationError) SetType(v string)`

SetType sets Type field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


