# DeleteResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Success** | **bool** |  | 
**Message** | Pointer to **NullableString** |  | [optional] 
**DeletedCount** | Pointer to **NullableInt32** |  | [optional] 

## Methods

### NewDeleteResponse

`func NewDeleteResponse(success bool, ) *DeleteResponse`

NewDeleteResponse instantiates a new DeleteResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewDeleteResponseWithDefaults

`func NewDeleteResponseWithDefaults() *DeleteResponse`

NewDeleteResponseWithDefaults instantiates a new DeleteResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetSuccess

`func (o *DeleteResponse) GetSuccess() bool`

GetSuccess returns the Success field if non-nil, zero value otherwise.

### GetSuccessOk

`func (o *DeleteResponse) GetSuccessOk() (*bool, bool)`

GetSuccessOk returns a tuple with the Success field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetSuccess

`func (o *DeleteResponse) SetSuccess(v bool)`

SetSuccess sets Success field to given value.


### GetMessage

`func (o *DeleteResponse) GetMessage() string`

GetMessage returns the Message field if non-nil, zero value otherwise.

### GetMessageOk

`func (o *DeleteResponse) GetMessageOk() (*string, bool)`

GetMessageOk returns a tuple with the Message field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMessage

`func (o *DeleteResponse) SetMessage(v string)`

SetMessage sets Message field to given value.

### HasMessage

`func (o *DeleteResponse) HasMessage() bool`

HasMessage returns a boolean if a field has been set.

### SetMessageNil

`func (o *DeleteResponse) SetMessageNil(b bool)`

 SetMessageNil sets the value for Message to be an explicit nil

### UnsetMessage
`func (o *DeleteResponse) UnsetMessage()`

UnsetMessage ensures that no value is present for Message, not even an explicit nil
### GetDeletedCount

`func (o *DeleteResponse) GetDeletedCount() int32`

GetDeletedCount returns the DeletedCount field if non-nil, zero value otherwise.

### GetDeletedCountOk

`func (o *DeleteResponse) GetDeletedCountOk() (*int32, bool)`

GetDeletedCountOk returns a tuple with the DeletedCount field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDeletedCount

`func (o *DeleteResponse) SetDeletedCount(v int32)`

SetDeletedCount sets DeletedCount field to given value.

### HasDeletedCount

`func (o *DeleteResponse) HasDeletedCount() bool`

HasDeletedCount returns a boolean if a field has been set.

### SetDeletedCountNil

`func (o *DeleteResponse) SetDeletedCountNil(b bool)`

 SetDeletedCountNil sets the value for DeletedCount to be an explicit nil

### UnsetDeletedCount
`func (o *DeleteResponse) UnsetDeletedCount()`

UnsetDeletedCount ensures that no value is present for DeletedCount, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


