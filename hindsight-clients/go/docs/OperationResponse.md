# OperationResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | **string** |  | 
**TaskType** | **string** |  | 
**ItemsCount** | **int32** |  | 
**DocumentId** | Pointer to **NullableString** |  | [optional] 
**CreatedAt** | **string** |  | 
**Status** | **string** |  | 
**ErrorMessage** | **NullableString** |  | 

## Methods

### NewOperationResponse

`func NewOperationResponse(id string, taskType string, itemsCount int32, createdAt string, status string, errorMessage NullableString, ) *OperationResponse`

NewOperationResponse instantiates a new OperationResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewOperationResponseWithDefaults

`func NewOperationResponseWithDefaults() *OperationResponse`

NewOperationResponseWithDefaults instantiates a new OperationResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *OperationResponse) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *OperationResponse) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *OperationResponse) SetId(v string)`

SetId sets Id field to given value.


### GetTaskType

`func (o *OperationResponse) GetTaskType() string`

GetTaskType returns the TaskType field if non-nil, zero value otherwise.

### GetTaskTypeOk

`func (o *OperationResponse) GetTaskTypeOk() (*string, bool)`

GetTaskTypeOk returns a tuple with the TaskType field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTaskType

`func (o *OperationResponse) SetTaskType(v string)`

SetTaskType sets TaskType field to given value.


### GetItemsCount

`func (o *OperationResponse) GetItemsCount() int32`

GetItemsCount returns the ItemsCount field if non-nil, zero value otherwise.

### GetItemsCountOk

`func (o *OperationResponse) GetItemsCountOk() (*int32, bool)`

GetItemsCountOk returns a tuple with the ItemsCount field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetItemsCount

`func (o *OperationResponse) SetItemsCount(v int32)`

SetItemsCount sets ItemsCount field to given value.


### GetDocumentId

`func (o *OperationResponse) GetDocumentId() string`

GetDocumentId returns the DocumentId field if non-nil, zero value otherwise.

### GetDocumentIdOk

`func (o *OperationResponse) GetDocumentIdOk() (*string, bool)`

GetDocumentIdOk returns a tuple with the DocumentId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDocumentId

`func (o *OperationResponse) SetDocumentId(v string)`

SetDocumentId sets DocumentId field to given value.

### HasDocumentId

`func (o *OperationResponse) HasDocumentId() bool`

HasDocumentId returns a boolean if a field has been set.

### SetDocumentIdNil

`func (o *OperationResponse) SetDocumentIdNil(b bool)`

 SetDocumentIdNil sets the value for DocumentId to be an explicit nil

### UnsetDocumentId
`func (o *OperationResponse) UnsetDocumentId()`

UnsetDocumentId ensures that no value is present for DocumentId, not even an explicit nil
### GetCreatedAt

`func (o *OperationResponse) GetCreatedAt() string`

GetCreatedAt returns the CreatedAt field if non-nil, zero value otherwise.

### GetCreatedAtOk

`func (o *OperationResponse) GetCreatedAtOk() (*string, bool)`

GetCreatedAtOk returns a tuple with the CreatedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetCreatedAt

`func (o *OperationResponse) SetCreatedAt(v string)`

SetCreatedAt sets CreatedAt field to given value.


### GetStatus

`func (o *OperationResponse) GetStatus() string`

GetStatus returns the Status field if non-nil, zero value otherwise.

### GetStatusOk

`func (o *OperationResponse) GetStatusOk() (*string, bool)`

GetStatusOk returns a tuple with the Status field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetStatus

`func (o *OperationResponse) SetStatus(v string)`

SetStatus sets Status field to given value.


### GetErrorMessage

`func (o *OperationResponse) GetErrorMessage() string`

GetErrorMessage returns the ErrorMessage field if non-nil, zero value otherwise.

### GetErrorMessageOk

`func (o *OperationResponse) GetErrorMessageOk() (*string, bool)`

GetErrorMessageOk returns a tuple with the ErrorMessage field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetErrorMessage

`func (o *OperationResponse) SetErrorMessage(v string)`

SetErrorMessage sets ErrorMessage field to given value.


### SetErrorMessageNil

`func (o *OperationResponse) SetErrorMessageNil(b bool)`

 SetErrorMessageNil sets the value for ErrorMessage to be an explicit nil

### UnsetErrorMessage
`func (o *OperationResponse) UnsetErrorMessage()`

UnsetErrorMessage ensures that no value is present for ErrorMessage, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


