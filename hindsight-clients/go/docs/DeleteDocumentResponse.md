# DeleteDocumentResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Success** | **bool** |  | 
**Message** | **string** |  | 
**DocumentId** | **string** |  | 
**MemoryUnitsDeleted** | **int32** |  | 

## Methods

### NewDeleteDocumentResponse

`func NewDeleteDocumentResponse(success bool, message string, documentId string, memoryUnitsDeleted int32, ) *DeleteDocumentResponse`

NewDeleteDocumentResponse instantiates a new DeleteDocumentResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewDeleteDocumentResponseWithDefaults

`func NewDeleteDocumentResponseWithDefaults() *DeleteDocumentResponse`

NewDeleteDocumentResponseWithDefaults instantiates a new DeleteDocumentResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetSuccess

`func (o *DeleteDocumentResponse) GetSuccess() bool`

GetSuccess returns the Success field if non-nil, zero value otherwise.

### GetSuccessOk

`func (o *DeleteDocumentResponse) GetSuccessOk() (*bool, bool)`

GetSuccessOk returns a tuple with the Success field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetSuccess

`func (o *DeleteDocumentResponse) SetSuccess(v bool)`

SetSuccess sets Success field to given value.


### GetMessage

`func (o *DeleteDocumentResponse) GetMessage() string`

GetMessage returns the Message field if non-nil, zero value otherwise.

### GetMessageOk

`func (o *DeleteDocumentResponse) GetMessageOk() (*string, bool)`

GetMessageOk returns a tuple with the Message field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMessage

`func (o *DeleteDocumentResponse) SetMessage(v string)`

SetMessage sets Message field to given value.


### GetDocumentId

`func (o *DeleteDocumentResponse) GetDocumentId() string`

GetDocumentId returns the DocumentId field if non-nil, zero value otherwise.

### GetDocumentIdOk

`func (o *DeleteDocumentResponse) GetDocumentIdOk() (*string, bool)`

GetDocumentIdOk returns a tuple with the DocumentId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetDocumentId

`func (o *DeleteDocumentResponse) SetDocumentId(v string)`

SetDocumentId sets DocumentId field to given value.


### GetMemoryUnitsDeleted

`func (o *DeleteDocumentResponse) GetMemoryUnitsDeleted() int32`

GetMemoryUnitsDeleted returns the MemoryUnitsDeleted field if non-nil, zero value otherwise.

### GetMemoryUnitsDeletedOk

`func (o *DeleteDocumentResponse) GetMemoryUnitsDeletedOk() (*int32, bool)`

GetMemoryUnitsDeletedOk returns a tuple with the MemoryUnitsDeleted field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMemoryUnitsDeleted

`func (o *DeleteDocumentResponse) SetMemoryUnitsDeleted(v int32)`

SetMemoryUnitsDeleted sets MemoryUnitsDeleted field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


