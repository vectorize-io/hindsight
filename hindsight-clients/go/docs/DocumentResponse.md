# DocumentResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | **string** |  | 
**BankId** | **string** |  | 
**OriginalText** | **string** |  | 
**ContentHash** | **NullableString** |  | 
**CreatedAt** | **string** |  | 
**UpdatedAt** | **string** |  | 
**MemoryUnitCount** | **int32** |  | 
**Tags** | Pointer to **[]string** | Tags associated with this document | [optional] [default to []]

## Methods

### NewDocumentResponse

`func NewDocumentResponse(id string, bankId string, originalText string, contentHash NullableString, createdAt string, updatedAt string, memoryUnitCount int32, ) *DocumentResponse`

NewDocumentResponse instantiates a new DocumentResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewDocumentResponseWithDefaults

`func NewDocumentResponseWithDefaults() *DocumentResponse`

NewDocumentResponseWithDefaults instantiates a new DocumentResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *DocumentResponse) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *DocumentResponse) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *DocumentResponse) SetId(v string)`

SetId sets Id field to given value.


### GetBankId

`func (o *DocumentResponse) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *DocumentResponse) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *DocumentResponse) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetOriginalText

`func (o *DocumentResponse) GetOriginalText() string`

GetOriginalText returns the OriginalText field if non-nil, zero value otherwise.

### GetOriginalTextOk

`func (o *DocumentResponse) GetOriginalTextOk() (*string, bool)`

GetOriginalTextOk returns a tuple with the OriginalText field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOriginalText

`func (o *DocumentResponse) SetOriginalText(v string)`

SetOriginalText sets OriginalText field to given value.


### GetContentHash

`func (o *DocumentResponse) GetContentHash() string`

GetContentHash returns the ContentHash field if non-nil, zero value otherwise.

### GetContentHashOk

`func (o *DocumentResponse) GetContentHashOk() (*string, bool)`

GetContentHashOk returns a tuple with the ContentHash field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetContentHash

`func (o *DocumentResponse) SetContentHash(v string)`

SetContentHash sets ContentHash field to given value.


### SetContentHashNil

`func (o *DocumentResponse) SetContentHashNil(b bool)`

 SetContentHashNil sets the value for ContentHash to be an explicit nil

### UnsetContentHash
`func (o *DocumentResponse) UnsetContentHash()`

UnsetContentHash ensures that no value is present for ContentHash, not even an explicit nil
### GetCreatedAt

`func (o *DocumentResponse) GetCreatedAt() string`

GetCreatedAt returns the CreatedAt field if non-nil, zero value otherwise.

### GetCreatedAtOk

`func (o *DocumentResponse) GetCreatedAtOk() (*string, bool)`

GetCreatedAtOk returns a tuple with the CreatedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetCreatedAt

`func (o *DocumentResponse) SetCreatedAt(v string)`

SetCreatedAt sets CreatedAt field to given value.


### GetUpdatedAt

`func (o *DocumentResponse) GetUpdatedAt() string`

GetUpdatedAt returns the UpdatedAt field if non-nil, zero value otherwise.

### GetUpdatedAtOk

`func (o *DocumentResponse) GetUpdatedAtOk() (*string, bool)`

GetUpdatedAtOk returns a tuple with the UpdatedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetUpdatedAt

`func (o *DocumentResponse) SetUpdatedAt(v string)`

SetUpdatedAt sets UpdatedAt field to given value.


### GetMemoryUnitCount

`func (o *DocumentResponse) GetMemoryUnitCount() int32`

GetMemoryUnitCount returns the MemoryUnitCount field if non-nil, zero value otherwise.

### GetMemoryUnitCountOk

`func (o *DocumentResponse) GetMemoryUnitCountOk() (*int32, bool)`

GetMemoryUnitCountOk returns a tuple with the MemoryUnitCount field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMemoryUnitCount

`func (o *DocumentResponse) SetMemoryUnitCount(v int32)`

SetMemoryUnitCount sets MemoryUnitCount field to given value.


### GetTags

`func (o *DocumentResponse) GetTags() []string`

GetTags returns the Tags field if non-nil, zero value otherwise.

### GetTagsOk

`func (o *DocumentResponse) GetTagsOk() (*[]string, bool)`

GetTagsOk returns a tuple with the Tags field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTags

`func (o *DocumentResponse) SetTags(v []string)`

SetTags sets Tags field to given value.

### HasTags

`func (o *DocumentResponse) HasTags() bool`

HasTags returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


