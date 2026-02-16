# EntityListResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Items** | [**[]EntityListItem**](EntityListItem.md) |  | 
**Total** | **int32** |  | 
**Limit** | **int32** |  | 
**Offset** | **int32** |  | 

## Methods

### NewEntityListResponse

`func NewEntityListResponse(items []EntityListItem, total int32, limit int32, offset int32, ) *EntityListResponse`

NewEntityListResponse instantiates a new EntityListResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewEntityListResponseWithDefaults

`func NewEntityListResponseWithDefaults() *EntityListResponse`

NewEntityListResponseWithDefaults instantiates a new EntityListResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetItems

`func (o *EntityListResponse) GetItems() []EntityListItem`

GetItems returns the Items field if non-nil, zero value otherwise.

### GetItemsOk

`func (o *EntityListResponse) GetItemsOk() (*[]EntityListItem, bool)`

GetItemsOk returns a tuple with the Items field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetItems

`func (o *EntityListResponse) SetItems(v []EntityListItem)`

SetItems sets Items field to given value.


### GetTotal

`func (o *EntityListResponse) GetTotal() int32`

GetTotal returns the Total field if non-nil, zero value otherwise.

### GetTotalOk

`func (o *EntityListResponse) GetTotalOk() (*int32, bool)`

GetTotalOk returns a tuple with the Total field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotal

`func (o *EntityListResponse) SetTotal(v int32)`

SetTotal sets Total field to given value.


### GetLimit

`func (o *EntityListResponse) GetLimit() int32`

GetLimit returns the Limit field if non-nil, zero value otherwise.

### GetLimitOk

`func (o *EntityListResponse) GetLimitOk() (*int32, bool)`

GetLimitOk returns a tuple with the Limit field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLimit

`func (o *EntityListResponse) SetLimit(v int32)`

SetLimit sets Limit field to given value.


### GetOffset

`func (o *EntityListResponse) GetOffset() int32`

GetOffset returns the Offset field if non-nil, zero value otherwise.

### GetOffsetOk

`func (o *EntityListResponse) GetOffsetOk() (*int32, bool)`

GetOffsetOk returns a tuple with the Offset field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOffset

`func (o *EntityListResponse) SetOffset(v int32)`

SetOffset sets Offset field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


