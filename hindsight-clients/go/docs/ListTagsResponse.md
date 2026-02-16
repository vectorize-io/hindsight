# ListTagsResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Items** | [**[]TagItem**](TagItem.md) |  | 
**Total** | **int32** |  | 
**Limit** | **int32** |  | 
**Offset** | **int32** |  | 

## Methods

### NewListTagsResponse

`func NewListTagsResponse(items []TagItem, total int32, limit int32, offset int32, ) *ListTagsResponse`

NewListTagsResponse instantiates a new ListTagsResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewListTagsResponseWithDefaults

`func NewListTagsResponseWithDefaults() *ListTagsResponse`

NewListTagsResponseWithDefaults instantiates a new ListTagsResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetItems

`func (o *ListTagsResponse) GetItems() []TagItem`

GetItems returns the Items field if non-nil, zero value otherwise.

### GetItemsOk

`func (o *ListTagsResponse) GetItemsOk() (*[]TagItem, bool)`

GetItemsOk returns a tuple with the Items field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetItems

`func (o *ListTagsResponse) SetItems(v []TagItem)`

SetItems sets Items field to given value.


### GetTotal

`func (o *ListTagsResponse) GetTotal() int32`

GetTotal returns the Total field if non-nil, zero value otherwise.

### GetTotalOk

`func (o *ListTagsResponse) GetTotalOk() (*int32, bool)`

GetTotalOk returns a tuple with the Total field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotal

`func (o *ListTagsResponse) SetTotal(v int32)`

SetTotal sets Total field to given value.


### GetLimit

`func (o *ListTagsResponse) GetLimit() int32`

GetLimit returns the Limit field if non-nil, zero value otherwise.

### GetLimitOk

`func (o *ListTagsResponse) GetLimitOk() (*int32, bool)`

GetLimitOk returns a tuple with the Limit field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLimit

`func (o *ListTagsResponse) SetLimit(v int32)`

SetLimit sets Limit field to given value.


### GetOffset

`func (o *ListTagsResponse) GetOffset() int32`

GetOffset returns the Offset field if non-nil, zero value otherwise.

### GetOffsetOk

`func (o *ListTagsResponse) GetOffsetOk() (*int32, bool)`

GetOffsetOk returns a tuple with the Offset field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetOffset

`func (o *ListTagsResponse) SetOffset(v int32)`

SetOffset sets Offset field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


