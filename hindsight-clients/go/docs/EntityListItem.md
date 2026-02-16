# EntityListItem

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Id** | **string** |  | 
**CanonicalName** | **string** |  | 
**MentionCount** | **int32** |  | 
**FirstSeen** | Pointer to **NullableString** |  | [optional] 
**LastSeen** | Pointer to **NullableString** |  | [optional] 
**Metadata** | Pointer to **map[string]interface{}** |  | [optional] 

## Methods

### NewEntityListItem

`func NewEntityListItem(id string, canonicalName string, mentionCount int32, ) *EntityListItem`

NewEntityListItem instantiates a new EntityListItem object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewEntityListItemWithDefaults

`func NewEntityListItemWithDefaults() *EntityListItem`

NewEntityListItemWithDefaults instantiates a new EntityListItem object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetId

`func (o *EntityListItem) GetId() string`

GetId returns the Id field if non-nil, zero value otherwise.

### GetIdOk

`func (o *EntityListItem) GetIdOk() (*string, bool)`

GetIdOk returns a tuple with the Id field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetId

`func (o *EntityListItem) SetId(v string)`

SetId sets Id field to given value.


### GetCanonicalName

`func (o *EntityListItem) GetCanonicalName() string`

GetCanonicalName returns the CanonicalName field if non-nil, zero value otherwise.

### GetCanonicalNameOk

`func (o *EntityListItem) GetCanonicalNameOk() (*string, bool)`

GetCanonicalNameOk returns a tuple with the CanonicalName field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetCanonicalName

`func (o *EntityListItem) SetCanonicalName(v string)`

SetCanonicalName sets CanonicalName field to given value.


### GetMentionCount

`func (o *EntityListItem) GetMentionCount() int32`

GetMentionCount returns the MentionCount field if non-nil, zero value otherwise.

### GetMentionCountOk

`func (o *EntityListItem) GetMentionCountOk() (*int32, bool)`

GetMentionCountOk returns a tuple with the MentionCount field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMentionCount

`func (o *EntityListItem) SetMentionCount(v int32)`

SetMentionCount sets MentionCount field to given value.


### GetFirstSeen

`func (o *EntityListItem) GetFirstSeen() string`

GetFirstSeen returns the FirstSeen field if non-nil, zero value otherwise.

### GetFirstSeenOk

`func (o *EntityListItem) GetFirstSeenOk() (*string, bool)`

GetFirstSeenOk returns a tuple with the FirstSeen field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetFirstSeen

`func (o *EntityListItem) SetFirstSeen(v string)`

SetFirstSeen sets FirstSeen field to given value.

### HasFirstSeen

`func (o *EntityListItem) HasFirstSeen() bool`

HasFirstSeen returns a boolean if a field has been set.

### SetFirstSeenNil

`func (o *EntityListItem) SetFirstSeenNil(b bool)`

 SetFirstSeenNil sets the value for FirstSeen to be an explicit nil

### UnsetFirstSeen
`func (o *EntityListItem) UnsetFirstSeen()`

UnsetFirstSeen ensures that no value is present for FirstSeen, not even an explicit nil
### GetLastSeen

`func (o *EntityListItem) GetLastSeen() string`

GetLastSeen returns the LastSeen field if non-nil, zero value otherwise.

### GetLastSeenOk

`func (o *EntityListItem) GetLastSeenOk() (*string, bool)`

GetLastSeenOk returns a tuple with the LastSeen field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLastSeen

`func (o *EntityListItem) SetLastSeen(v string)`

SetLastSeen sets LastSeen field to given value.

### HasLastSeen

`func (o *EntityListItem) HasLastSeen() bool`

HasLastSeen returns a boolean if a field has been set.

### SetLastSeenNil

`func (o *EntityListItem) SetLastSeenNil(b bool)`

 SetLastSeenNil sets the value for LastSeen to be an explicit nil

### UnsetLastSeen
`func (o *EntityListItem) UnsetLastSeen()`

UnsetLastSeen ensures that no value is present for LastSeen, not even an explicit nil
### GetMetadata

`func (o *EntityListItem) GetMetadata() map[string]interface{}`

GetMetadata returns the Metadata field if non-nil, zero value otherwise.

### GetMetadataOk

`func (o *EntityListItem) GetMetadataOk() (*map[string]interface{}, bool)`

GetMetadataOk returns a tuple with the Metadata field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMetadata

`func (o *EntityListItem) SetMetadata(v map[string]interface{})`

SetMetadata sets Metadata field to given value.

### HasMetadata

`func (o *EntityListItem) HasMetadata() bool`

HasMetadata returns a boolean if a field has been set.

### SetMetadataNil

`func (o *EntityListItem) SetMetadataNil(b bool)`

 SetMetadataNil sets the value for Metadata to be an explicit nil

### UnsetMetadata
`func (o *EntityListItem) UnsetMetadata()`

UnsetMetadata ensures that no value is present for Metadata, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


