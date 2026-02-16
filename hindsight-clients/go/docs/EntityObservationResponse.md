# EntityObservationResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Text** | **string** |  | 
**MentionedAt** | Pointer to **NullableString** |  | [optional] 

## Methods

### NewEntityObservationResponse

`func NewEntityObservationResponse(text string, ) *EntityObservationResponse`

NewEntityObservationResponse instantiates a new EntityObservationResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewEntityObservationResponseWithDefaults

`func NewEntityObservationResponseWithDefaults() *EntityObservationResponse`

NewEntityObservationResponseWithDefaults instantiates a new EntityObservationResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetText

`func (o *EntityObservationResponse) GetText() string`

GetText returns the Text field if non-nil, zero value otherwise.

### GetTextOk

`func (o *EntityObservationResponse) GetTextOk() (*string, bool)`

GetTextOk returns a tuple with the Text field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetText

`func (o *EntityObservationResponse) SetText(v string)`

SetText sets Text field to given value.


### GetMentionedAt

`func (o *EntityObservationResponse) GetMentionedAt() string`

GetMentionedAt returns the MentionedAt field if non-nil, zero value otherwise.

### GetMentionedAtOk

`func (o *EntityObservationResponse) GetMentionedAtOk() (*string, bool)`

GetMentionedAtOk returns a tuple with the MentionedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMentionedAt

`func (o *EntityObservationResponse) SetMentionedAt(v string)`

SetMentionedAt sets MentionedAt field to given value.

### HasMentionedAt

`func (o *EntityObservationResponse) HasMentionedAt() bool`

HasMentionedAt returns a boolean if a field has been set.

### SetMentionedAtNil

`func (o *EntityObservationResponse) SetMentionedAtNil(b bool)`

 SetMentionedAtNil sets the value for MentionedAt to be an explicit nil

### UnsetMentionedAt
`func (o *EntityObservationResponse) UnsetMentionedAt()`

UnsetMentionedAt ensures that no value is present for MentionedAt, not even an explicit nil

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


