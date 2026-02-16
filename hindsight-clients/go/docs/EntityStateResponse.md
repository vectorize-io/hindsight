# EntityStateResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**EntityId** | **string** |  | 
**CanonicalName** | **string** |  | 
**Observations** | [**[]EntityObservationResponse**](EntityObservationResponse.md) |  | 

## Methods

### NewEntityStateResponse

`func NewEntityStateResponse(entityId string, canonicalName string, observations []EntityObservationResponse, ) *EntityStateResponse`

NewEntityStateResponse instantiates a new EntityStateResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewEntityStateResponseWithDefaults

`func NewEntityStateResponseWithDefaults() *EntityStateResponse`

NewEntityStateResponseWithDefaults instantiates a new EntityStateResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetEntityId

`func (o *EntityStateResponse) GetEntityId() string`

GetEntityId returns the EntityId field if non-nil, zero value otherwise.

### GetEntityIdOk

`func (o *EntityStateResponse) GetEntityIdOk() (*string, bool)`

GetEntityIdOk returns a tuple with the EntityId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetEntityId

`func (o *EntityStateResponse) SetEntityId(v string)`

SetEntityId sets EntityId field to given value.


### GetCanonicalName

`func (o *EntityStateResponse) GetCanonicalName() string`

GetCanonicalName returns the CanonicalName field if non-nil, zero value otherwise.

### GetCanonicalNameOk

`func (o *EntityStateResponse) GetCanonicalNameOk() (*string, bool)`

GetCanonicalNameOk returns a tuple with the CanonicalName field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetCanonicalName

`func (o *EntityStateResponse) SetCanonicalName(v string)`

SetCanonicalName sets CanonicalName field to given value.


### GetObservations

`func (o *EntityStateResponse) GetObservations() []EntityObservationResponse`

GetObservations returns the Observations field if non-nil, zero value otherwise.

### GetObservationsOk

`func (o *EntityStateResponse) GetObservationsOk() (*[]EntityObservationResponse, bool)`

GetObservationsOk returns a tuple with the Observations field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetObservations

`func (o *EntityStateResponse) SetObservations(v []EntityObservationResponse)`

SetObservations sets Observations field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


