# VersionResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ApiVersion** | **string** | API version string | 
**Features** | [**FeaturesInfo**](FeaturesInfo.md) | Enabled feature flags | 

## Methods

### NewVersionResponse

`func NewVersionResponse(apiVersion string, features FeaturesInfo, ) *VersionResponse`

NewVersionResponse instantiates a new VersionResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewVersionResponseWithDefaults

`func NewVersionResponseWithDefaults() *VersionResponse`

NewVersionResponseWithDefaults instantiates a new VersionResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetApiVersion

`func (o *VersionResponse) GetApiVersion() string`

GetApiVersion returns the ApiVersion field if non-nil, zero value otherwise.

### GetApiVersionOk

`func (o *VersionResponse) GetApiVersionOk() (*string, bool)`

GetApiVersionOk returns a tuple with the ApiVersion field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetApiVersion

`func (o *VersionResponse) SetApiVersion(v string)`

SetApiVersion sets ApiVersion field to given value.


### GetFeatures

`func (o *VersionResponse) GetFeatures() FeaturesInfo`

GetFeatures returns the Features field if non-nil, zero value otherwise.

### GetFeaturesOk

`func (o *VersionResponse) GetFeaturesOk() (*FeaturesInfo, bool)`

GetFeaturesOk returns a tuple with the Features field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetFeatures

`func (o *VersionResponse) SetFeatures(v FeaturesInfo)`

SetFeatures sets Features field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


