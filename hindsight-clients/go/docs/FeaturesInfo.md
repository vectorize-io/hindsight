# FeaturesInfo

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Observations** | **bool** | Whether observations (auto-consolidation) are enabled | 
**Mcp** | **bool** | Whether MCP (Model Context Protocol) server is enabled | 
**Worker** | **bool** | Whether the background worker is enabled | 
**BankConfigApi** | **bool** | Whether per-bank configuration API is enabled | 

## Methods

### NewFeaturesInfo

`func NewFeaturesInfo(observations bool, mcp bool, worker bool, bankConfigApi bool, ) *FeaturesInfo`

NewFeaturesInfo instantiates a new FeaturesInfo object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewFeaturesInfoWithDefaults

`func NewFeaturesInfoWithDefaults() *FeaturesInfo`

NewFeaturesInfoWithDefaults instantiates a new FeaturesInfo object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetObservations

`func (o *FeaturesInfo) GetObservations() bool`

GetObservations returns the Observations field if non-nil, zero value otherwise.

### GetObservationsOk

`func (o *FeaturesInfo) GetObservationsOk() (*bool, bool)`

GetObservationsOk returns a tuple with the Observations field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetObservations

`func (o *FeaturesInfo) SetObservations(v bool)`

SetObservations sets Observations field to given value.


### GetMcp

`func (o *FeaturesInfo) GetMcp() bool`

GetMcp returns the Mcp field if non-nil, zero value otherwise.

### GetMcpOk

`func (o *FeaturesInfo) GetMcpOk() (*bool, bool)`

GetMcpOk returns a tuple with the Mcp field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetMcp

`func (o *FeaturesInfo) SetMcp(v bool)`

SetMcp sets Mcp field to given value.


### GetWorker

`func (o *FeaturesInfo) GetWorker() bool`

GetWorker returns the Worker field if non-nil, zero value otherwise.

### GetWorkerOk

`func (o *FeaturesInfo) GetWorkerOk() (*bool, bool)`

GetWorkerOk returns a tuple with the Worker field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetWorker

`func (o *FeaturesInfo) SetWorker(v bool)`

SetWorker sets Worker field to given value.


### GetBankConfigApi

`func (o *FeaturesInfo) GetBankConfigApi() bool`

GetBankConfigApi returns the BankConfigApi field if non-nil, zero value otherwise.

### GetBankConfigApiOk

`func (o *FeaturesInfo) GetBankConfigApiOk() (*bool, bool)`

GetBankConfigApiOk returns a tuple with the BankConfigApi field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankConfigApi

`func (o *FeaturesInfo) SetBankConfigApi(v bool)`

SetBankConfigApi sets BankConfigApi field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


