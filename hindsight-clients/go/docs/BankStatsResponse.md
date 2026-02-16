# BankStatsResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**BankId** | **string** |  | 
**TotalNodes** | **int32** |  | 
**TotalLinks** | **int32** |  | 
**TotalDocuments** | **int32** |  | 
**NodesByFactType** | **map[string]int32** |  | 
**LinksByLinkType** | **map[string]int32** |  | 
**LinksByFactType** | **map[string]int32** |  | 
**LinksBreakdown** | **map[string]map[string]int32** |  | 
**PendingOperations** | **int32** |  | 
**FailedOperations** | **int32** |  | 
**LastConsolidatedAt** | Pointer to **NullableString** |  | [optional] 
**PendingConsolidation** | Pointer to **int32** | Number of memories not yet processed into observations | [optional] [default to 0]
**TotalObservations** | Pointer to **int32** | Total number of observations | [optional] [default to 0]

## Methods

### NewBankStatsResponse

`func NewBankStatsResponse(bankId string, totalNodes int32, totalLinks int32, totalDocuments int32, nodesByFactType map[string]int32, linksByLinkType map[string]int32, linksByFactType map[string]int32, linksBreakdown map[string]map[string]int32, pendingOperations int32, failedOperations int32, ) *BankStatsResponse`

NewBankStatsResponse instantiates a new BankStatsResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewBankStatsResponseWithDefaults

`func NewBankStatsResponseWithDefaults() *BankStatsResponse`

NewBankStatsResponseWithDefaults instantiates a new BankStatsResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetBankId

`func (o *BankStatsResponse) GetBankId() string`

GetBankId returns the BankId field if non-nil, zero value otherwise.

### GetBankIdOk

`func (o *BankStatsResponse) GetBankIdOk() (*string, bool)`

GetBankIdOk returns a tuple with the BankId field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetBankId

`func (o *BankStatsResponse) SetBankId(v string)`

SetBankId sets BankId field to given value.


### GetTotalNodes

`func (o *BankStatsResponse) GetTotalNodes() int32`

GetTotalNodes returns the TotalNodes field if non-nil, zero value otherwise.

### GetTotalNodesOk

`func (o *BankStatsResponse) GetTotalNodesOk() (*int32, bool)`

GetTotalNodesOk returns a tuple with the TotalNodes field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotalNodes

`func (o *BankStatsResponse) SetTotalNodes(v int32)`

SetTotalNodes sets TotalNodes field to given value.


### GetTotalLinks

`func (o *BankStatsResponse) GetTotalLinks() int32`

GetTotalLinks returns the TotalLinks field if non-nil, zero value otherwise.

### GetTotalLinksOk

`func (o *BankStatsResponse) GetTotalLinksOk() (*int32, bool)`

GetTotalLinksOk returns a tuple with the TotalLinks field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotalLinks

`func (o *BankStatsResponse) SetTotalLinks(v int32)`

SetTotalLinks sets TotalLinks field to given value.


### GetTotalDocuments

`func (o *BankStatsResponse) GetTotalDocuments() int32`

GetTotalDocuments returns the TotalDocuments field if non-nil, zero value otherwise.

### GetTotalDocumentsOk

`func (o *BankStatsResponse) GetTotalDocumentsOk() (*int32, bool)`

GetTotalDocumentsOk returns a tuple with the TotalDocuments field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotalDocuments

`func (o *BankStatsResponse) SetTotalDocuments(v int32)`

SetTotalDocuments sets TotalDocuments field to given value.


### GetNodesByFactType

`func (o *BankStatsResponse) GetNodesByFactType() map[string]int32`

GetNodesByFactType returns the NodesByFactType field if non-nil, zero value otherwise.

### GetNodesByFactTypeOk

`func (o *BankStatsResponse) GetNodesByFactTypeOk() (*map[string]int32, bool)`

GetNodesByFactTypeOk returns a tuple with the NodesByFactType field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetNodesByFactType

`func (o *BankStatsResponse) SetNodesByFactType(v map[string]int32)`

SetNodesByFactType sets NodesByFactType field to given value.


### GetLinksByLinkType

`func (o *BankStatsResponse) GetLinksByLinkType() map[string]int32`

GetLinksByLinkType returns the LinksByLinkType field if non-nil, zero value otherwise.

### GetLinksByLinkTypeOk

`func (o *BankStatsResponse) GetLinksByLinkTypeOk() (*map[string]int32, bool)`

GetLinksByLinkTypeOk returns a tuple with the LinksByLinkType field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLinksByLinkType

`func (o *BankStatsResponse) SetLinksByLinkType(v map[string]int32)`

SetLinksByLinkType sets LinksByLinkType field to given value.


### GetLinksByFactType

`func (o *BankStatsResponse) GetLinksByFactType() map[string]int32`

GetLinksByFactType returns the LinksByFactType field if non-nil, zero value otherwise.

### GetLinksByFactTypeOk

`func (o *BankStatsResponse) GetLinksByFactTypeOk() (*map[string]int32, bool)`

GetLinksByFactTypeOk returns a tuple with the LinksByFactType field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLinksByFactType

`func (o *BankStatsResponse) SetLinksByFactType(v map[string]int32)`

SetLinksByFactType sets LinksByFactType field to given value.


### GetLinksBreakdown

`func (o *BankStatsResponse) GetLinksBreakdown() map[string]map[string]int32`

GetLinksBreakdown returns the LinksBreakdown field if non-nil, zero value otherwise.

### GetLinksBreakdownOk

`func (o *BankStatsResponse) GetLinksBreakdownOk() (*map[string]map[string]int32, bool)`

GetLinksBreakdownOk returns a tuple with the LinksBreakdown field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLinksBreakdown

`func (o *BankStatsResponse) SetLinksBreakdown(v map[string]map[string]int32)`

SetLinksBreakdown sets LinksBreakdown field to given value.


### GetPendingOperations

`func (o *BankStatsResponse) GetPendingOperations() int32`

GetPendingOperations returns the PendingOperations field if non-nil, zero value otherwise.

### GetPendingOperationsOk

`func (o *BankStatsResponse) GetPendingOperationsOk() (*int32, bool)`

GetPendingOperationsOk returns a tuple with the PendingOperations field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetPendingOperations

`func (o *BankStatsResponse) SetPendingOperations(v int32)`

SetPendingOperations sets PendingOperations field to given value.


### GetFailedOperations

`func (o *BankStatsResponse) GetFailedOperations() int32`

GetFailedOperations returns the FailedOperations field if non-nil, zero value otherwise.

### GetFailedOperationsOk

`func (o *BankStatsResponse) GetFailedOperationsOk() (*int32, bool)`

GetFailedOperationsOk returns a tuple with the FailedOperations field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetFailedOperations

`func (o *BankStatsResponse) SetFailedOperations(v int32)`

SetFailedOperations sets FailedOperations field to given value.


### GetLastConsolidatedAt

`func (o *BankStatsResponse) GetLastConsolidatedAt() string`

GetLastConsolidatedAt returns the LastConsolidatedAt field if non-nil, zero value otherwise.

### GetLastConsolidatedAtOk

`func (o *BankStatsResponse) GetLastConsolidatedAtOk() (*string, bool)`

GetLastConsolidatedAtOk returns a tuple with the LastConsolidatedAt field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLastConsolidatedAt

`func (o *BankStatsResponse) SetLastConsolidatedAt(v string)`

SetLastConsolidatedAt sets LastConsolidatedAt field to given value.

### HasLastConsolidatedAt

`func (o *BankStatsResponse) HasLastConsolidatedAt() bool`

HasLastConsolidatedAt returns a boolean if a field has been set.

### SetLastConsolidatedAtNil

`func (o *BankStatsResponse) SetLastConsolidatedAtNil(b bool)`

 SetLastConsolidatedAtNil sets the value for LastConsolidatedAt to be an explicit nil

### UnsetLastConsolidatedAt
`func (o *BankStatsResponse) UnsetLastConsolidatedAt()`

UnsetLastConsolidatedAt ensures that no value is present for LastConsolidatedAt, not even an explicit nil
### GetPendingConsolidation

`func (o *BankStatsResponse) GetPendingConsolidation() int32`

GetPendingConsolidation returns the PendingConsolidation field if non-nil, zero value otherwise.

### GetPendingConsolidationOk

`func (o *BankStatsResponse) GetPendingConsolidationOk() (*int32, bool)`

GetPendingConsolidationOk returns a tuple with the PendingConsolidation field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetPendingConsolidation

`func (o *BankStatsResponse) SetPendingConsolidation(v int32)`

SetPendingConsolidation sets PendingConsolidation field to given value.

### HasPendingConsolidation

`func (o *BankStatsResponse) HasPendingConsolidation() bool`

HasPendingConsolidation returns a boolean if a field has been set.

### GetTotalObservations

`func (o *BankStatsResponse) GetTotalObservations() int32`

GetTotalObservations returns the TotalObservations field if non-nil, zero value otherwise.

### GetTotalObservationsOk

`func (o *BankStatsResponse) GetTotalObservationsOk() (*int32, bool)`

GetTotalObservationsOk returns a tuple with the TotalObservations field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotalObservations

`func (o *BankStatsResponse) SetTotalObservations(v int32)`

SetTotalObservations sets TotalObservations field to given value.

### HasTotalObservations

`func (o *BankStatsResponse) HasTotalObservations() bool`

HasTotalObservations returns a boolean if a field has been set.


[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


