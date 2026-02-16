# GraphDataResponse

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**Nodes** | **[]map[string]interface{}** |  | 
**Edges** | **[]map[string]interface{}** |  | 
**TableRows** | **[]map[string]interface{}** |  | 
**TotalUnits** | **int32** |  | 
**Limit** | **int32** |  | 

## Methods

### NewGraphDataResponse

`func NewGraphDataResponse(nodes []map[string]interface{}, edges []map[string]interface{}, tableRows []map[string]interface{}, totalUnits int32, limit int32, ) *GraphDataResponse`

NewGraphDataResponse instantiates a new GraphDataResponse object
This constructor will assign default values to properties that have it defined,
and makes sure properties required by API are set, but the set of arguments
will change when the set of required properties is changed

### NewGraphDataResponseWithDefaults

`func NewGraphDataResponseWithDefaults() *GraphDataResponse`

NewGraphDataResponseWithDefaults instantiates a new GraphDataResponse object
This constructor will only assign default values to properties that have it defined,
but it doesn't guarantee that properties required by API are set

### GetNodes

`func (o *GraphDataResponse) GetNodes() []map[string]interface{}`

GetNodes returns the Nodes field if non-nil, zero value otherwise.

### GetNodesOk

`func (o *GraphDataResponse) GetNodesOk() (*[]map[string]interface{}, bool)`

GetNodesOk returns a tuple with the Nodes field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetNodes

`func (o *GraphDataResponse) SetNodes(v []map[string]interface{})`

SetNodes sets Nodes field to given value.


### GetEdges

`func (o *GraphDataResponse) GetEdges() []map[string]interface{}`

GetEdges returns the Edges field if non-nil, zero value otherwise.

### GetEdgesOk

`func (o *GraphDataResponse) GetEdgesOk() (*[]map[string]interface{}, bool)`

GetEdgesOk returns a tuple with the Edges field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetEdges

`func (o *GraphDataResponse) SetEdges(v []map[string]interface{})`

SetEdges sets Edges field to given value.


### GetTableRows

`func (o *GraphDataResponse) GetTableRows() []map[string]interface{}`

GetTableRows returns the TableRows field if non-nil, zero value otherwise.

### GetTableRowsOk

`func (o *GraphDataResponse) GetTableRowsOk() (*[]map[string]interface{}, bool)`

GetTableRowsOk returns a tuple with the TableRows field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTableRows

`func (o *GraphDataResponse) SetTableRows(v []map[string]interface{})`

SetTableRows sets TableRows field to given value.


### GetTotalUnits

`func (o *GraphDataResponse) GetTotalUnits() int32`

GetTotalUnits returns the TotalUnits field if non-nil, zero value otherwise.

### GetTotalUnitsOk

`func (o *GraphDataResponse) GetTotalUnitsOk() (*int32, bool)`

GetTotalUnitsOk returns a tuple with the TotalUnits field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetTotalUnits

`func (o *GraphDataResponse) SetTotalUnits(v int32)`

SetTotalUnits sets TotalUnits field to given value.


### GetLimit

`func (o *GraphDataResponse) GetLimit() int32`

GetLimit returns the Limit field if non-nil, zero value otherwise.

### GetLimitOk

`func (o *GraphDataResponse) GetLimitOk() (*int32, bool)`

GetLimitOk returns a tuple with the Limit field if it's non-nil, zero value otherwise
and a boolean to check if the value has been set.

### SetLimit

`func (o *GraphDataResponse) SetLimit(v int32)`

SetLimit sets Limit field to given value.



[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


