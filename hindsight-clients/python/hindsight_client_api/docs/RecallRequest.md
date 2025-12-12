# RecallRequest

Request model for recall endpoint.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**query** | **str** |  | 
**types** | **List[str]** |  | [optional] 
**budget** | [**Budget**](Budget.md) |  | [optional] 
**max_tokens** | **int** |  | [optional] [default to 4096]
**trace** | **bool** |  | [optional] [default to False]
**query_timestamp** | **str** |  | [optional] 
**include** | [**IncludeOptions**](IncludeOptions.md) | Options for including additional data (entities are included by default) | [optional] 

## Example

```python
from hindsight_client_api.models.recall_request import RecallRequest

# TODO update the JSON string below
json = "{}"
# create an instance of RecallRequest from a JSON string
recall_request_instance = RecallRequest.from_json(json)
# print the JSON string representation of the object
print(RecallRequest.to_json())

# convert the object into a dict
recall_request_dict = recall_request_instance.to_dict()
# create an instance of RecallRequest from a dict
recall_request_from_dict = RecallRequest.from_dict(recall_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


