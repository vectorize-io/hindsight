# MemoryItem

Single memory item for retain.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**content** | **str** |  | 
**timestamp** | **datetime** |  | [optional] 
**context** | **str** |  | [optional] 
**metadata** | **Dict[str, str]** |  | [optional] 
**document_id** | **str** |  | [optional] 
**entities** | [**List[EntityInput]**](EntityInput.md) |  | [optional] 

## Example

```python
from hindsight_client_api.models.memory_item import MemoryItem

# TODO update the JSON string below
json = "{}"
# create an instance of MemoryItem from a JSON string
memory_item_instance = MemoryItem.from_json(json)
# print the JSON string representation of the object
print(MemoryItem.to_json())

# convert the object into a dict
memory_item_dict = memory_item_instance.to_dict()
# create an instance of MemoryItem from a dict
memory_item_from_dict = MemoryItem.from_dict(memory_item_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


