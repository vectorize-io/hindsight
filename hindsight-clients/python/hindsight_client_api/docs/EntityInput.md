# EntityInput

Entity to associate with retained content.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**text** | **str** | The entity name/text | 
**type** | **str** |  | [optional] 

## Example

```python
from hindsight_client_api.models.entity_input import EntityInput

# TODO update the JSON string below
json = "{}"
# create an instance of EntityInput from a JSON string
entity_input_instance = EntityInput.from_json(json)
# print the JSON string representation of the object
print(EntityInput.to_json())

# convert the object into a dict
entity_input_dict = entity_input_instance.to_dict()
# create an instance of EntityInput from a dict
entity_input_from_dict = EntityInput.from_dict(entity_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


