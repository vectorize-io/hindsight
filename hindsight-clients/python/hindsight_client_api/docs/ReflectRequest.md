# ReflectRequest

Request model for reflect endpoint.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**query** | **str** |  | 
**budget** | [**Budget**](Budget.md) |  | [optional] 
**context** | **str** |  | [optional] 
**include** | [**ReflectIncludeOptions**](ReflectIncludeOptions.md) | Options for including additional data (disabled by default) | [optional] 

## Example

```python
from hindsight_client_api.models.reflect_request import ReflectRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ReflectRequest from a JSON string
reflect_request_instance = ReflectRequest.from_json(json)
# print the JSON string representation of the object
print(ReflectRequest.to_json())

# convert the object into a dict
reflect_request_dict = reflect_request_instance.to_dict()
# create an instance of ReflectRequest from a dict
reflect_request_from_dict = ReflectRequest.from_dict(reflect_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


