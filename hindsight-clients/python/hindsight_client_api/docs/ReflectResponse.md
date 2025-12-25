# ReflectResponse

Response model for think endpoint.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**text** | **str** |  | 
**based_on** | [**List[ReflectFact]**](ReflectFact.md) |  | [optional] [default to []]
**structured_output** | **Dict[str, object]** |  | [optional] 

## Example

```python
from hindsight_client_api.models.reflect_response import ReflectResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ReflectResponse from a JSON string
reflect_response_instance = ReflectResponse.from_json(json)
# print the JSON string representation of the object
print(ReflectResponse.to_json())

# convert the object into a dict
reflect_response_dict = reflect_response_instance.to_dict()
# create an instance of ReflectResponse from a dict
reflect_response_from_dict = ReflectResponse.from_dict(reflect_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


