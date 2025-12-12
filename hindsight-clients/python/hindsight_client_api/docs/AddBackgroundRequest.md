# AddBackgroundRequest

Request model for adding/merging background information.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**content** | **str** | New background information to add or merge | 
**update_disposition** | **bool** | If true, infer disposition traits from the merged background (default: true) | [optional] [default to True]

## Example

```python
from hindsight_client_api.models.add_background_request import AddBackgroundRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AddBackgroundRequest from a JSON string
add_background_request_instance = AddBackgroundRequest.from_json(json)
# print the JSON string representation of the object
print(AddBackgroundRequest.to_json())

# convert the object into a dict
add_background_request_dict = add_background_request_instance.to_dict()
# create an instance of AddBackgroundRequest from a dict
add_background_request_from_dict = AddBackgroundRequest.from_dict(add_background_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


