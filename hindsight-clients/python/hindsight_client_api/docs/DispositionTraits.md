# DispositionTraits

Disposition traits that influence how memories are formed and interpreted.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**skepticism** | **int** | How skeptical vs trusting (1&#x3D;trusting, 5&#x3D;skeptical) | 
**literalism** | **int** | How literally to interpret information (1&#x3D;flexible, 5&#x3D;literal) | 
**empathy** | **int** | How much to consider emotional context (1&#x3D;detached, 5&#x3D;empathetic) | 

## Example

```python
from hindsight_client_api.models.disposition_traits import DispositionTraits

# TODO update the JSON string below
json = "{}"
# create an instance of DispositionTraits from a JSON string
disposition_traits_instance = DispositionTraits.from_json(json)
# print the JSON string representation of the object
print(DispositionTraits.to_json())

# convert the object into a dict
disposition_traits_dict = disposition_traits_instance.to_dict()
# create an instance of DispositionTraits from a dict
disposition_traits_from_dict = DispositionTraits.from_dict(disposition_traits_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


