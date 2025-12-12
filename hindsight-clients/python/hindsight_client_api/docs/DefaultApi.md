# hindsight_client_api.DefaultApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**add_bank_background**](DefaultApi.md#add_bank_background) | **POST** /v1/default/banks/{bank_id}/background | Add/merge memory bank background
[**cancel_operation**](DefaultApi.md#cancel_operation) | **DELETE** /v1/default/banks/{bank_id}/operations/{operation_id} | Cancel a pending async operation
[**clear_bank_memories**](DefaultApi.md#clear_bank_memories) | **DELETE** /v1/default/banks/{bank_id}/memories | Clear memory bank memories
[**create_or_update_bank**](DefaultApi.md#create_or_update_bank) | **PUT** /v1/default/banks/{bank_id} | Create or update memory bank
[**delete_document**](DefaultApi.md#delete_document) | **DELETE** /v1/default/banks/{bank_id}/documents/{document_id} | Delete a document
[**get_agent_stats**](DefaultApi.md#get_agent_stats) | **GET** /v1/default/banks/{bank_id}/stats | Get statistics for memory bank
[**get_bank_profile**](DefaultApi.md#get_bank_profile) | **GET** /v1/default/banks/{bank_id}/profile | Get memory bank profile
[**get_chunk**](DefaultApi.md#get_chunk) | **GET** /v1/default/chunks/{chunk_id} | Get chunk details
[**get_document**](DefaultApi.md#get_document) | **GET** /v1/default/banks/{bank_id}/documents/{document_id} | Get document details
[**get_entity**](DefaultApi.md#get_entity) | **GET** /v1/default/banks/{bank_id}/entities/{entity_id} | Get entity details
[**get_graph**](DefaultApi.md#get_graph) | **GET** /v1/default/banks/{bank_id}/graph | Get memory graph data
[**list_banks**](DefaultApi.md#list_banks) | **GET** /v1/default/banks | List all memory banks
[**list_documents**](DefaultApi.md#list_documents) | **GET** /v1/default/banks/{bank_id}/documents | List documents
[**list_entities**](DefaultApi.md#list_entities) | **GET** /v1/default/banks/{bank_id}/entities | List entities
[**list_memories**](DefaultApi.md#list_memories) | **GET** /v1/default/banks/{bank_id}/memories/list | List memory units
[**list_operations**](DefaultApi.md#list_operations) | **GET** /v1/default/banks/{bank_id}/operations | List async operations
[**recall_memories**](DefaultApi.md#recall_memories) | **POST** /v1/default/banks/{bank_id}/memories/recall | Recall memory
[**reflect**](DefaultApi.md#reflect) | **POST** /v1/default/banks/{bank_id}/reflect | Reflect and generate answer
[**regenerate_entity_observations**](DefaultApi.md#regenerate_entity_observations) | **POST** /v1/default/banks/{bank_id}/entities/{entity_id}/regenerate | Regenerate entity observations
[**retain_memories**](DefaultApi.md#retain_memories) | **POST** /v1/default/banks/{bank_id}/memories | Retain memories
[**update_bank_disposition**](DefaultApi.md#update_bank_disposition) | **PUT** /v1/default/banks/{bank_id}/profile | Update memory bank disposition


# **add_bank_background**
> BackgroundResponse add_bank_background(bank_id, add_background_request)

Add/merge memory bank background

Add new background information or merge with existing. LLM intelligently resolves conflicts, normalizes to first person, and optionally infers disposition traits.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.add_background_request import AddBackgroundRequest
from hindsight_client_api.models.background_response import BackgroundResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    add_background_request = hindsight_client_api.AddBackgroundRequest() # AddBackgroundRequest | 

    try:
        # Add/merge memory bank background
        api_response = await api_instance.add_bank_background(bank_id, add_background_request)
        print("The response of DefaultApi->add_bank_background:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->add_bank_background: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **add_background_request** | [**AddBackgroundRequest**](AddBackgroundRequest.md)|  | 

### Return type

[**BackgroundResponse**](BackgroundResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **cancel_operation**
> object cancel_operation(bank_id, operation_id)

Cancel a pending async operation

Cancel a pending async operation by removing it from the queue

### Example


```python
import hindsight_client_api
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    operation_id = 'operation_id_example' # str | 

    try:
        # Cancel a pending async operation
        api_response = await api_instance.cancel_operation(bank_id, operation_id)
        print("The response of DefaultApi->cancel_operation:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->cancel_operation: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **operation_id** | **str**|  | 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **clear_bank_memories**
> DeleteResponse clear_bank_memories(bank_id, type=type)

Clear memory bank memories

Delete memory units for a memory bank. Optionally filter by type (world, experience, opinion) to delete only specific types. This is a destructive operation that cannot be undone. The bank profile (disposition and background) will be preserved.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.delete_response import DeleteResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    type = 'type_example' # str | Optional fact type filter (world, experience, opinion) (optional)

    try:
        # Clear memory bank memories
        api_response = await api_instance.clear_bank_memories(bank_id, type=type)
        print("The response of DefaultApi->clear_bank_memories:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->clear_bank_memories: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **type** | **str**| Optional fact type filter (world, experience, opinion) | [optional] 

### Return type

[**DeleteResponse**](DeleteResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_or_update_bank**
> BankProfileResponse create_or_update_bank(bank_id, create_bank_request)

Create or update memory bank

Create a new agent or update existing agent with disposition and background. Auto-fills missing fields with defaults.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.models.create_bank_request import CreateBankRequest
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    create_bank_request = hindsight_client_api.CreateBankRequest() # CreateBankRequest | 

    try:
        # Create or update memory bank
        api_response = await api_instance.create_or_update_bank(bank_id, create_bank_request)
        print("The response of DefaultApi->create_or_update_bank:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->create_or_update_bank: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **create_bank_request** | [**CreateBankRequest**](CreateBankRequest.md)|  | 

### Return type

[**BankProfileResponse**](BankProfileResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_document**
> object delete_document(bank_id, document_id)

Delete a document

Delete a document and all its associated memory units and links.

This will cascade delete:
- The document itself
- All memory units extracted from this document
- All links (temporal, semantic, entity) associated with those memory units

This operation cannot be undone.

### Example


```python
import hindsight_client_api
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    document_id = 'document_id_example' # str | 

    try:
        # Delete a document
        api_response = await api_instance.delete_document(bank_id, document_id)
        print("The response of DefaultApi->delete_document:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->delete_document: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **document_id** | **str**|  | 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_agent_stats**
> object get_agent_stats(bank_id)

Get statistics for memory bank

Get statistics about nodes and links for a specific agent

### Example


```python
import hindsight_client_api
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 

    try:
        # Get statistics for memory bank
        api_response = await api_instance.get_agent_stats(bank_id)
        print("The response of DefaultApi->get_agent_stats:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_agent_stats: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_bank_profile**
> BankProfileResponse get_bank_profile(bank_id)

Get memory bank profile

Get disposition traits and background for a memory bank. Auto-creates agent with defaults if not exists.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 

    try:
        # Get memory bank profile
        api_response = await api_instance.get_bank_profile(bank_id)
        print("The response of DefaultApi->get_bank_profile:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_bank_profile: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 

### Return type

[**BankProfileResponse**](BankProfileResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_chunk**
> ChunkResponse get_chunk(chunk_id)

Get chunk details

Get a specific chunk by its ID

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.chunk_response import ChunkResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    chunk_id = 'chunk_id_example' # str | 

    try:
        # Get chunk details
        api_response = await api_instance.get_chunk(chunk_id)
        print("The response of DefaultApi->get_chunk:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_chunk: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **chunk_id** | **str**|  | 

### Return type

[**ChunkResponse**](ChunkResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_document**
> DocumentResponse get_document(bank_id, document_id)

Get document details

Get a specific document including its original text

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.document_response import DocumentResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    document_id = 'document_id_example' # str | 

    try:
        # Get document details
        api_response = await api_instance.get_document(bank_id, document_id)
        print("The response of DefaultApi->get_document:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_document: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **document_id** | **str**|  | 

### Return type

[**DocumentResponse**](DocumentResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_entity**
> EntityDetailResponse get_entity(bank_id, entity_id)

Get entity details

Get detailed information about an entity including observations (mental model).

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.entity_detail_response import EntityDetailResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    entity_id = 'entity_id_example' # str | 

    try:
        # Get entity details
        api_response = await api_instance.get_entity(bank_id, entity_id)
        print("The response of DefaultApi->get_entity:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_entity: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **entity_id** | **str**|  | 

### Return type

[**EntityDetailResponse**](EntityDetailResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_graph**
> GraphDataResponse get_graph(bank_id, type=type)

Get memory graph data

Retrieve graph data for visualization, optionally filtered by type (world/experience/opinion). Limited to 1000 most recent items.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.graph_data_response import GraphDataResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    type = 'type_example' # str |  (optional)

    try:
        # Get memory graph data
        api_response = await api_instance.get_graph(bank_id, type=type)
        print("The response of DefaultApi->get_graph:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_graph: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **type** | **str**|  | [optional] 

### Return type

[**GraphDataResponse**](GraphDataResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_banks**
> BankListResponse list_banks()

List all memory banks

Get a list of all agents with their profiles

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.bank_list_response import BankListResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)

    try:
        # List all memory banks
        api_response = await api_instance.list_banks()
        print("The response of DefaultApi->list_banks:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->list_banks: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

[**BankListResponse**](BankListResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_documents**
> ListDocumentsResponse list_documents(bank_id, q=q, limit=limit, offset=offset)

List documents

List documents with pagination and optional search. Documents are the source content from which memory units are extracted.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.list_documents_response import ListDocumentsResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    q = 'q_example' # str |  (optional)
    limit = 100 # int |  (optional) (default to 100)
    offset = 0 # int |  (optional) (default to 0)

    try:
        # List documents
        api_response = await api_instance.list_documents(bank_id, q=q, limit=limit, offset=offset)
        print("The response of DefaultApi->list_documents:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->list_documents: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **q** | **str**|  | [optional] 
 **limit** | **int**|  | [optional] [default to 100]
 **offset** | **int**|  | [optional] [default to 0]

### Return type

[**ListDocumentsResponse**](ListDocumentsResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_entities**
> EntityListResponse list_entities(bank_id, limit=limit)

List entities

List all entities (people, organizations, etc.) known by the bank, ordered by mention count.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.entity_list_response import EntityListResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    limit = 100 # int | Maximum number of entities to return (optional) (default to 100)

    try:
        # List entities
        api_response = await api_instance.list_entities(bank_id, limit=limit)
        print("The response of DefaultApi->list_entities:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->list_entities: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **limit** | **int**| Maximum number of entities to return | [optional] [default to 100]

### Return type

[**EntityListResponse**](EntityListResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_memories**
> ListMemoryUnitsResponse list_memories(bank_id, type=type, q=q, limit=limit, offset=offset)

List memory units

List memory units with pagination and optional full-text search. Supports filtering by type. Results are sorted by most recent first (mentioned_at DESC, then created_at DESC).

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.list_memory_units_response import ListMemoryUnitsResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    type = 'type_example' # str |  (optional)
    q = 'q_example' # str |  (optional)
    limit = 100 # int |  (optional) (default to 100)
    offset = 0 # int |  (optional) (default to 0)

    try:
        # List memory units
        api_response = await api_instance.list_memories(bank_id, type=type, q=q, limit=limit, offset=offset)
        print("The response of DefaultApi->list_memories:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->list_memories: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **type** | **str**|  | [optional] 
 **q** | **str**|  | [optional] 
 **limit** | **int**|  | [optional] [default to 100]
 **offset** | **int**|  | [optional] [default to 0]

### Return type

[**ListMemoryUnitsResponse**](ListMemoryUnitsResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_operations**
> object list_operations(bank_id)

List async operations

Get a list of all async operations (pending and failed) for a specific agent, including error messages for failed operations

### Example


```python
import hindsight_client_api
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 

    try:
        # List async operations
        api_response = await api_instance.list_operations(bank_id)
        print("The response of DefaultApi->list_operations:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->list_operations: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **recall_memories**
> RecallResponse recall_memories(bank_id, recall_request)

Recall memory

Recall memory using semantic similarity and spreading activation.

    The type parameter is optional and must be one of:
    - 'world': General knowledge about people, places, events, and things that happen
    - 'experience': Memories about experience, conversations, actions taken, and tasks performed
    - 'opinion': The bank's formed beliefs, perspectives, and viewpoints

    Set include_entities=true to get entity observations alongside recall results.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.recall_request import RecallRequest
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    recall_request = hindsight_client_api.RecallRequest() # RecallRequest | 

    try:
        # Recall memory
        api_response = await api_instance.recall_memories(bank_id, recall_request)
        print("The response of DefaultApi->recall_memories:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->recall_memories: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **recall_request** | [**RecallRequest**](RecallRequest.md)|  | 

### Return type

[**RecallResponse**](RecallResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **reflect**
> ReflectResponse reflect(bank_id, reflect_request)

Reflect and generate answer

Reflect and formulate an answer using bank identity, world facts, and opinions.

    This endpoint:
    1. Retrieves experience (conversations and events)
    2. Retrieves world facts relevant to the query
    3. Retrieves existing opinions (bank's perspectives)
    4. Uses LLM to formulate a contextual answer
    5. Extracts and stores any new opinions formed
    6. Returns plain text answer, the facts used, and new opinions

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.reflect_request import ReflectRequest
from hindsight_client_api.models.reflect_response import ReflectResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    reflect_request = hindsight_client_api.ReflectRequest() # ReflectRequest | 

    try:
        # Reflect and generate answer
        api_response = await api_instance.reflect(bank_id, reflect_request)
        print("The response of DefaultApi->reflect:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->reflect: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **reflect_request** | [**ReflectRequest**](ReflectRequest.md)|  | 

### Return type

[**ReflectResponse**](ReflectResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **regenerate_entity_observations**
> EntityDetailResponse regenerate_entity_observations(bank_id, entity_id)

Regenerate entity observations

Regenerate observations for an entity based on all facts mentioning it.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.entity_detail_response import EntityDetailResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    entity_id = 'entity_id_example' # str | 

    try:
        # Regenerate entity observations
        api_response = await api_instance.regenerate_entity_observations(bank_id, entity_id)
        print("The response of DefaultApi->regenerate_entity_observations:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->regenerate_entity_observations: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **entity_id** | **str**|  | 

### Return type

[**EntityDetailResponse**](EntityDetailResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **retain_memories**
> RetainResponse retain_memories(bank_id, retain_request)

Retain memories

Retain memory items with automatic fact extraction.

    This is the main endpoint for storing memories. It supports both synchronous and asynchronous processing
    via the async parameter.

    Features:
    - Efficient batch processing
    - Automatic fact extraction from natural language
    - Entity recognition and linking
    - Document tracking with automatic upsert (when document_id is provided on items)
    - Temporal and semantic linking
    - Optional asynchronous processing

    The system automatically:
    1. Extracts semantic facts from the content
    2. Generates embeddings
    3. Deduplicates similar facts
    4. Creates temporal, semantic, and entity links
    5. Tracks document metadata

    When async=true:
    - Returns immediately after queuing the task
    - Processing happens in the background
    - Use the operations endpoint to monitor progress

    When async=false (default):
    - Waits for processing to complete
    - Returns after all memories are stored

    Note: If a memory item has a document_id that already exists, the old document and its memory units will be deleted before creating new ones (upsert behavior). Items with the same document_id are grouped together for efficient processing.

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.retain_request import RetainRequest
from hindsight_client_api.models.retain_response import RetainResponse
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    retain_request = hindsight_client_api.RetainRequest() # RetainRequest | 

    try:
        # Retain memories
        api_response = await api_instance.retain_memories(bank_id, retain_request)
        print("The response of DefaultApi->retain_memories:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->retain_memories: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **retain_request** | [**RetainRequest**](RetainRequest.md)|  | 

### Return type

[**RetainResponse**](RetainResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_bank_disposition**
> BankProfileResponse update_bank_disposition(bank_id, update_disposition_request)

Update memory bank disposition

Update bank's disposition traits (skepticism, literalism, empathy)

### Example


```python
import hindsight_client_api
from hindsight_client_api.models.bank_profile_response import BankProfileResponse
from hindsight_client_api.models.update_disposition_request import UpdateDispositionRequest
from hindsight_client_api.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = hindsight_client_api.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
async with hindsight_client_api.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = hindsight_client_api.DefaultApi(api_client)
    bank_id = 'bank_id_example' # str | 
    update_disposition_request = hindsight_client_api.UpdateDispositionRequest() # UpdateDispositionRequest | 

    try:
        # Update memory bank disposition
        api_response = await api_instance.update_bank_disposition(bank_id, update_disposition_request)
        print("The response of DefaultApi->update_bank_disposition:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->update_bank_disposition: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bank_id** | **str**|  | 
 **update_disposition_request** | [**UpdateDispositionRequest**](UpdateDispositionRequest.md)|  | 

### Return type

[**BankProfileResponse**](BankProfileResponse.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

