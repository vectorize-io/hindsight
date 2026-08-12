---
sidebar_position: 3
---

# Go Client

Official Go client for the Hindsight API, generated from the OpenAPI 3.1 spec using [OpenAPI Generator](https://github.com/OpenAPITools/openapi-generator).

import CodeSnippet from '@site/src/components/CodeSnippet';
import quickstartGo from '!!raw-loader!@site/examples/api/quickstart.go';

## Installation

```bash
go get github.com/vectorize-io/hindsight/hindsight-clients/go
```

Requires Go 1.23+.

## Quick Start

<CodeSnippet code={quickstartGo} section="quickstart-full" language="go" />

## API Structure

The Go client provides access to all Hindsight API operations through structured namespaces:

- **`client.MemoryAPI`** - Retain, recall, reflect operations
- **`client.BanksAPI`** - Bank management
- **`client.DirectivesAPI`** - Directive management
- **`client.MentalModelsAPI`** - Mental model management
- **`client.DocumentsAPI`** - Document operations
- **`client.EntitiesAPI`** - Entity operations
- **`client.OperationsAPI`** - Async operation monitoring

## Working with Nullable Fields

The Go client uses `NullableString`, `NullableTime`, and similar types for optional fields:

<CodeSnippet code={quickstartGo} section="nullable-fields" language="go" />

## Error Handling

<CodeSnippet code={quickstartGo} section="error-handling" language="go" />

## More Examples

For detailed examples of all operations, see:
- [Python SDK documentation](./python.md) - API concepts are the same
- [Node.js SDK documentation](./nodejs.md) - API concepts are the same
- [OpenAPI specification](https://hindsight.dev/openapi.json) - Complete API reference
