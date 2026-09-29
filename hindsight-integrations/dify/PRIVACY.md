# Privacy Policy

The Hindsight Dify plugin sends data to a Hindsight API instance (Hindsight Cloud or self-hosted) configured by the workflow author.

## Data sent

- **Retain**: the content you pass to the tool, plus any tags, are sent to Hindsight to be stored as memories. Hindsight extracts facts from the content using an LLM.
- **Recall** / **Reflect**: the query string is sent to Hindsight; relevant memories from the configured bank are returned.

## What is stored

- All retained content, extracted facts, and metadata are stored in the Hindsight instance you configure.
- The plugin itself does not retain any data — it is a thin adapter to Hindsight's API.

## Where data goes

- **Hindsight Cloud (`https://api.hindsight.vectorize.io`)**: data is processed and stored by Vectorize, Inc. See [https://hindsight.vectorize.io/privacy](https://hindsight.vectorize.io/privacy).
- **Self-hosted Hindsight**: data stays inside your own infrastructure.

## Credentials

The API key you configure is stored as a Dify secret credential and is sent to Hindsight on each tool invocation in the `Authorization: Bearer` header.
