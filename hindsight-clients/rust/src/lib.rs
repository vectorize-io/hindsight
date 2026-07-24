//! Hindsight API Client
//!
//! A Rust client library for the Hindsight semantic memory system API.
//!
//! # Example
//!
//! ```rust,no_run
//! use hindsight_client::Client;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::new("http://localhost:8888");
//!
//!     // List memory banks
//!     let banks = client.list_banks(None).await?;
//!     println!("Found {} banks", banks.into_inner().banks.len());
//!
//!     Ok(())
//! }
//! ```

// Include the generated client code (which already exports Error and ResponseValue)
include!(concat!(env!("OUT_DIR"), "/hindsight_client_generated.rs"));

/// Semantic version of this Rust client, kept in sync with the other language
/// wrappers when a coordinated release is cut.
pub const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default `User-Agent` header sent on every request unless overridden.
pub const DEFAULT_USER_AGENT: &str = concat!("hindsight-client-rust/", env!("CARGO_PKG_VERSION"));

/// Build a [`reqwest::Client`] with the given `User-Agent` header.
///
/// Integrations should use this to identify themselves (e.g.
/// `"hindsight-cli/0.6.2"`) so self-hosted deployments behind Cloudflare or
/// other UA-based filters accept the traffic. Pass the resulting client to
/// [`Client::new_with_client`].
pub fn reqwest_client_with_user_agent(
    user_agent: impl Into<String>,
) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .user_agent(user_agent.into())
        .build()
}

/// Construct a [`Client`] with a custom `User-Agent` header.
///
/// Equivalent to [`Client::new`] but sets the UA string. Use this instead of
/// the bare `Client::new` when pointing at a hosted Hindsight deployment.
pub fn client_with_user_agent(
    base_url: &str,
    user_agent: impl Into<String>,
) -> Result<Client, reqwest::Error> {
    let http = reqwest_client_with_user_agent(user_agent)?;
    Ok(Client::new_with_client(base_url, http))
}

/// Construct a [`Client`] with the default Hindsight `User-Agent`.
///
/// Prefer this over `Client::new` — the bare `Client::new` uses reqwest's
/// default UA which is blocked by some reverse proxies (e.g. Cloudflare).
pub fn default_client(base_url: &str) -> Result<Client, reqwest::Error> {
    client_with_user_agent(base_url, DEFAULT_USER_AGENT)
}

/// One image part for [`Client::retain_images`].
#[derive(Debug)]
pub struct ImageRetainFile {
    pub filename: String,
    pub mime_type: String,
    pub content: Vec<u8>,
}

/// A managed image descriptor and bytes returned by one authenticated request.
#[derive(Clone, Debug)]
pub struct DownloadedImageAsset {
    pub descriptor: types::ImageAssetDescriptor,
    pub content: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageClientError {
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error("invalid image metadata response: {0}")]
    Metadata(String),
    #[error("image API returned HTTP {status}: {body}")]
    Api { status: u16, body: String },
}

#[derive(Debug, thiserror::Error)]
pub enum TransferClientError {
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("document transfer API returned HTTP {status}: {body}")]
    Api { status: u16, body: String },
}

fn encode_path_segment(value: &str) -> String {
    percent_encoding::utf8_percent_encode(value, percent_encoding::NON_ALPHANUMERIC).to_string()
}

impl Client {
    /// Stream a Document Transfer v2 archive directly to a file.
    pub async fn export_documents_to_file(
        &self,
        bank_id: &str,
        destination: impl AsRef<std::path::Path>,
        authorization: Option<&str>,
    ) -> Result<(), TransferClientError> {
        use tokio::io::AsyncWriteExt;

        let url = format!(
            "{}/v1/default/banks/{}/document-transfer",
            self.baseurl,
            encode_path_segment(bank_id)
        );
        let mut builder = self.client.get(url);
        if let Some(value) = authorization {
            builder = builder.header(reqwest::header::AUTHORIZATION, value);
        }
        let mut response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(TransferClientError::Api {
                status: status.as_u16(),
                body: response.text().await?,
            });
        }
        let mut output = tokio::fs::File::create(destination).await?;
        while let Some(chunk) = response.chunk().await? {
            output.write_all(&chunk).await?;
        }
        output.flush().await?;
        Ok(())
    }

    /// Stream a file-backed Document Transfer v2 archive as multipart data.
    pub async fn import_documents_from_file(
        &self,
        bank_id: &str,
        source: impl AsRef<std::path::Path>,
        on_conflict: &str,
        authorization: Option<&str>,
    ) -> Result<types::DocumentImportSubmitResponse, TransferClientError> {
        let source = source.as_ref();
        let file = tokio::fs::File::open(source).await?;
        let length = file.metadata().await?.len();
        let filename = source
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("transfer.zip")
            .to_string();
        let stream = tokio_util::io::ReaderStream::new(file);
        let part = reqwest::multipart::Part::stream_with_length(
            reqwest::Body::wrap_stream(stream),
            length,
        )
        .file_name(filename)
        .mime_str("application/zip")?;
        let form = reqwest::multipart::Form::new().part("file", part);
        let url = format!(
            "{}/v1/default/banks/{}/document-transfer?on_conflict={}",
            self.baseurl,
            encode_path_segment(bank_id),
            encode_path_segment(on_conflict)
        );
        let mut builder = self.client.post(url).multipart(form);
        if let Some(value) = authorization {
            builder = builder.header(reqwest::header::AUTHORIZATION, value);
        }
        let response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(TransferClientError::Api {
                status: status.as_u16(),
                body: response.text().await?,
            });
        }
        Ok(response.json().await?)
    }

    /// Retain all files using one ordered multipart request.
    pub async fn retain_images(
        &self,
        bank_id: &str,
        files: impl IntoIterator<Item = ImageRetainFile>,
        request: &serde_json::Value,
        idempotency_key: Option<&str>,
        authorization: Option<&str>,
    ) -> Result<types::ImageRetainAccepted, ImageClientError> {
        let mut form = reqwest::multipart::Form::new().text("request", request.to_string());
        for file in files {
            let part = reqwest::multipart::Part::bytes(file.content)
                .file_name(file.filename)
                .mime_str(&file.mime_type)?;
            form = form.part("files", part);
        }
        let url = format!(
            "{}/v1/default/banks/{}/memories/image-retain",
            self.baseurl,
            encode_path_segment(bank_id)
        );
        let mut builder = self.client.post(url).multipart(form);
        if let Some(value) = idempotency_key {
            builder = builder.header("Idempotency-Key", value);
        }
        if let Some(value) = authorization {
            builder = builder.header(reqwest::header::AUTHORIZATION, value);
        }
        let response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(ImageClientError::Api {
                status: status.as_u16(),
                body: response.text().await?,
            });
        }
        Ok(response.json().await?)
    }

    /// Download one managed image and reconstruct its descriptor from headers.
    pub async fn get_image_asset(
        &self,
        bank_id: &str,
        asset_id: &str,
        authorization: Option<&str>,
    ) -> Result<DownloadedImageAsset, ImageClientError> {
        let url = format!(
            "{}/v1/default/banks/{}/image-assets/{}",
            self.baseurl,
            encode_path_segment(bank_id),
            encode_path_segment(asset_id)
        );
        let mut builder = self.client.get(url);
        if let Some(value) = authorization {
            builder = builder.header(reqwest::header::AUTHORIZATION, value);
        }
        let response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(ImageClientError::Api {
                status: status.as_u16(),
                body: response.text().await?,
            });
        }
        let headers = response.headers().clone();
        let required = |name: &'static str| -> Result<&str, ImageClientError> {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| ImageClientError::Metadata(format!("missing {name}")))
        };
        let parse_i64 = |name: &'static str| -> Result<i64, ImageClientError> {
            required(name)?
                .parse()
                .map_err(|_| ImageClientError::Metadata(format!("invalid {name}")))
        };
        let parse_date = |name: &'static str| {
            required(name)?
                .parse::<chrono::DateTime<chrono::Utc>>()
                .map_err(|_| ImageClientError::Metadata(format!("invalid {name}")))
        };
        let mime_type = required("content-type")?
            .split(';')
            .next()
            .unwrap_or_default()
            .to_string();
        let descriptor = types::ImageAssetDescriptor {
            asset_id: asset_id.to_string(),
            created_at: parse_date("x-hindsight-asset-created-at")?,
            document_ids: Vec::new(),
            height: parse_i64("x-hindsight-image-height")?,
            mime_type,
            sha256: required("x-hindsight-image-sha256")?.to_string(),
            size_bytes: parse_i64("content-length")?,
            status: types::ImageAssetStatus::Ready,
            updated_at: parse_date("x-hindsight-asset-updated-at")?,
            width: parse_i64("x-hindsight-image-width")?,
        };
        Ok(DownloadedImageAsset {
            descriptor,
            content: response.bytes().await?.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn one_shot_server(
        response: Vec<u8>,
    ) -> (String, tokio::sync::oneshot::Receiver<Vec<u8>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            let header_end = loop {
                let count = socket.read(&mut buffer).await.unwrap();
                request.extend_from_slice(&buffer[..count]);
                if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(str::to_owned)
                })
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            while request.len() < header_end + content_length {
                let count = socket.read(&mut buffer).await.unwrap();
                if count == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..count]);
            }
            // Some contract tests only care about the response. Dropping the
            // optional request observer must not abort the mock server before
            // it writes that response.
            let _ = sender.send(request);
            socket.write_all(&response).await.unwrap();
        });
        (format!("http://{address}"), receiver)
    }

    #[test]
    fn test_client_creation() {
        let _client = Client::new("http://localhost:8888");
        // Just verify we can create a client
        assert!(true);
    }

    #[tokio::test]
    async fn image_multipart_preserves_order_metadata_and_idempotency() {
        let body = r#"{"operation_id":"op","document_id":"doc","image_assets":[]}"#;
        let response = format!(
            "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (base_url, request) = one_shot_server(response.into_bytes()).await;
        let client = Client::new(&base_url);
        let files = vec![
            ImageRetainFile {
                filename: "first.jpg".into(),
                mime_type: "image/jpeg".into(),
                content: b"first".to_vec(),
            },
            ImageRetainFile {
                filename: "second.png".into(),
                mime_type: "image/png".into(),
                content: b"second".to_vec(),
            },
        ];
        let metadata = serde_json::json!({"images":[{"asset_id":"a/1"},{"asset_id":"b/2"}]});
        client
            .retain_images("bank", files, &metadata, Some("idem"), None)
            .await
            .unwrap();
        let raw = String::from_utf8_lossy(&request.await.unwrap()).to_string();
        assert!(raw.contains("Idempotency-Key: idem") || raw.contains("idempotency-key: idem"));
        assert!(raw.find("first").unwrap() < raw.find("second").unwrap());
        assert!(raw.contains("a/1") && raw.contains("b/2"));
    }

    #[tokio::test]
    async fn image_path_is_encoded_and_binary_headers_are_parsed() {
        let body = b"bytes";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nX-Hindsight-Image-SHA256: {}\r\nX-Hindsight-Image-Width: 8\r\nX-Hindsight-Image-Height: 6\r\nX-Hindsight-Asset-Created-At: 2026-01-01T00:00:00Z\r\nX-Hindsight-Asset-Updated-At: 2026-01-01T00:00:00Z\r\n\r\nbytes",
            body.len(),
            "a".repeat(64)
        );
        let (base_url, request) = one_shot_server(response.into_bytes()).await;
        let client = Client::new(&base_url);
        let downloaded = client
            .get_image_asset("bank", "folder/a.jpg", None)
            .await
            .unwrap();
        assert_eq!(downloaded.content, body);
        let raw = String::from_utf8_lossy(&request.await.unwrap()).to_string();
        assert!(raw.starts_with("GET /v1/default/banks/bank/image-assets/folder%2Fa%2Ejpg "));
    }

    #[tokio::test]
    async fn transfer_helpers_stream_file_backed_archives() {
        let archive = b"streamed-archive";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/zip\r\nContent-Length: {}\r\n\r\n{}",
            archive.len(),
            String::from_utf8_lossy(archive)
        );
        let (base_url, _) = one_shot_server(response.into_bytes()).await;
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("download.zip");
        Client::new(&base_url)
            .export_documents_to_file("bank", &destination, None)
            .await
            .unwrap();
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), archive);

        let body = r#"{"operation_id":"op","status":"pending"}"#;
        let response = format!(
            "HTTP/1.1 202 Accepted\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (base_url, request) = one_shot_server(response.into_bytes()).await;
        let source = directory.path().join("source.zip");
        tokio::fs::write(&source, archive).await.unwrap();
        let result = Client::new(&base_url)
            .import_documents_from_file("bank", &source, "skip", None)
            .await
            .unwrap();
        assert_eq!(result.operation_id, "op");
        let request = String::from_utf8_lossy(&request.await.unwrap()).to_string();
        assert!(request.contains("streamed-archive"));
    }

    #[tokio::test]
    async fn image_list_delete_and_recall_map_contract() {
        let descriptor = serde_json::json!({
            "asset_id":"a/1", "mime_type":"image/jpeg", "size_bytes":5,
            "sha256":"a".repeat(64), "width":8, "height":6, "status":"ready",
            "document_ids":["doc"],
            "created_at":"2026-01-01T00:00:00Z", "updated_at":"2026-01-01T00:00:00Z",
            "ordinal": 0
        });
        let body =
            serde_json::json!({"items":[descriptor.clone()],"total":1,"limit":100,"offset":0})
                .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let (base_url, _) = one_shot_server(response.into_bytes()).await;
        let listed = Client::new(&base_url)
            .list_image_assets("bank", Some("doc"), None, None, Some("ready"), None)
            .await
            .unwrap();
        assert_eq!(listed.into_inner().items[0].asset_id, "a/1");

        let response = b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n".to_vec();
        let (base_url, request) = one_shot_server(response).await;
        Client::new(&base_url)
            .delete_image_asset("bank", "a/1", None)
            .await
            .unwrap();
        let request = String::from_utf8_lossy(&request.await.unwrap()).to_string();
        assert!(request.starts_with("DELETE ") && request.contains("a%2F1"));

        let recall: types::RecallResponse = serde_json::from_value(serde_json::json!({
            "results": [], "image_assets": {"doc": [descriptor]}
        }))
        .unwrap();
        assert_eq!(recall.image_assets.unwrap()["doc"][0].asset_id, "a/1");
    }

    #[tokio::test]
    async fn test_memory_lifecycle() {
        let api_url = std::env::var("HINDSIGHT_API_URL")
            .unwrap_or_else(|_| "http://localhost:8888".to_string());

        // Use a custom reqwest client with longer timeout for LLM operations
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        let client = Client::new_with_client(&api_url, http_client);

        // Generate unique bank ID for this test
        let bank_id = format!("rust-test-{}", uuid::Uuid::new_v4());

        // 1. Create a bank
        let create_request = types::CreateBankRequest {
            name: Some(format!("Rust Test Bank")),
            ..Default::default()
        };
        let create_response = client
            .create_or_update_bank(&bank_id, None, &create_request)
            .await
            .expect("Failed to create bank");
        assert_eq!(create_response.into_inner().bank_id, bank_id);

        // 2. Retain some memories
        let retain_request = types::RetainRequest {
            async_: false,
            items: vec![
                types::MemoryItem {
                    content: "Alice is a software engineer at Google".to_string(),
                    context: None,
                    document_id: None,
                    metadata: None,
                    timestamp: None,
                    entities: None,
                    tags: None,
                    observation_scopes: None,
                    strategy: None,
                    update_mode: None,
                },
                types::MemoryItem {
                    content: "Bob works with Alice on the search team".to_string(),
                    context: None,
                    document_id: None,
                    metadata: None,
                    timestamp: None,
                    entities: None,
                    tags: None,
                    observation_scopes: None,
                    strategy: None,
                    update_mode: None,
                },
            ],
            document_tags: None,
        };
        let retain_response = client
            .retain_memories(&bank_id, None, &retain_request)
            .await
            .expect("Failed to retain memories");
        assert!(retain_response.into_inner().success);

        // 3. Recall memories
        let recall_request = types::RecallRequest {
            query: "Who is Alice?".to_string(),
            max_tokens: 4096,
            trace: false,
            prefer_observations: false,
            budget: None,
            include: None,
            query_timestamp: None,
            types: None,
            tags: None,
            tags_match: types::TagsMatch::Any,
            tag_groups: None,
            min_scores: None,
        };
        let recall_response = client
            .recall_memories(&bank_id, None, &recall_request)
            .await
            .expect("Failed to recall memories");
        let recall_result = recall_response.into_inner();
        assert!(
            !recall_result.results.is_empty(),
            "Should recall at least one memory"
        );

        // Cleanup: delete the test bank's memories
        let _ = client.clear_bank_memories(&bank_id, None, None).await;
    }
}
