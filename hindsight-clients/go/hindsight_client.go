package hindsight

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"path"
	"runtime/debug"
	"strconv"
	"strings"
	"time"
)

func (c *APIClient) applyMaintainedRequestHeaders(request *http.Request) {
	for key, value := range c.cfg.DefaultHeader {
		request.Header.Add(key, value)
	}
	if request.Header.Get("User-Agent") == "" && c.cfg.UserAgent != "" {
		request.Header.Set("User-Agent", c.cfg.UserAgent)
	}
}

// ExportDocumentsTo streams a Document Transfer v2 archive to output.
func (c *APIClient) ExportDocumentsTo(ctx context.Context, bankID string, output io.Writer) error {
	baseURL, err := c.cfg.ServerURLWithContext(ctx, "DocumentTransferAPIService.ExportDocuments")
	if err != nil {
		return err
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/v1/default/banks/" + url.PathEscape(bankID) + "/document-transfer"
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	c.applyMaintainedRequestHeaders(request)
	response, err := c.callAPI(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		return fmt.Errorf("export documents returned %s: %s", response.Status, body)
	}
	_, err = io.Copy(output, response.Body)
	return err
}

// ImportDocumentsFrom streams source through a multipart upload without buffering the archive.
func (c *APIClient) ImportDocumentsFrom(
	ctx context.Context,
	bankID string,
	filename string,
	source io.Reader,
	onConflict string,
) (*DocumentImportSubmitResponse, error) {
	baseURL, err := c.cfg.ServerURLWithContext(ctx, "DocumentTransferAPIService.ImportDocuments")
	if err != nil {
		return nil, err
	}
	values := url.Values{"on_conflict": []string{onConflict}}
	endpoint := strings.TrimRight(baseURL, "/") + "/v1/default/banks/" + url.PathEscape(bankID) +
		"/document-transfer?" + values.Encode()
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, reader)
	if err != nil {
		_ = reader.Close()
		_ = writer.Close()
		return nil, err
	}
	request.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	c.applyMaintainedRequestHeaders(request)
	copyDone := make(chan error, 1)
	go func() {
		part, createErr := multipartWriter.CreateFormFile("file", path.Base(filename))
		if createErr == nil {
			_, createErr = io.Copy(part, source)
		}
		if closeErr := multipartWriter.Close(); createErr == nil {
			createErr = closeErr
		}
		_ = writer.CloseWithError(createErr)
		copyDone <- createErr
	}()
	response, err := c.callAPI(request)
	if err != nil {
		_ = reader.CloseWithError(err)
		<-copyDone
		return nil, err
	}
	copyErr := <-copyDone
	if copyErr != nil {
		return nil, copyErr
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		return nil, fmt.Errorf("import documents returned %s: %s", response.Status, body)
	}
	var result DocumentImportSubmitResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

// defaultUserAgent returns the User-Agent string sent on every request unless
// the caller overrides cfg.UserAgent. The version is read from build info so
// it stays in sync with the module version automatically; falls back to
// "devel" when running from an unpinned local checkout.
func defaultUserAgent() string {
	version := "devel"
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range info.Deps {
			if dep.Path == "github.com/vectorize-io/hindsight/hindsight-clients/go" {
				version = dep.Version
				break
			}
		}
	}
	return "hindsight-client-go/" + version
}

// DefaultUserAgent is the User-Agent string sent on every request unless the
// caller overrides cfg.UserAgent (e.g. for integrations identifying themselves).
var DefaultUserAgent = defaultUserAgent()

// NewAPIClientWithToken creates a new API client configured with a base URL and API token.
// The token is sent as a Bearer token in the Authorization header for all requests.
// Note: this uses http.DefaultClient which has no timeout. Use NewAPIClientWithTimeout
// to set a request timeout.
//
// Example:
//
//	client := hindsight.NewAPIClientWithToken("https://api.example.com", "your-api-token")
//	resp, _, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
func NewAPIClientWithToken(baseURL, token string) *APIClient {
	cfg := NewConfiguration()
	cfg.UserAgent = DefaultUserAgent
	cfg.Servers = ServerConfigurations{
		{URL: baseURL},
	}
	cfg.AddDefaultHeader("Authorization", "Bearer "+token)
	return NewAPIClient(cfg)
}

// NewAPIClientWithTimeout creates a new API client configured with a base URL, API token,
// and a request timeout. Use 0 for no timeout.
//
// Example:
//
//	client := hindsight.NewAPIClientWithTimeout("https://api.example.com", "your-api-token", 30*time.Second)
//	resp, _, err := client.MemoryAPI.RetainMemories(ctx, bankID).RetainRequest(req).Execute()
func NewAPIClientWithTimeout(baseURL, token string, timeout time.Duration) *APIClient {
	cfg := NewConfiguration()
	cfg.UserAgent = DefaultUserAgent
	cfg.Servers = ServerConfigurations{
		{URL: baseURL},
	}
	cfg.AddDefaultHeader("Authorization", "Bearer "+token)
	cfg.HTTPClient = &http.Client{Timeout: timeout}
	return NewAPIClient(cfg)
}

// ImageAssetDownload contains descriptor metadata and image bytes from one request.
type ImageAssetDownload struct {
	Descriptor ImageAssetDescriptor
	Content    []byte
}

// GetManagedImageAsset downloads a managed image and derives its descriptor from response headers.
func (c *APIClient) GetManagedImageAsset(ctx context.Context, bankID, assetID string) (*ImageAssetDownload, error) {
	baseURL, err := c.cfg.ServerURLWithContext(ctx, "ImagesAPIService.GetImageAsset")
	if err != nil {
		return nil, err
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/v1/default/banks/" + url.PathEscape(bankID) +
		"/image-assets/" + url.PathEscape(assetID)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	c.applyMaintainedRequestHeaders(request)
	response, err := c.callAPI(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		return nil, fmt.Errorf("get image asset returned %s: %s", response.Status, body)
	}
	content, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	parseInt32 := func(name string) (int32, error) {
		value, parseErr := strconv.ParseInt(response.Header.Get(name), 10, 32)
		if parseErr != nil {
			return 0, fmt.Errorf("invalid %s image header: %w", name, parseErr)
		}
		return int32(value), nil
	}
	width, err := parseInt32("X-Hindsight-Image-Width")
	if err != nil {
		return nil, err
	}
	height, err := parseInt32("X-Hindsight-Image-Height")
	if err != nil {
		return nil, err
	}
	size, err := parseInt32("Content-Length")
	if err != nil {
		return nil, err
	}
	createdAt, err := time.Parse(time.RFC3339Nano, response.Header.Get("X-Hindsight-Asset-Created-At"))
	if err != nil {
		return nil, fmt.Errorf("invalid created-at image header: %w", err)
	}
	updatedAt, err := time.Parse(time.RFC3339Nano, response.Header.Get("X-Hindsight-Asset-Updated-At"))
	if err != nil {
		return nil, fmt.Errorf("invalid updated-at image header: %w", err)
	}
	descriptor := NewImageAssetDescriptor(
		assetID,
		strings.Split(response.Header.Get("Content-Type"), ";")[0],
		size,
		response.Header.Get("X-Hindsight-Image-SHA256"),
		width,
		height,
		"ready",
		createdAt,
		updatedAt,
	)
	descriptor.DocumentIds = []string{}
	return &ImageAssetDownload{Descriptor: *descriptor, Content: content}, nil
}
