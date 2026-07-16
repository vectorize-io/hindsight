package hindsight

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func testClient(server *httptest.Server) *APIClient {
	cfg := NewConfiguration()
	cfg.Servers = ServerConfigurations{{URL: server.URL}}
	return NewAPIClient(cfg)
}

func TestImageMultipartOrderHeadersAndManagement(t *testing.T) {
	var parts []string
	var requestJSON, idempotency string
	var imagePath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			idempotency = r.Header.Get("Idempotency-Key")
			reader, err := r.MultipartReader()
			if err != nil {
				t.Fatal(err)
			}
			for {
				part, err := reader.NextPart()
				if err == io.EOF {
					break
				}
				data, _ := io.ReadAll(part)
				if part.FormName() == "files" {
					parts = append(parts, string(data))
				} else if part.FormName() == "request" {
					requestJSON = string(data)
				}
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)
			_, _ = io.WriteString(w, `{"operation_id":"op","document_id":"doc","image_assets":[]}`)
		case http.MethodGet:
			if strings.HasSuffix(r.URL.Path, "/image-assets") {
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"items":[],"total":0,"limit":100,"offset":0}`)
				return
			}
			imagePath = r.URL.EscapedPath()
			w.Header().Set("Content-Type", "image/jpeg")
			w.Header().Set("X-Hindsight-Image-SHA256", strings.Repeat("a", 64))
			w.Header().Set("X-Hindsight-Image-Width", "8")
			w.Header().Set("X-Hindsight-Image-Height", "6")
			w.Header().Set("X-Hindsight-Asset-Created-At", "2026-01-01T00:00:00Z")
			w.Header().Set("X-Hindsight-Asset-Updated-At", "2026-01-01T00:00:00Z")
			_, _ = io.WriteString(w, "bytes")
		case http.MethodDelete:
			w.WriteHeader(http.StatusNoContent)
		}
	}))
	defer server.Close()

	first, _ := os.CreateTemp(t.TempDir(), "first-*.jpg")
	second, _ := os.CreateTemp(t.TempDir(), "second-*.png")
	_, _ = first.WriteString("first")
	_, _ = second.WriteString("second")
	_, _ = first.Seek(0, io.SeekStart)
	_, _ = second.Seek(0, io.SeekStart)
	request := `{"images":[{"asset_id":"a/1"},{"asset_id":"b/2"}]}`
	client := testClient(server)
	_, _, err := client.ImagesAPI.ImageRetain(context.Background(), "bank").
		Files([]*os.File{first, second}).Request(request).IdempotencyKey("idem").Execute()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(parts, ",") != "first,second" || requestJSON != request || idempotency != "idem" {
		t.Fatalf("multipart contract mismatch: parts=%v request=%s idempotency=%s", parts, requestJSON, idempotency)
	}
	listed, _, err := client.ImagesAPI.ListImageAssets(context.Background(), "bank").DocumentId("doc/one").Execute()
	if err != nil || listed.Total != 0 {
		t.Fatalf("list failed: listed=%#v err=%v", listed, err)
	}
	downloaded, err := client.GetManagedImageAsset(context.Background(), "bank", "folder/a.jpg")
	if err != nil {
		t.Fatal(err)
	}
	if string(downloaded.Content) != "bytes" || len(downloaded.Descriptor.DocumentIds) != 0 || !strings.Contains(imagePath, "folder%2Fa.jpg") {
		t.Fatalf("download contract mismatch: %#v", downloaded)
	}
	if _, err := client.ImagesAPI.DeleteImageAsset(context.Background(), "bank", "a/1").Execute(); err != nil {
		t.Fatal(err)
	}
	var recalled RecallResponse
	if err := json.Unmarshal([]byte(`{"results":[],"image_assets":{"doc":[]}}`), &recalled); err != nil {
		t.Fatal(err)
	}
	if recalled.ImageAssets["doc"] == nil {
		t.Fatal("recall image map was not decoded")
	}
}

func TestDocumentTransferHelpersStreamReadersAndWriters(t *testing.T) {
	archive := bytes.Repeat([]byte("archive"), 1024)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			_, _ = w.Write(archive)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		uploaded, _ := io.ReadAll(file)
		if !bytes.Equal(uploaded, archive) {
			t.Fatal("uploaded archive changed")
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(DocumentImportSubmitResponse{OperationId: "op"})
	}))
	defer server.Close()
	client := testClient(server)
	var output bytes.Buffer
	if err := client.ExportDocumentsTo(context.Background(), "bank", &output); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(output.Bytes(), archive) {
		t.Fatal("downloaded archive changed")
	}
	result, err := client.ImportDocumentsFrom(context.Background(), "bank", "transfer.zip", bytes.NewReader(archive), "skip")
	if err != nil || result.OperationId != "op" {
		t.Fatalf("import failed: result=%#v err=%v", result, err)
	}
}

type failOnRead struct {
	read bool
}

func (reader *failOnRead) Read(_ []byte) (int, error) {
	reader.read = true
	return 0, io.EOF
}

func TestImportDocumentsDoesNotStartSourceReadWhenRequestIsInvalid(t *testing.T) {
	cfg := NewConfiguration()
	cfg.Servers = ServerConfigurations{{URL: ":"}}
	client := NewAPIClient(cfg)
	source := &failOnRead{}

	if _, err := client.ImportDocumentsFrom(
		context.Background(), "bank", "archive.zip", source, "skip",
	); err == nil {
		t.Fatal("expected invalid request URL to fail")
	}
	if source.read {
		t.Fatal("source was read before the request was constructed")
	}
}
