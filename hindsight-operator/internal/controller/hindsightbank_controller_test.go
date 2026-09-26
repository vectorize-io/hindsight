package controller

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	hindsightv1alpha1 "github.com/vectorize-io/hindsight/hindsight-operator/api/v1alpha1"
)

// importRecorder stands in for the Hindsight import endpoint.
type importRecorder struct {
	mu      sync.Mutex
	status  int
	imports []string
	auth    []string
}

func (f *importRecorder) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	body, _ := io.ReadAll(r.Body)
	f.imports = append(f.imports, r.URL.Path+" "+string(body))
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	if f.status != 0 {
		http.Error(w, `{"detail":"rejected"}`, f.status)
		return
	}
	_, _ = io.WriteString(w, `{}`)
}

func (f *importRecorder) calls() ([]string, []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.imports...), append([]string(nil), f.auth...)
}

func start(t *testing.T, api *importRecorder) (context.Context, client.Client, string) {
	t.Helper()
	ctrl.SetLogger(zap.New(zap.WriteTo(io.Discard)))
	assets := os.Getenv("KUBEBUILDER_ASSETS")
	if assets == "" {
		t.Skip("KUBEBUILDER_ASSETS is not set; run `make test`")
	}
	env := &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
		BinaryAssetsDirectory: assets,
	}
	cfg, err := env.Start()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = env.Stop() })

	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = hindsightv1alpha1.AddToScheme(scheme)
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme:     scheme,
		Metrics:    metricsserver.Options{BindAddress: "0"},
		Controller: config.Controller{SkipNameValidation: ptr.To(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(api)
	t.Cleanup(server.Close)
	if err := (&HindsightBankReconciler{
		Client:       mgr.GetClient(),
		APIReader:    mgr.GetAPIReader(),
		HTTPClient:   server.Client(),
		ResyncPeriod: 200 * time.Millisecond,
	}).SetupWithManager(mgr); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = mgr.Start(ctx) }()

	c := mgr.GetClient()
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "hindsight-api", Namespace: "default"},
		Data:       map[string][]byte{"token": []byte("secret-token\n")},
	}
	if err := c.Create(ctx, secret); err != nil {
		t.Fatal(err)
	}
	return ctx, c, server.URL
}

func newBank(url, template string) *hindsightv1alpha1.HindsightBank {
	return &hindsightv1alpha1.HindsightBank{
		ObjectMeta: metav1.ObjectMeta{Name: "team", Namespace: "default"},
		Spec: hindsightv1alpha1.HindsightBankSpec{
			Connection: hindsightv1alpha1.Connection{
				URL:             url,
				APIKeySecretRef: &hindsightv1alpha1.SecretKeyRef{Name: "hindsight-api", Key: "token"},
			},
			Template: apiextensionsv1.JSON{Raw: []byte(template)},
		},
	}
}

func waitReady(t *testing.T, ctx context.Context, c client.Client, reason string) *metav1.Condition {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		var got hindsightv1alpha1.HindsightBank
		if c.Get(ctx, client.ObjectKey{Namespace: "default", Name: "team"}, &got) == nil {
			if cond := meta.FindStatusCondition(got.Status.Conditions, ConditionReady); cond != nil && cond.Reason == reason {
				return cond
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for Ready reason %s", reason)
	return nil
}

const template = `{"version":"1","bank":{"retain_mission":"Keep decisions."}}`

// The template is imported as-is into the bank named by the resource, with the
// API key from the Secret, and again on every resync.
func TestImportsTemplateAndResyncs(t *testing.T) {
	api := &importRecorder{}
	ctx, c, url := start(t, api)
	if err := c.Create(ctx, newBank(url, template)); err != nil {
		t.Fatal(err)
	}
	waitReady(t, ctx, c, "Imported")
	time.Sleep(time.Second)

	imports, auth := api.calls()
	if len(imports) < 3 {
		t.Fatalf("expected resync imports, got %d", len(imports))
	}
	path, body, _ := strings.Cut(imports[0], " ")
	var got, want any
	_ = json.Unmarshal([]byte(body), &got)
	_ = json.Unmarshal([]byte(template), &want)
	if path != "/v1/default/banks/team/import" || !reflect.DeepEqual(got, want) {
		t.Fatalf("import = %s %s", path, body)
	}
	if auth[0] != "Bearer secret-token" {
		t.Fatalf("Authorization = %q", auth[0])
	}
}

// A rejected template reports the API's message and is not retried with
// backoff; it is retried only on the normal resync.
func TestRejectedTemplateReportsAPIMessage(t *testing.T) {
	api := &importRecorder{status: http.StatusBadRequest}
	ctx, c, url := start(t, api)
	if err := c.Create(ctx, newBank(url, template)); err != nil {
		t.Fatal(err)
	}
	cond := waitReady(t, ctx, c, "ImportFailed")
	if cond.Status != metav1.ConditionFalse || !strings.Contains(cond.Message, "rejected") {
		t.Fatalf("condition = %+v", cond)
	}
}

func TestMissingSecretReportsFailureWithoutCallingAPI(t *testing.T) {
	api := &importRecorder{}
	ctx, c, url := start(t, api)
	b := newBank(url, template)
	b.Spec.Connection.APIKeySecretRef.Name = "missing"
	if err := c.Create(ctx, b); err != nil {
		t.Fatal(err)
	}
	cond := waitReady(t, ctx, c, "ImportFailed")
	if !strings.Contains(cond.Message, "missing") {
		t.Fatalf("condition = %+v", cond)
	}
	if imports, _ := api.calls(); len(imports) != 0 {
		t.Fatalf("API was called without credentials: %v", imports)
	}
}

func TestBankIDIsImmutable(t *testing.T) {
	ctx, c, url := start(t, &importRecorder{})
	b := newBank(url, template)
	b.Spec.BankID = "one"
	if err := c.Create(ctx, b); err != nil {
		t.Fatal(err)
	}
	b.Spec.BankID = "two"
	if err := c.Update(ctx, b); err == nil || !strings.Contains(err.Error(), "bankId is immutable") {
		t.Fatalf("expected immutability error, got %v", err)
	}
}
