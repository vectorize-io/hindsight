package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	hindsightv1alpha1 "github.com/vectorize-io/hindsight/hindsight-operator/api/v1alpha1"
)

// fakeHindsight mimics the bank template endpoints closely enough to test
// reconciliation: import creates missing banks, merges settings, upserts
// mental models and directives, and counts every mental model it regenerates.
type fakeHindsight struct {
	mu           sync.Mutex
	banks        map[string]map[string]any
	imports      []map[string]any
	regenerated  map[string]int
	deleted      []string
	apiKey       string
	failImportOn string
	// failExportStatus, when set, makes export return that status once.
	failExportStatus int
	// fixedExport, when set, is returned by export instead of the stored bank,
	// to simulate a server that stores a template differently.
	fixedExport map[string]any
}

func newFakeHindsight(apiKey string) *fakeHindsight {
	return &fakeHindsight{banks: map[string]map[string]any{}, regenerated: map[string]int{}, apiKey: apiKey}
}

func (f *fakeHindsight) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.apiKey != "" && r.Header.Get("Authorization") != "Bearer "+f.apiKey {
		http.Error(w, `{"detail":"unauthorized"}`, http.StatusUnauthorized)
		return
	}
	if r.URL.Path == "/v1/bank-template-schema" {
		_, _ = io.WriteString(w, `{"$defs":{
			"BankTemplateMentalModel":{"properties":{"tags":{"default":[]},"max_tokens":{"default":2048},"trigger":{"default":{}}}},
			"MentalModelTrigger":{"properties":{"mode":{"default":"full"},"refresh_after_consolidation":{"default":false}}},
			"BankTemplateDirective":{"properties":{"priority":{"default":0},"is_active":{"default":true},"tags":{"default":[]}}}}}`)
		return
	}
	rest, ok := strings.CutPrefix(r.URL.Path, "/v1/default/banks/")
	if !ok {
		http.NotFound(w, r)
		return
	}
	bankID, action, _ := strings.Cut(rest, "/")
	switch {
	case r.Method == http.MethodGet && action == "export" && f.failExportStatus != 0:
		status := f.failExportStatus
		f.failExportStatus = 0
		http.Error(w, `{"detail":"unavailable"}`, status)
	case r.Method == http.MethodGet && action == "export" && f.fixedExport != nil && f.banks[bankID] != nil:
		_ = json.NewEncoder(w).Encode(f.fixedExport)
	case r.Method == http.MethodGet && action == "export":
		bank, ok := f.banks[bankID]
		if !ok {
			http.Error(w, `{"detail":"not found"}`, http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(f.export(bank))
	case r.Method == http.MethodPost && action == "import":
		var manifest map[string]any
		_ = json.NewDecoder(r.Body).Decode(&manifest)
		if f.failImportOn != "" && strings.Contains(fmt.Sprint(manifest), f.failImportOn) {
			http.Error(w, `{"detail":"invalid manifest"}`, http.StatusBadRequest)
			return
		}
		f.imports = append(f.imports, manifest)
		f.apply(bankID, manifest)
		_, _ = io.WriteString(w, `{"config_applied":true}`)
	case r.Method == http.MethodDelete && action == "":
		delete(f.banks, bankID)
		f.deleted = append(f.deleted, bankID)
		_, _ = io.WriteString(w, `{"success":true}`)
	default:
		http.NotFound(w, r)
	}
}

func (f *fakeHindsight) apply(bankID string, manifest map[string]any) {
	bank, ok := f.banks[bankID]
	if !ok {
		bank = map[string]any{"bank": map[string]any{}, "mental_models": map[string]any{}, "directives": map[string]any{}}
		f.banks[bankID] = bank
	}
	if settings, ok := manifest["bank"].(map[string]any); ok {
		for k, v := range settings {
			if v != nil {
				bank["bank"].(map[string]any)[k] = v
			}
		}
	}
	for _, m := range asList(manifest["mental_models"]) {
		id := m["id"].(string)
		bank["mental_models"].(map[string]any)[id] = m
		f.regenerated[bankID+"/"+id]++
	}
	for _, d := range asList(manifest["directives"]) {
		bank["directives"].(map[string]any)[d["name"].(string)] = d
	}
}

// export returns the stored bank with every default filled in, like the real API.
func (f *fakeHindsight) export(bank map[string]any) map[string]any {
	var models, directives []any
	for _, raw := range bank["mental_models"].(map[string]any) {
		m := copyMap(raw.(map[string]any))
		setDefault(m, "tags", []any{})
		setDefault(m, "max_tokens", 2048)
		trigger, _ := m["trigger"].(map[string]any)
		trigger = copyMap(trigger)
		setDefault(trigger, "mode", "full")
		setDefault(trigger, "refresh_after_consolidation", false)
		m["trigger"] = trigger
		models = append(models, m)
	}
	for _, raw := range bank["directives"].(map[string]any) {
		d := copyMap(raw.(map[string]any))
		setDefault(d, "priority", 0)
		setDefault(d, "is_active", true)
		setDefault(d, "tags", []any{})
		directives = append(directives, d)
	}
	return map[string]any{"version": "1", "bank": bank["bank"], "mental_models": models, "directives": directives}
}

func (f *fakeHindsight) setSetting(bankID, key string, value any) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.banks[bankID]["bank"].(map[string]any)[key] = value
}

func (f *fakeHindsight) seed(bankID string, manifest map[string]any) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.apply(bankID, manifest)
}

func (f *fakeHindsight) snapshot() (imports []map[string]any, regenerated map[string]int, deleted []string, banks map[string]bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	regenerated = map[string]int{}
	for k, v := range f.regenerated {
		regenerated[k] = v
	}
	banks = map[string]bool{}
	for id := range f.banks {
		banks[id] = true
	}
	return append([]map[string]any(nil), f.imports...), regenerated, append([]string(nil), f.deleted...), banks
}

func asList(v any) []map[string]any {
	list, _ := v.([]any)
	out := make([]map[string]any, 0, len(list))
	for _, item := range list {
		out = append(out, item.(map[string]any))
	}
	return out
}

func copyMap(m map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range m {
		out[k] = v
	}
	return out
}

func setDefault(m map[string]any, key string, value any) {
	if _, ok := m[key]; !ok {
		m[key] = value
	}
}

type harness struct {
	t      *testing.T
	ctx    context.Context
	client client.Client
	api    *fakeHindsight
	url    string
	ns     string
}

func setup(t *testing.T) *harness {
	t.Helper()
	ctrl.SetLogger(zap.New(zap.WriteTo(io.Discard)))
	assets := os.Getenv("KUBEBUILDER_ASSETS")
	if assets == "" {
		t.Skip("KUBEBUILDER_ASSETS is not set; run `make test` to download the test API server")
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
		Scheme:  scheme,
		Metrics: metricsserver.Options{BindAddress: "0"},
		// Each test starts its own manager in the same process.
		Controller: config.Controller{SkipNameValidation: ptr.To(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	api := newFakeHindsight("secret-token")
	server := httptest.NewServer(api)
	t.Cleanup(server.Close)

	if err := (&HindsightBankReconciler{
		Client:       mgr.GetClient(),
		APIReader:    mgr.GetAPIReader(),
		Recorder:     events.NewFakeRecorder(100),
		HTTPClient:   server.Client(),
		ResyncPeriod: 300 * time.Millisecond,
	}).SetupWithManager(mgr); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go func() { _ = mgr.Start(ctx) }()

	h := &harness{t: t, ctx: ctx, client: mgr.GetClient(), api: api, url: server.URL, ns: "default"}
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "hindsight-api", Namespace: h.ns},
		Data:       map[string][]byte{"token": []byte("secret-token\n")},
	}
	if err := h.client.Create(ctx, secret); err != nil {
		t.Fatal(err)
	}
	return h
}

func (h *harness) bank(name, bankID string, policy hindsightv1alpha1.DeletionPolicy, template string) *hindsightv1alpha1.HindsightBank {
	return &hindsightv1alpha1.HindsightBank{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: h.ns},
		Spec: hindsightv1alpha1.HindsightBankSpec{
			BankID: bankID,
			Connection: hindsightv1alpha1.Connection{
				URL:             h.url,
				APIKeySecretRef: &hindsightv1alpha1.SecretKeyRef{Name: "hindsight-api", Key: "token"},
			},
			Template:       apiextensionsv1.JSON{Raw: []byte(template)},
			DeletionPolicy: policy,
		},
	}
}

func (h *harness) waitFor(what string, cond func() bool) {
	h.t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	h.t.Fatalf("timed out waiting for %s", what)
}

func (h *harness) ready(name string, generation int64) *metav1.Condition {
	var bank hindsightv1alpha1.HindsightBank
	if err := h.client.Get(h.ctx, types.NamespacedName{Namespace: h.ns, Name: name}, &bank); err != nil {
		return nil
	}
	c := meta.FindStatusCondition(bank.Status.Conditions, ConditionReady)
	if c == nil || c.ObservedGeneration < generation {
		return nil
	}
	return c
}

const template = `{"version":"1","bank":{"retain_mission":"Keep decisions."},
	"mental_models":[{"id":"prefs","name":"Preferences","source_query":"What does the user prefer?"},
		{"id":"conventions","name":"Conventions","source_query":"Which conventions apply?"}],
	"directives":[{"name":"Be concise","content":"Answer briefly."}]}`

func TestCreatesBankAndConverges(t *testing.T) {
	h := setup(t)
	bank := h.bank("team", "team-memory", "", template)
	if err := h.client.Create(h.ctx, bank); err != nil {
		t.Fatal(err)
	}
	h.waitFor("Ready=True", func() bool {
		c := h.ready("team", 1)
		return c != nil && c.Status == metav1.ConditionTrue
	})

	// Several resyncs pass; an in-sync bank must not be imported again,
	// because every import regenerates its mental models.
	time.Sleep(1500 * time.Millisecond)
	imports, regenerated, _, banks := h.api.snapshot()
	if !banks["team-memory"] {
		t.Fatal("bank was not created")
	}
	if len(imports) != 1 {
		t.Fatalf("expected exactly one import, got %d", len(imports))
	}
	if regenerated["team-memory/prefs"] != 1 {
		t.Fatalf("prefs regenerated %d times, want 1", regenerated["team-memory/prefs"])
	}

	var got hindsightv1alpha1.HindsightBank
	_ = h.client.Get(h.ctx, types.NamespacedName{Namespace: h.ns, Name: "team"}, &got)
	if !got.Status.BankCreated {
		t.Fatal("status.bankCreated should be true for a bank the operator created")
	}
}

func TestSpecChangeImportsOnlyTheChangedModel(t *testing.T) {
	h := setup(t)
	bank := h.bank("team", "", "", template)
	if err := h.client.Create(h.ctx, bank); err != nil {
		t.Fatal(err)
	}
	h.waitFor("first sync", func() bool { c := h.ready("team", 1); return c != nil && c.Status == metav1.ConditionTrue })

	var current hindsightv1alpha1.HindsightBank
	_ = h.client.Get(h.ctx, types.NamespacedName{Namespace: h.ns, Name: "team"}, &current)
	current.Spec.Template.Raw = []byte(strings.Replace(template, "Which conventions apply?", "Which conventions are explicit?", 1))
	if err := h.client.Update(h.ctx, &current); err != nil {
		t.Fatal(err)
	}
	h.waitFor("second sync", func() bool {
		c := h.ready("team", current.Generation)
		return c != nil && c.Status == metav1.ConditionTrue
	})

	imports, regenerated, _, _ := h.api.snapshot()
	if len(imports) != 2 {
		t.Fatalf("expected 2 imports, got %d", len(imports))
	}
	second := imports[1]
	if _, ok := second["bank"]; ok {
		t.Fatalf("unchanged bank settings were re-sent: %v", second["bank"])
	}
	if _, ok := second["directives"]; ok {
		t.Fatalf("unchanged directives were re-sent: %v", second["directives"])
	}
	if regenerated["team/prefs"] != 1 || regenerated["team/conventions"] != 2 {
		t.Fatalf("regenerations = %v; only conventions should regenerate", regenerated)
	}
}

func TestCorrectsDriftMadeOutsideKubernetes(t *testing.T) {
	h := setup(t)
	if err := h.client.Create(h.ctx, h.bank("team", "", "", template)); err != nil {
		t.Fatal(err)
	}
	h.waitFor("first sync", func() bool { c := h.ready("team", 1); return c != nil && c.Status == metav1.ConditionTrue })

	h.api.setSetting("team", "retain_mission", "edited in the UI")
	h.waitFor("drift corrected", func() bool {
		imports, _, _, _ := h.api.snapshot()
		return len(imports) == 2
	})
	imports, _, _, _ := h.api.snapshot()
	if got := imports[1]["bank"]; fmt.Sprint(got) != "map[retain_mission:Keep decisions.]" {
		t.Fatalf("drift import = %v", got)
	}
	if _, ok := imports[1]["mental_models"]; ok {
		t.Fatal("drift correction must not regenerate unchanged mental models")
	}
}

func TestDeletePolicyNeverDeletesAdoptedBank(t *testing.T) {
	h := setup(t)
	h.api.seed("existing", map[string]any{"bank": map[string]any{"retain_mission": "old"}})

	adopted := h.bank("adopted", "existing", hindsightv1alpha1.DeletionPolicyDelete, template)
	created := h.bank("created", "fresh", hindsightv1alpha1.DeletionPolicyDelete, template)
	for _, b := range []*hindsightv1alpha1.HindsightBank{adopted, created} {
		if err := h.client.Create(h.ctx, b); err != nil {
			t.Fatal(err)
		}
	}
	h.waitFor("both synced", func() bool {
		a, c := h.ready("adopted", 1), h.ready("created", 1)
		return a != nil && c != nil && a.Status == metav1.ConditionTrue && c.Status == metav1.ConditionTrue
	})
	for _, b := range []*hindsightv1alpha1.HindsightBank{adopted, created} {
		if err := h.client.Delete(h.ctx, b); err != nil {
			t.Fatal(err)
		}
	}
	h.waitFor("resources removed", func() bool {
		var list hindsightv1alpha1.HindsightBankList
		_ = h.client.List(h.ctx, &list, client.InNamespace(h.ns))
		return len(list.Items) == 0
	})
	_, _, deleted, banks := h.api.snapshot()
	if fmt.Sprint(deleted) != "[fresh]" {
		t.Fatalf("deleted banks = %v, want only [fresh]", deleted)
	}
	if !banks["existing"] {
		t.Fatal("adopted bank was deleted")
	}
}

func TestRetainPolicyKeepsBank(t *testing.T) {
	h := setup(t)
	b := h.bank("team", "", "", template)
	if err := h.client.Create(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	h.waitFor("sync", func() bool { c := h.ready("team", 1); return c != nil && c.Status == metav1.ConditionTrue })
	if err := h.client.Delete(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	h.waitFor("resource removed", func() bool {
		var got hindsightv1alpha1.HindsightBank
		return apierrors.IsNotFound(h.client.Get(h.ctx, client.ObjectKeyFromObject(b), &got))
	})
	_, _, deleted, banks := h.api.snapshot()
	if len(deleted) != 0 || !banks["team"] {
		t.Fatalf("Retain must keep the bank; deleted=%v", deleted)
	}
}

func TestRejectedTemplateReportsInvalidTemplate(t *testing.T) {
	h := setup(t)
	h.api.failImportOn = "bad-model"
	b := h.bank("team", "", "", `{"version":"1","mental_models":[{"id":"bad-model","name":"x","source_query":"q"}]}`)
	if err := h.client.Create(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	h.waitFor("Ready=False InvalidTemplate", func() bool {
		c := h.ready("team", 1)
		return c != nil && c.Status == metav1.ConditionFalse && c.Reason == ReasonInvalidTemplate
	})
}

func TestMissingSecretReportsSecretError(t *testing.T) {
	h := setup(t)
	b := h.bank("team", "", "", template)
	b.Spec.Connection.APIKeySecretRef.Name = "missing"
	if err := h.client.Create(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	h.waitFor("Ready=False SecretError", func() bool {
		c := h.ready("team", 1)
		return c != nil && c.Status == metav1.ConditionFalse && c.Reason == ReasonSecretError
	})
	imports, _, _, _ := h.api.snapshot()
	if len(imports) != 0 {
		t.Fatal("nothing should be imported without credentials")
	}
}

func TestBankIDIsImmutable(t *testing.T) {
	h := setup(t)
	b := h.bank("team", "one", "", template)
	if err := h.client.Create(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	b.Spec.BankID = "two"
	if err := h.client.Update(h.ctx, b); err == nil || !strings.Contains(err.Error(), "bankId is immutable") {
		t.Fatalf("expected immutability error, got %v", err)
	}
}

func (f *fakeHindsight) set(fn func(*fakeHindsight)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	fn(f)
}

// Pointing a resource at another server must never let Delete remove a bank
// there, even though this resource created a bank of the same ID elsewhere.
func TestDeleteSkipsBankOnAnotherServer(t *testing.T) {
	h := setup(t)
	b := h.bank("team", "", hindsightv1alpha1.DeletionPolicyDelete, template)
	if err := h.client.Create(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	h.waitFor("sync", func() bool { c := h.ready("team", 1); return c != nil && c.Status == metav1.ConditionTrue })

	other := newFakeHindsight("secret-token")
	other.seed("team", map[string]any{"bank": map[string]any{"retain_mission": "someone else's"}})
	otherServer := httptest.NewServer(other)
	defer otherServer.Close()

	var current hindsightv1alpha1.HindsightBank
	_ = h.client.Get(h.ctx, client.ObjectKeyFromObject(b), &current)
	current.Spec.Connection.URL = otherServer.URL
	if err := h.client.Update(h.ctx, &current); err != nil {
		t.Fatal(err)
	}
	h.waitFor("sync on other server", func() bool {
		c := h.ready("team", current.Generation)
		return c != nil && c.Status == metav1.ConditionTrue
	})
	if err := h.client.Delete(h.ctx, &current); err != nil {
		t.Fatal(err)
	}
	h.waitFor("resource removed", func() bool {
		return apierrors.IsNotFound(h.client.Get(h.ctx, client.ObjectKeyFromObject(b), &current))
	})
	_, _, deleted, banks := other.snapshot()
	if len(deleted) != 0 || !banks["team"] {
		t.Fatalf("bank on the other server was deleted: %v", deleted)
	}
}

// When the API key Secret is gone, deletion is blocked until the user opts
// out with the skip annotation, and then the bank is kept.
func TestSkipAnnotationUnblocksDeletion(t *testing.T) {
	h := setup(t)
	b := h.bank("team", "", hindsightv1alpha1.DeletionPolicyDelete, template)
	if err := h.client.Create(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	h.waitFor("sync", func() bool { c := h.ready("team", 1); return c != nil && c.Status == metav1.ConditionTrue })
	if err := h.client.Delete(h.ctx, &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "hindsight-api", Namespace: h.ns}}); err != nil {
		t.Fatal(err)
	}
	if err := h.client.Delete(h.ctx, b); err != nil {
		t.Fatal(err)
	}
	time.Sleep(time.Second)
	var current hindsightv1alpha1.HindsightBank
	if err := h.client.Get(h.ctx, client.ObjectKeyFromObject(b), &current); err != nil {
		t.Fatalf("resource should be held by its finalizer: %v", err)
	}
	current.Annotations = map[string]string{AnnotationSkipBankDeletion: "true"}
	if err := h.client.Update(h.ctx, &current); err != nil {
		t.Fatal(err)
	}
	h.waitFor("resource removed", func() bool {
		return apierrors.IsNotFound(h.client.Get(h.ctx, client.ObjectKeyFromObject(b), &current))
	})
	_, _, deleted, banks := h.api.snapshot()
	if len(deleted) != 0 || !banks["team"] {
		t.Fatalf("skip annotation must keep the bank; deleted=%v", deleted)
	}
}

// A template the server stores differently must not be imported on every
// resync, including after a transient API error.
func TestNotConvergedStopsReimportAcrossTransientErrors(t *testing.T) {
	h := setup(t)
	h.api.set(func(f *fakeHindsight) {
		f.fixedExport = map[string]any{"version": "1", "bank": map[string]any{"retain_mission": "stored differently"}}
	})
	if err := h.client.Create(h.ctx, h.bank("team", "", "", template)); err != nil {
		t.Fatal(err)
	}
	h.waitFor("NotConverged", func() bool {
		c := h.ready("team", 1)
		return c != nil && c.Reason == ReasonNotConverged
	})
	h.api.set(func(f *fakeHindsight) { f.failExportStatus = http.StatusServiceUnavailable })
	h.waitFor("APIError", func() bool {
		c := h.ready("team", 1)
		return c != nil && c.Reason == ReasonAPIError
	})
	time.Sleep(1500 * time.Millisecond)
	imports, _, _, _ := h.api.snapshot()
	if len(imports) != 1 {
		t.Fatalf("expected one import, got %d", len(imports))
	}
}
