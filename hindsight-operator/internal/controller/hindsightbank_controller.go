// Package controller reconciles HindsightBank resources.
package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	hindsightv1alpha1 "github.com/vectorize-io/hindsight/hindsight-operator/api/v1alpha1"
	"github.com/vectorize-io/hindsight/hindsight-operator/internal/hindsight"
)

const (
	finalizer = "hindsight.vectorize.io/bank"

	// AnnotationSkipBankDeletion, set to "true", lets a resource with
	// DeletionPolicy Delete finish deleting while keeping the bank, for
	// example when the API or its Secret is already gone.
	AnnotationSkipBankDeletion = "hindsight.vectorize.io/skip-bank-deletion"

	// ConditionReady is True when the bank matches the template.
	ConditionReady = "Ready"

	ReasonSynced          = "Synced"
	ReasonInvalidTemplate = "InvalidTemplate"
	ReasonSecretError     = "SecretError"
	ReasonAPIError        = "APIError"
	// ReasonNotConverged means the bank still differed from the template right
	// after an import. The operator stops importing until the spec changes,
	// because every import regenerates the imported mental models.
	ReasonNotConverged = "NotConverged"
)

// HindsightBankReconciler makes Hindsight banks match HindsightBank resources.
type HindsightBankReconciler struct {
	client.Client
	// APIReader reads Secrets directly, so the operator needs only "get" on
	// Secrets and does not cache every Secret in the cluster.
	APIReader  client.Reader
	Recorder   events.EventRecorder
	HTTPClient *http.Client
	// ResyncPeriod is how often a synced bank is compared again, which
	// corrects changes made outside Kubernetes.
	ResyncPeriod time.Duration
}

// +kubebuilder:rbac:groups=hindsight.vectorize.io,resources=hindsightbanks,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=hindsight.vectorize.io,resources=hindsightbanks/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=hindsight.vectorize.io,resources=hindsightbanks/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get
// +kubebuilder:rbac:groups=events.k8s.io,resources=events,verbs=create;patch

// Reconcile applies one HindsightBank.
func (r *HindsightBankReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var bank hindsightv1alpha1.HindsightBank
	if err := r.Get(ctx, req.NamespacedName, &bank); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !bank.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, r.finalize(ctx, &bank)
	}
	if controllerutil.AddFinalizer(&bank, finalizer) {
		if err := r.Update(ctx, &bank); err != nil {
			return ctrl.Result{}, err
		}
	}

	api, err := r.apiClient(ctx, &bank)
	if err != nil {
		return r.fail(ctx, &bank, ReasonSecretError, err, false)
	}
	return r.sync(ctx, &bank, api)
}

func (r *HindsightBankReconciler) sync(ctx context.Context, bank *hindsightv1alpha1.HindsightBank, api *hindsight.Client) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	bankID := bank.EffectiveBankID()

	var desired map[string]any
	if err := json.Unmarshal(bank.Spec.Template.Raw, &desired); err != nil || desired == nil {
		return r.fail(ctx, bank, ReasonInvalidTemplate, fmt.Errorf("spec.template must be a JSON object"), false)
	}

	defaults, err := api.Defaults(ctx)
	if err != nil {
		return r.failAPI(ctx, bank, err)
	}
	current, err := api.Export(ctx, bankID)
	exists := true
	if errors.Is(err, hindsight.ErrNotFound) {
		exists, current = false, nil
	} else if err != nil {
		return r.failAPI(ctx, bank, err)
	}

	plan, err := hindsight.ComputePlan(desired, current, defaults)
	if err != nil {
		return r.fail(ctx, bank, ReasonInvalidTemplate, err, false)
	}
	if exists && plan.Empty() {
		return r.ready(ctx, bank, "bank matches the template")
	}

	if bank.Status.NotConvergedGeneration == bank.Generation {
		// Importing again would regenerate the same mental models on every
		// resync without converging. Wait for a spec change. This is a status
		// field rather than the Ready reason so that a transient API error in
		// between does not re-arm the import.
		return ctrl.Result{RequeueAfter: r.ResyncPeriod}, nil
	}

	manifest := plan.Manifest
	if !exists {
		// Import creates a missing bank; send the full template so an empty
		// template still creates it.
		manifest = desired
		// Record the creation before importing. Import can create the bank
		// and still fail on a later item; without this marker the next
		// reconcile would treat the bank as adopted and Delete would leave
		// it behind.
		bank.Status.BankCreated = true
		bank.Status.CreatedOnURL = bank.Spec.Connection.URL
		if err := r.Status().Update(ctx, bank); err != nil {
			return ctrl.Result{}, err
		}
	}
	if _, err := api.Import(ctx, bankID, manifest, false); err != nil {
		return r.failAPI(ctx, bank, err)
	}
	logger.Info("imported bank template", "bank", bankID, "changes", plan.Changes, "created", !exists)
	r.Recorder.Eventf(bank, nil, corev1.EventTypeNormal, "Applied", "Apply", "imported %s into bank %q", describe(plan.Changes, !exists), bankID)

	now := metav1.Now()
	bank.Status.LastAppliedTime = &now
	bank.Status.LastAppliedChanges = plan.Changes

	// Confirm convergence immediately. A template that the server normalizes
	// differently from what the operator compares would otherwise be
	// re-imported on every resync.
	after, err := api.Export(ctx, bankID)
	if err != nil {
		return r.failAPI(ctx, bank, err)
	}
	remaining, err := hindsight.ComputePlan(desired, after, defaults)
	if err != nil {
		return r.fail(ctx, bank, ReasonInvalidTemplate, err, false)
	}
	if !remaining.Empty() {
		bank.Status.NotConvergedGeneration = bank.Generation
		return r.fail(ctx, bank, ReasonNotConverged,
			fmt.Errorf("bank still differs from the template after import: %s", strings.Join(remaining.Changes, ", ")), false)
	}
	bank.Status.NotConvergedGeneration = 0
	return r.ready(ctx, bank, "bank matches the template")
}

// finalize handles deletion. Delete removes the bank only when this resource
// created it on the server it still points at; adopted banks are always kept.
func (r *HindsightBankReconciler) finalize(ctx context.Context, bank *hindsightv1alpha1.HindsightBank) error {
	if !controllerutil.ContainsFinalizer(bank, finalizer) {
		return nil
	}
	owned := bank.Status.BankCreated && bank.Status.CreatedOnURL == bank.Spec.Connection.URL
	if bank.Spec.DeletionPolicy == hindsightv1alpha1.DeletionPolicyDelete && owned {
		if bank.Annotations[AnnotationSkipBankDeletion] == "true" {
			r.Recorder.Eventf(bank, nil, corev1.EventTypeWarning, "BankRetained", "Delete",
				"kept bank %q because the %s annotation is set", bank.EffectiveBankID(), AnnotationSkipBankDeletion)
		} else if err := r.deleteBank(ctx, bank); err != nil {
			// The finalizer stays so the bank is not orphaned silently. Tell the
			// user why deletion is blocked and how to proceed without it.
			r.Recorder.Eventf(bank, nil, corev1.EventTypeWarning, "DeleteFailed", "Delete",
				"cannot delete bank %q: %v. Restore access, or set the %s=true annotation to keep the bank and finish deletion.",
				bank.EffectiveBankID(), err, AnnotationSkipBankDeletion)
			return err
		}
	}
	controllerutil.RemoveFinalizer(bank, finalizer)
	return r.Update(ctx, bank)
}

func (r *HindsightBankReconciler) deleteBank(ctx context.Context, bank *hindsightv1alpha1.HindsightBank) error {
	api, err := r.apiClient(ctx, bank)
	if err != nil {
		return err
	}
	if err := api.DeleteBank(ctx, bank.EffectiveBankID()); err != nil {
		return err
	}
	log.FromContext(ctx).Info("deleted bank", "bank", bank.EffectiveBankID())
	return nil
}

func (r *HindsightBankReconciler) apiClient(ctx context.Context, bank *hindsightv1alpha1.HindsightBank) (*hindsight.Client, error) {
	var apiKey string
	if ref := bank.Spec.Connection.APIKeySecretRef; ref != nil {
		var secret corev1.Secret
		key := types.NamespacedName{Namespace: bank.Namespace, Name: ref.Name}
		if err := r.APIReader.Get(ctx, key, &secret); err != nil {
			return nil, fmt.Errorf("read Secret %s: %w", ref.Name, err)
		}
		value, ok := secret.Data[ref.Key]
		if !ok {
			return nil, fmt.Errorf("secret %s has no key %q", ref.Name, ref.Key)
		}
		apiKey = strings.TrimSpace(string(value))
	}
	return hindsight.NewClient(bank.Spec.Connection.URL, apiKey, r.HTTPClient), nil
}

func (r *HindsightBankReconciler) ready(ctx context.Context, bank *hindsightv1alpha1.HindsightBank, message string) (ctrl.Result, error) {
	r.setReady(bank, metav1.ConditionTrue, ReasonSynced, message)
	return ctrl.Result{RequeueAfter: r.ResyncPeriod}, r.Status().Update(ctx, bank)
}

func (r *HindsightBankReconciler) failAPI(ctx context.Context, bank *hindsightv1alpha1.HindsightBank, err error) (ctrl.Result, error) {
	var apiErr *hindsight.APIError
	if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusBadRequest {
		return r.fail(ctx, bank, ReasonInvalidTemplate, err, false)
	}
	permanent := errors.As(err, &apiErr) && apiErr.Permanent()
	return r.fail(ctx, bank, ReasonAPIError, err, !permanent)
}

// fail records the error on the Ready condition. Transient errors are
// returned so the controller retries with backoff; permanent ones wait for a
// spec change or the next resync.
func (r *HindsightBankReconciler) fail(ctx context.Context, bank *hindsightv1alpha1.HindsightBank, reason string, cause error, transient bool) (ctrl.Result, error) {
	r.setReady(bank, metav1.ConditionFalse, reason, cause.Error())
	r.Recorder.Eventf(bank, nil, corev1.EventTypeWarning, reason, "Reconcile", "%s", cause.Error())
	if err := r.Status().Update(ctx, bank); err != nil && !apierrors.IsNotFound(err) {
		return ctrl.Result{}, err
	}
	if transient {
		return ctrl.Result{}, cause
	}
	return ctrl.Result{RequeueAfter: r.ResyncPeriod}, nil
}

func (r *HindsightBankReconciler) setReady(bank *hindsightv1alpha1.HindsightBank, status metav1.ConditionStatus, reason, message string) {
	bank.Status.ObservedGeneration = bank.Generation
	meta.SetStatusCondition(&bank.Status.Conditions, metav1.Condition{
		Type:               ConditionReady,
		Status:             status,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: bank.Generation,
	})
}

func describe(changes []string, created bool) string {
	if created {
		return "the full template (new bank)"
	}
	return strings.Join(changes, ", ")
}

// SetupWithManager registers the controller. Status-only updates do not
// trigger a reconcile; resyncs come from RequeueAfter.
func (r *HindsightBankReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&hindsightv1alpha1.HindsightBank{}, builder.WithPredicates(
			predicate.Or(predicate.GenerationChangedPredicate{}, predicate.AnnotationChangedPredicate{}),
		)).
		Named("hindsightbank").
		Complete(r)
}
