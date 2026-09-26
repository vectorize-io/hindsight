// Package controller reconciles HindsightBank resources.
package controller

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	hindsightv1alpha1 "github.com/vectorize-io/hindsight/hindsight-operator/api/v1alpha1"
	"github.com/vectorize-io/hindsight/hindsight-operator/internal/hindsight"
)

// ConditionReady is True when the last import succeeded.
const ConditionReady = "Ready"

// HindsightBankReconciler imports each HindsightBank's template into Hindsight.
type HindsightBankReconciler struct {
	client.Client
	// APIReader reads Secrets without caching every Secret in the cluster.
	APIReader  client.Reader
	HTTPClient *http.Client
	// ResyncPeriod is how often the template is imported again, which restores
	// settings changed outside Kubernetes.
	ResyncPeriod time.Duration
}

// +kubebuilder:rbac:groups=hindsight.vectorize.io,resources=hindsightbanks,verbs=get;list;watch
// +kubebuilder:rbac:groups=hindsight.vectorize.io,resources=hindsightbanks/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get

// Reconcile imports the template. Import creates a missing bank and skips
// unchanged mental models and directives, so it is safe to repeat.
func (r *HindsightBankReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var bank hindsightv1alpha1.HindsightBank
	if err := r.Get(ctx, req.NamespacedName, &bank); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	err := r.apply(ctx, &bank)
	status, reason, message := metav1.ConditionTrue, "Imported", "template imported"
	if err != nil {
		status, reason, message = metav1.ConditionFalse, "ImportFailed", err.Error()
	}
	bank.Status.ObservedGeneration = bank.Generation
	meta.SetStatusCondition(&bank.Status.Conditions, metav1.Condition{
		Type:               ConditionReady,
		Status:             status,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: bank.Generation,
	})
	if updateErr := r.Status().Update(ctx, &bank); updateErr != nil {
		return ctrl.Result{}, updateErr
	}
	// A rejected template will not succeed until the spec changes.
	var apiErr *hindsight.APIError
	if err != nil && !(errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusBadRequest) {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: r.ResyncPeriod}, nil
}

func (r *HindsightBankReconciler) apply(ctx context.Context, bank *hindsightv1alpha1.HindsightBank) error {
	var apiKey string
	if ref := bank.Spec.Connection.APIKeySecretRef; ref != nil {
		var secret corev1.Secret
		if err := r.APIReader.Get(ctx, types.NamespacedName{Namespace: bank.Namespace, Name: ref.Name}, &secret); err != nil {
			return fmt.Errorf("read Secret %s: %w", ref.Name, err)
		}
		value, ok := secret.Data[ref.Key]
		if !ok {
			return fmt.Errorf("secret %s has no key %q", ref.Name, ref.Key)
		}
		apiKey = strings.TrimSpace(string(value))
	}
	api := hindsight.NewClient(bank.Spec.Connection.URL, apiKey, r.HTTPClient)
	return api.Import(ctx, bank.EffectiveBankID(), bank.Spec.Template.Raw)
}

// SetupWithManager registers the controller. Status updates do not trigger a
// reconcile; resyncs come from RequeueAfter.
func (r *HindsightBankReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&hindsightv1alpha1.HindsightBank{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Named("hindsightbank").
		Complete(r)
}
