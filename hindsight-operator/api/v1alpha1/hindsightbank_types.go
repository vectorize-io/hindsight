package v1alpha1

import (
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SecretKeyRef selects a key of a Secret in the resource's namespace.
type SecretKeyRef struct {
	// Name of the Secret.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`
	// Key within the Secret that holds the value.
	// +kubebuilder:validation:MinLength=1
	Key string `json:"key"`
}

// Connection tells the operator how to reach the Hindsight API.
type Connection struct {
	// URL is the base URL of the Hindsight API, for example
	// http://hindsight-api.hindsight.svc.cluster.local:8888.
	// +kubebuilder:validation:Pattern=`^https?://`
	URL string `json:"url"`

	// APIKeySecretRef selects the API key sent as a bearer token. Omit it when
	// the API does not require authentication.
	// +optional
	APIKeySecretRef *SecretKeyRef `json:"apiKeySecretRef,omitempty"`
}

// HindsightBankSpec defines the desired state of a Hindsight bank.
// +kubebuilder:validation:XValidation:rule="has(self.bankId) == has(oldSelf.bankId) && (!has(self.bankId) || self.bankId == oldSelf.bankId)",message="bankId is immutable"
type HindsightBankSpec struct {
	// BankID is the Hindsight bank to manage. Defaults to metadata.name.
	// +optional
	// +kubebuilder:validation:MinLength=1
	BankID string `json:"bankId,omitempty"`

	// Connection to the Hindsight API.
	Connection Connection `json:"connection"`

	// Template is a Hindsight bank template manifest, the format that
	// GET /v1/default/banks/{bank_id}/export returns and
	// POST /v1/default/banks/{bank_id}/import accepts. Deleting the resource
	// keeps the bank.
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Schemaless
	// +kubebuilder:validation:Type=object
	Template apiextensionsv1.JSON `json:"template"`
}

// HindsightBankStatus is the observed state of a HindsightBank.
type HindsightBankStatus struct {
	// ObservedGeneration is the most recent generation the operator acted on.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions describe the current state. Ready is True when the last
	// import succeeded.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// HindsightBank declares the configuration, summary definitions
// (mental models), and directives of one Hindsight memory bank.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=hsbank
// +kubebuilder:printcolumn:name="Bank",type=string,JSONPath=`.spec.bankId`
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
type HindsightBank struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   HindsightBankSpec   `json:"spec"`
	Status HindsightBankStatus `json:"status,omitempty"`
}

// EffectiveBankID returns spec.bankId, or metadata.name when it is unset.
func (b *HindsightBank) EffectiveBankID() string {
	if b.Spec.BankID != "" {
		return b.Spec.BankID
	}
	return b.Name
}

// HindsightBankList contains a list of HindsightBank.
// +kubebuilder:object:root=true
type HindsightBankList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []HindsightBank `json:"items"`
}

func init() {
	SchemeBuilder.Register(&HindsightBank{}, &HindsightBankList{})
}
