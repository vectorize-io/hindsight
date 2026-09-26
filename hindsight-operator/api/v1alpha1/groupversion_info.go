// Package v1alpha1 contains the v1alpha1 API of the Hindsight operator.
// +kubebuilder:object:generate=true
// +groupName=hindsight.vectorize.io
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	// GroupVersion is the group and version used to register these objects.
	GroupVersion = schema.GroupVersion{Group: "hindsight.vectorize.io", Version: "v1alpha1"}

	// SchemeBuilder adds the types in this package to a scheme.
	SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}

	// AddToScheme adds the types in this group-version to the given scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)
