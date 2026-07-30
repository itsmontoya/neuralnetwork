package model

import "github.com/itsmontoya/neuralnetwork/data"

// ViewFitConfig configures explicitly opt-in view-backed ordinary fitting.
//
// FitConfig retains all ordinary fit semantics. Policy controls whether
// shuffled, non-contiguous training may use the documented copied fallback.
// Dataset views are scoped read-only aliases and must not be retained,
// mutated, or used concurrently.
type ViewFitConfig struct {
	FitConfig FitConfig
	Policy    data.ViewPolicy
}
