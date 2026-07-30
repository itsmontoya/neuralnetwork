package model

import "github.com/itsmontoya/neuralnetwork/data"

// SequenceViewFitConfig configures explicitly opt-in aligned sequence fitting.
//
// SequenceFitConfig retains all length-aware fit semantics. Policy controls
// whether shuffled, non-contiguous training may use the documented copied
// fallback. Input, target, and logical-length views remain one scoped
// read-only association and must not be retained, mutated, or used
// concurrently.
type SequenceViewFitConfig struct {
	SequenceFitConfig SequenceFitConfig
	Policy            data.ViewPolicy
}
