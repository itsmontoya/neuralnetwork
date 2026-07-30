package data

// SequenceDatasetViewFunc consumes a temporary aligned SequenceDatasetView
// synchronously. The view, its matrices, and its length slice must not be
// retained, mutated, or used concurrently.
type SequenceDatasetViewFunc func(view *SequenceDatasetView) (err error)
