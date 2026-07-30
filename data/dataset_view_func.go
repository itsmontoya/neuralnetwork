package data

// DatasetViewFunc consumes a temporary paired DatasetView synchronously. The
// view and its matrices must not be retained, mutated, or used concurrently.
type DatasetViewFunc func(view *DatasetView) (err error)
