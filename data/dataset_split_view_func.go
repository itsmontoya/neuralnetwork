package data

// DatasetSplitViewFunc consumes temporary paired train and test views
// synchronously. Neither view nor its matrices may be retained, mutated, or
// used concurrently.
type DatasetSplitViewFunc func(train, test *DatasetView) (err error)
